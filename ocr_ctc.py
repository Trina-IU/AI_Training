"""
Minimal CRNN + CTC training script for handwritten OCR.
Usage (example):
  python ocr_ctc.py --dataset .\dataset --epochs 50 --batch-size 32 --lr 1e-3 --device cpu

Assumptions:
 - Each label folder has a labels.csv with rows: filename,transcript
 - Transcripts are plain UTF-8 strings (spaces allowed). We build a char vocabulary from them.
 - Images are grayscale PNGs.

This script is kept intentionally small and pragmatic for getting a working OCR pipeline.
"""
import argparse
from pathlib import Path
import csv
import re
import random
import math
import time

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------- Dataset & utilities ---------------------

def build_index(root: Path):
    items = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        csvf = sub / 'labels.csv'
        if csvf.exists():
            with open(csvf, newline='', encoding='utf-8') as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if not row:
                        continue
                    fname = row[0]
                    transcript = row[1] if len(row) > 1 else Path(fname).stem
                    path = sub / fname
                    if path.exists():
                        items.append((str(path), transcript))
        else:
            for img in sub.glob('*.png'):
                items.append((str(img), sub.name))
    return items


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """Read image robustly on Windows with non-ASCII paths.
    Try cv2.imread first; on failure fall back to numpy.fromfile + cv2.imdecode.
    Returns None on failure.
    """
    img = cv2.imread(path, flags)
    if img is not None:
        return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def build_charset(items):
    chars = set()
    for _, t in items:
        chars.update(list(t))
    # Reserve index 0 for CTC blank
    chars = sorted(chars)
    idx_to_char = ['<blank>'] + chars
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


class OCRCTCDataset(Dataset):
    def __init__(self, items, char_map, target_size=(64,256)):
        self.items = items
        self.char_map = char_map
        self.target_size = target_size  # (H,W)

    def __len__(self):
        return len(self.items)

    def encode_transcript(self, s):
        # map characters to ints; unknown chars get omitted
        seq = [self.char_map[c] for c in s if c in self.char_map]
        return torch.LongTensor(seq)

    def __getitem__(self, idx):
        path, transcript = self.items[idx]
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Failed to read " + path)
        img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]

        target = self.encode_transcript(transcript)
        return img, target


def ctc_collate(batch, downsample_factor=4):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    batch_size = len(imgs)
    imgs = torch.stack(imgs, dim=0)
    target_lengths = torch.LongTensor([t.numel() for t in targets])
    if target_lengths.sum().item() == 0:
        targets_concat = torch.LongTensor([])
    else:
        targets_concat = torch.cat(targets)

    # Input lengths (approx): width after CNN downsampling
    # Assume images all same width
    _, _, H, W = imgs.shape
    T = W // downsample_factor
    input_lengths = torch.LongTensor([T] * batch_size)

    return imgs, targets_concat, input_lengths, target_lengths


# --------------------- Model ---------------------
class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # simple conv feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),  # /2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),  # /4
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256*2, n_classes)

    def forward(self, x):
        # x: [B,1,H,W]
        conv = self.cnn(x)  # [B,C,H',W']
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)  # [B, W', C*H'] -> sequence
        out, _ = self.rnn(conv)  # [B, W', 512]
        out = self.fc(out)  # [B, W', n_classes]
        # CTCLoss expects [T, N, C]
        out = out.permute(1, 0, 2)
        return out


# --------------------- Training ---------------------

def train(args):
    root = Path(args.dataset)
    items = build_index(root)
    if len(items) == 0:
        raise RuntimeError('No data found in ' + str(root))

    random.shuffle(items)
    split1 = int(len(items) * (1 - args.val_split - args.test_split))
    split2 = split1 + int(len(items) * args.val_split)
    train_items = items[:split1]
    val_items = items[split1:split2]
    test_items = items[split2:]

    char_map, idx_to_char = build_charset(items)
    n_classes = len(idx_to_char)
    print(f"Vocab size (incl. blank): {n_classes} chars")

    train_ds = OCRCTCDataset(train_items, char_map, target_size=(args.target_h, args.target_w))
    val_ds = OCRCTCDataset(val_items, char_map, target_size=(args.target_h, args.target_w))

    collate = lambda b: ctc_collate(b, downsample_factor=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device(args.device)
    model = CRNN(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for imgs, targets, input_lengths, target_lengths in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(imgs)  # [T,B,C]
            log_probs = logits.log_softmax(2)

            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(train_loader) if len(train_loader)>0 else 1)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f} time={t1-t0:.1f}s")

        # Simple validation: compute loss on val set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets, input_lengths, target_lengths in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)
                logits = model(imgs)
                log_probs = logits.log_softmax(2)
                loss = ctc(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()
        val_loss = val_loss / (len(val_loader) if len(val_loader)>0 else 1)
        print(f"  Validation loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'char_map': char_map,
                'idx_to_char': idx_to_char,
                'val_loss': val_loss,
                'epoch': epoch,
                'model_name': 'crnn_ctc'
            }, args.save_path)
            print(f"  Saved best model to {args.save_path}")

    print('Training complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to dataset root')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-h', type=int, default=64)
    p.add_argument('--target-w', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--save-path', default='best_crnn_ctc.pth')
    args = p.parse_args()

    train(args)
