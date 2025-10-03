"""
Enhanced CRNN with Attention Mechanism for Handwritten OCR

This is an improved version of the CRNN model specifically designed for
challenging handwritten text (like doctor's prescriptions).

Key improvements:
1. Attention mechanism to focus on relevant parts of text
2. ResNet-inspired CNN backbone for better feature extraction
3. Enhanced data augmentation for robustness
4. Better handling of varying text lengths

Usage (same as ocr_ctc.py):
  python ocr_ctc_attention.py --dataset ./dataset --epochs 100 --batch-size 16 --device cuda
"""

import argparse
from pathlib import Path
import csv
import random
import time
import sys

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Read image robustly on Windows with non-ASCII paths."""
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
    chars = sorted(chars)
    idx_to_char = ['<blank>'] + chars
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


def augment_handwriting(img):
    """
    Enhanced augmentation specifically for handwritten text.
    Simulates variations in doctor's handwriting.
    """
    # Random rotation
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    # Random perspective transform
    if random.random() < 0.2:
        h, w = img.shape
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dx = random.randint(-5, 5)
        dy = random.randint(-5, 5)
        pts2 = np.float32([[dx, dy], [w-dx, dy], [dx, h-dy], [w-dx, h-dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=255)

    # Random brightness/contrast
    if random.random() < 0.3:
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random erosion/dilation (simulate pen thickness variations)
    if random.random() < 0.2:
        kernel = np.ones((2, 2), np.uint8)
        if random.random() < 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)

    # Random Gaussian noise
    if random.random() < 0.2:
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    return img


class OCRCTCDataset(Dataset):
    def __init__(self, items, char_map, target_size=(64, 256), augment=False):
        self.items = items
        self.char_map = char_map
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def encode_transcript(self, s):
        seq = [self.char_map[c] for c in s if c in self.char_map]
        return torch.LongTensor(seq)

    def __getitem__(self, idx):
        path, transcript = self.items[idx]
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Failed to read " + path)

        # Apply augmentation
        if self.augment:
            img = augment_handwriting(img)

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

    _, _, H, W = imgs.shape
    T = W // downsample_factor
    input_lengths = torch.LongTensor([T] * batch_size)

    return imgs, targets_concat, input_lengths, target_lengths


# --------------------- Attention Module ---------------------

class AttentionModule(nn.Module):
    """
    Spatial attention mechanism to focus on relevant parts of handwriting.
    Particularly useful for messy or overlapping text.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)  # [B, 1, H, W]

        # Apply attention
        out = x * attention
        return out


# --------------------- Residual Block for Better Features ---------------------

class ResidualBlock(nn.Module):
    """Residual block inspired by ResNet for better feature learning."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


# --------------------- Enhanced CRNN with Attention ---------------------

class CRNNWithAttention(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Enhanced CNN backbone with residual connections
        self.cnn = nn.Sequential(
            # Stage 1
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2

            # Stage 2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),  # /4

            # Stage 3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.MaxPool2d((2, 1), (2, 1)),  # /8 height, /4 width

            # Stage 4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
        )

        # Attention mechanism
        self.attention = AttentionModule(512)

        # Additional pooling
        self.pool = nn.MaxPool2d((2, 1), (2, 1))

        # Bidirectional LSTM for sequence modeling
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)

        # Output layer
        self.fc = nn.Linear(256 * 2, n_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        conv = self.cnn(x)  # [B, 512, H', W']

        # Apply attention
        conv = self.attention(conv)

        # Additional pooling
        conv = self.pool(conv)

        # Collapse height dimension
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)  # [B, W', C]

        # RNN
        out, _ = self.rnn(conv)  # [B, W', 512]

        # Output
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
    print(f"Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")

    train_ds = OCRCTCDataset(train_items, char_map, target_size=(args.target_h, args.target_w), augment=True)
    val_ds = OCRCTCDataset(val_items, char_map, target_size=(args.target_h, args.target_w), augment=False)

    collate = lambda b: ctc_collate(b, downsample_factor=4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate, num_workers=0, pin_memory=True)

    # Device selection
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    model = CRNNWithAttention(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss = 1e9
    start_epoch = 1
    no_improve = 0

    # Resume from checkpoint if provided
    if getattr(args, 'resume_from', None):
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print('Resuming from', str(resume_path))
            ck = torch.load(str(resume_path), map_location='cpu')
            if 'model_state_dict' in ck:
                model.load_state_dict(ck['model_state_dict'])
            if 'optimizer_state_dict' in ck:
                try:
                    optimizer.load_state_dict(ck['optimizer_state_dict'])
                except Exception:
                    print('Warning: failed to fully load optimizer state')
            if 'scheduler_state_dict' in ck:
                try:
                    scheduler.load_state_dict(ck['scheduler_state_dict'])
                except Exception:
                    pass
            if 'val_loss' in ck:
                best_loss = ck.get('val_loss', best_loss)
            if 'epoch' in ck:
                start_epoch = int(ck.get('epoch', 0)) + 1
            print(f"  start_epoch={start_epoch}, best_loss={best_loss:.4f}")
        else:
            print('Warning: resume-from path not found:', str(resume_path))

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            total_loss = 0.0
            batch_idx = 0

            for imgs, targets, input_lengths, target_lengths in train_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(imgs)
                log_probs = logits.log_softmax(2)

                loss = ctc(log_probs, targets, input_lengths, target_lengths)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping")
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_idx += 1

                # Periodic checkpoint
                if args.checkpoint_batches and args.checkpoint_batches > 0 and (batch_idx % args.checkpoint_batches) == 0:
                    last_ckp = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'char_map': char_map,
                        'idx_to_char': idx_to_char,
                        'val_loss': best_loss,
                        'epoch': epoch,
                        'batch': batch_idx,
                        'model_name': 'crnn_attention'
                    }
                    try:
                        torch.save(last_ckp, str(Path(args.save_path).with_suffix('.last.pth')))
                        print(f"  Saved periodic checkpoint at epoch {epoch} batch {batch_idx}")
                    except Exception:
                        pass

            avg_loss = total_loss / max(batch_idx, 1)
            t1 = time.time()
            print(f"Epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f} time={t1-t0:.1f}s")

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for imgs, targets, input_lengths, target_lengths in val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                    logits = model(imgs)
                    log_probs = logits.log_softmax(2)
                    loss = ctc(log_probs, targets, input_lengths, target_lengths)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss += loss.item()
                        val_batches += 1

            val_loss = val_loss / max(val_batches, 1)
            print(f"  Validation loss: {val_loss:.4f}")

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save last checkpoint every epoch
            last_ck = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'char_map': char_map,
                'idx_to_char': idx_to_char,
                'val_loss': val_loss,
                'epoch': epoch,
                'model_name': 'crnn_attention'
            }
            try:
                torch.save(last_ck, str(Path(args.save_path).with_suffix('.last.pth')))
            except Exception:
                pass

            # Save best
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                best_ck = last_ck.copy()
                try:
                    torch.save(best_ck, args.save_path)
                    print(f"  ✓ Saved best model to {args.save_path}")
                except Exception as e:
                    print('Warning: failed to save best model:', e)
            else:
                no_improve += 1
                print(f"  No improvement for {no_improve} epoch(s)")

            # Early stopping
            if getattr(args, 'patience', None) is not None and args.patience > 0:
                if no_improve >= args.patience:
                    print(f"Stopping early after {no_improve} epochs with no improvement")
                    break

    except KeyboardInterrupt:
        print('\nTraining interrupted. Saving checkpoint...')
        last_ck = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'char_map': char_map,
            'idx_to_char': idx_to_char,
            'val_loss': best_loss,
            'epoch': epoch,
            'model_name': 'crnn_attention'
        }
        try:
            torch.save(last_ck, str(Path(args.save_path).with_suffix('.last.pth')))
            print('Saved to', str(Path(args.save_path).with_suffix('.last.pth')))
        except Exception as e:
            print('Failed to save:', e)
        raise
    except Exception:
        print('\nException during training. Saving checkpoint...')
        last_ck = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'char_map': char_map,
            'idx_to_char': idx_to_char,
            'val_loss': best_loss,
            'epoch': locals().get('epoch', -1),
            'model_name': 'crnn_attention'
        }
        try:
            torch.save(last_ck, str(Path(args.save_path).with_suffix('.last.pth')))
            print('Saved to', str(Path(args.save_path).with_suffix('.last.pth')))
        except Exception as e:
            print('Failed to save:', e)
        raise

    print('Training complete!')
    print(f'Best validation loss: {best_loss:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to dataset root')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-h', type=int, default=64)
    p.add_argument('--target-w', type=int, default=256)
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--test-split', type=float, default=0.15)
    p.add_argument('--save-path', default='best_crnn_attention.pth')
    p.add_argument('--resume-from', default=None)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--checkpoint-batches', type=int, default=0)
    args = p.parse_args()

    train(args)
