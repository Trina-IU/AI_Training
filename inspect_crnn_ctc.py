"""
inspect_crnn_ctc.py

Quick inspector for a CRNN+CTC checkpoint and a single image. Prints
logit shapes, non-blank timesteps, top-k probabilities for first timesteps,
and the greedy-decoded text. Uses a Unicode-safe imread fallback for Windows.

Usage:
  python inspect_crnn_ctc.py --checkpoint best_crnn_ctc.pth --image "test\test1.png" [--device cpu|cuda]
"""
import argparse
import sys
from pathlib import Path
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    # Try normal imread first (works for ASCII paths), fallback for Windows/Unicode paths
    img = cv2.imread(str(path), flags)
    if img is None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, flags)
        except Exception:
            img = None
    return img


# CRNN matching the one used for export_model.py
class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1))
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256*2, n_classes)

    def forward(self, x):
        conv = self.cnn(x)
        # collapse height dimension
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)   # [T, N, C]
        return out


def decode_mapping(ckpt):
    raw = ckpt.get('idx_to_char', None)
    if raw is None:
        raise KeyError("Checkpoint doesn't include 'idx_to_char'")
    # Accept either list (index -> char) or dict (possibly string keys)
    if isinstance(raw, list):
        return {i: raw[i] for i in range(len(raw))}
    if isinstance(raw, dict):
        conv = {}
        for k, v in raw.items():
            try:
                ik = int(k)
            except Exception:
                raise ValueError(f"Non-integer key in idx_to_char: {k}")
            conv[int(ik)] = v
        return conv
    raise ValueError("Unsupported idx_to_char format in checkpoint")


def topk_for_timestep(probs_t, k=3):
    topk = torch.topk(probs_t, k)
    vals = topk.values.cpu().numpy().tolist()
    inds = topk.indices.cpu().numpy().tolist()
    return list(zip(inds, vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', '-c', required=True)
    p.add_argument('--image', '-i', required=True)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    p.add_argument('--show-topk', type=int, default=3, help='how many top classes to show per timestep')
    p.add_argument('--timesteps-to-show', type=int, default=8, help='show topk for first N timesteps')
    args = p.parse_args()

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    try:
        idx_to_char = decode_mapping(ckpt)
    except Exception as e:
        print('Error reading idx_to_char from checkpoint:', e)
        print('Checkpoint keys:', list(ckpt.keys()))
        sys.exit(2)

    n_classes = len(idx_to_char)
    print(f"Loaded checkpoint '{args.checkpoint}'. Vocab size (incl blank) = {n_classes}")

    model = CRNN(n_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    img_path = Path(args.image)
    img = imread_unicode(img_path, flags=cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERROR: couldn't read image:", img_path)
        sys.exit(2)
    print("Read OK:", img_path, "shape:", img.shape, "dtype:", img.dtype)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_f = img.astype(np.float32) / 255.0
    img_f = (img_f - 0.5) / 0.5
    tensor = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)   # [T, N, C]
        logits_cpu = logits.detach().cpu()
        T, N, C = logits_cpu.shape
        print("Logits shape (T, N, C):", logits_cpu.shape)

        probs = torch.softmax(logits_cpu, dim=2)   # [T, N, C]
        top1 = torch.argmax(probs, dim=2)          # [T, N]

        seq = top1[:, 0].cpu().numpy().tolist()   # length T

        non_blank = sum(1 for s in seq if int(s) != 0)
        print(f"Non-blank timesteps: {non_blank} / {T}")

        # Greedy decode
        decoded = []
        prev = -1
        for s in seq:
            s_int = int(s)
            if s_int != prev and s_int != 0:
                ch = idx_to_char.get(s_int, '?')
                decoded.append(ch)
            prev = s_int
        decoded_text = ''.join(decoded)
        print("Greedy-decoded text:", repr(decoded_text))

        # Show top-k at first few timesteps
        k = args.show_topk
        Nshow = min(args.timesteps_to_show, T)
        print(f"\nTop {k} classes for first {Nshow} timesteps (index,prob, char):")
        for t in range(Nshow):
            row = probs[t, 0, :]
            tk = topk_for_timestep(row, k)
            items = []
            for idx, val in tk:
                ch = idx_to_char.get(int(idx), '?')
                items.append(f"({int(idx)},{val:.3f},{repr(ch)})")
            print(f"t={t:02d}: " + " ".join(items))

        blank_prob = float(probs[0, 0, 0].item())
        print(f"\nBlank (index 0) prob at t0: {blank_prob:.4f}")


if __name__ == '__main__':
    main()
