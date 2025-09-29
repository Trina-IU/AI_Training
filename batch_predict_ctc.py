"""
Batch prediction for CRNN+CTC model.
Usage:
  python batch_predict_ctc.py --model ./best_crnn_ctc.pth --input ./test_folder --output ./preds.csv --device cpu --recursive

Outputs a CSV with columns: filepath,prediction
Handles Unicode filenames on Windows and matches the default target size (64x256) unless overridden.
"""
import argparse
from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

# Recreate CRNN model from training
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
        conv = nn.functional.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        return out


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    img = cv2.imread(str(path), flags)
    if img is not None:
        return img
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def preprocess(img, target_h=64, target_w=256):
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor


def greedy_decode(logits, idx_to_char):
    probs = logits.softmax(2).cpu().numpy()
    seq = probs.argmax(axis=2).squeeze(1)
    res = []
    prev = -1
    for s in seq:
        if s != prev and s != 0:
            res.append(idx_to_char[s])
        prev = s
    return ''.join(res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True, help='Folder containing images')
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-h', type=int, default=64)
    p.add_argument('--target-w', type=int, default=256)
    p.add_argument('--recursive', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.model, map_location='cpu')
    idx_to_char = ckpt['idx_to_char']
    model = CRNN(len(idx_to_char))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    input_path = Path(args.input)
    if not input_path.exists():
        print('Input path does not exist:', input_path)
        sys.exit(1)

    if args.recursive:
        files = list(input_path.rglob('*.png'))
    else:
        files = list(input_path.glob('*.png'))

    print(f'Found {len(files)} PNGs to predict')

    with open(args.output, 'w', encoding='utf-8', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['filepath', 'prediction'])
        for i, f in enumerate(files, 1):
            img = imread_unicode(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f'WARN: failed to read {f}')
                writer.writerow([str(f), ''])
                continue
            tensor = preprocess(img, target_h=args.target_h, target_w=args.target_w).to(device)
            with torch.no_grad():
                logits = model(tensor)
            pred = greedy_decode(logits, idx_to_char)
            writer.writerow([str(f), pred])
            if i % 100 == 0:
                print(f'Processed {i}/{len(files)}')

    print('Done. Predictions written to', args.output)

if __name__ == '__main__':
    main()
