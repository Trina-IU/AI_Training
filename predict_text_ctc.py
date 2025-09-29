import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import numpy as np


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

# Reconstruct minimal CRNN from ocr_ctc.py
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
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        return out


def preprocess(path, target_size=(64,256)):
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = (img - 0.5) / 0.5
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img


def greedy_decode(logits, idx_to_char):
    # logits: [T, N=1, C]
    probs = logits.softmax(2).cpu().numpy()
    seq = probs.argmax(axis=2).squeeze(1)  # [T]
    # collapse repeats and remove blanks(0)
    res = []
    prev = -1
    for s in seq:
        if s != prev and s != 0:
            res.append(idx_to_char[s])
        prev = s
    return ''.join(res)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('image_path')
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-h', type=int, default=64)
    p.add_argument('--target-w', type=int, default=256)
    args = p.parse_args()

    ckpt = torch.load(args.model_path, map_location=args.device)
    idx_to_char = ckpt['idx_to_char']
    model = CRNN(len(idx_to_char)).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    img = preprocess(args.image_path, target_size=(args.target_h, args.target_w)).to(args.device)
    with torch.no_grad():
        logits = model(img)
    text = greedy_decode(logits, idx_to_char)
    print(text)
