import argparse
from pathlib import Path
import cv2
import torch
import torch.nn as nn
import numpy as np

# Minimal model reconstruction helpers copied to match train.py variants
class ImprovedSimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)

class EnhancedCNN(nn.Module):
    # Lightweight reimplementation of the structure used in train.py for inference
    def __init__(self, n_classes, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class HandwritingCRNN(nn.Module):
    def __init__(self, n_classes, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        output, _ = self.rnn(conv)
        output = self.classifier(output)
        output = output.mean(dim=1)
        return output


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    labels = ckpt.get('labels')
    model_name = ckpt.get('model_name', 'enhanced_cnn')

    if model_name == 'enhanced_cnn':
        model = EnhancedCNN(len(labels))
    elif model_name == 'crnn':
        model = HandwritingCRNN(len(labels))
    else:
        model = ImprovedSimpleCNN(len(labels))

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, labels


def preprocess_image(img_path, target_size=(64,256)):
    # robust read for Windows/non-ascii filenames
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        try:
            data = np.fromfile(str(img_path), dtype=np.uint8)
            if data.size == 0:
                raise RuntimeError(f"Failed to read image {img_path}")
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            raise RuntimeError(f"Failed to read image {img_path}")
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0
    img = (img - 0.5) / 0.5
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('image_path')
    p.add_argument('--device', default='cpu')
    p.add_argument('--target-size', nargs=2, type=int, default=[64,256])
    args = p.parse_args()

    device = torch.device(args.device)
    model, labels = load_model_from_checkpoint(args.model_path, device)

    img = preprocess_image(args.image_path, target_size=tuple(args.target_size))
    img = img.to(device)

    with torch.no_grad():
        out = model(img)
        probs = nn.functional.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_label = labels[pred_idx]

    print(f"Predicted: {pred_label} (index {pred_idx})")
    # Print top-5 probabilities
    topk = 5
    inds = probs.argsort()[::-1][:topk]
    print("Top predictions:")
    for i in inds:
        print(f"  {labels[i]}: {probs[i]:.4f}")

if __name__ == '__main__':
    main()
