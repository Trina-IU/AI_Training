"""
Export trained CRNN/CRNN+Attention (CTC) to a TorchScript module for Android.
Usage:
  python export_model.py best_crnn_ctc_or_attention.pth exported_model.pt

The exported module accepts FloatTensor [1,1,H,W] and returns LongTensor of label indices.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Minimal CRNN same as training
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
        conv = nn.functional.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = nn.functional.relu(out)
        return out


class AttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = nn.functional.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        return x * attention


class CRNNWithAttention(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ResidualBlock(512),
        )
        self.attention = AttentionModule(512)
        self.pool = nn.MaxPool2d((2, 1), (2, 1))
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256 * 2, n_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = self.attention(conv)
        conv = self.pool(conv)
        conv = nn.functional.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        return out

class ExportWrapper(nn.Module):
    def __init__(self, model, idx_to_char):
        super().__init__()
        self.model = model
        self.idx_to_char = idx_to_char

    def forward(self, x):
        # x: [1,1,H,W]
        logits = self.model(x)  # [T, N, C]
        probs = torch.softmax(logits, dim=2)
        seq = torch.argmax(probs, dim=2).squeeze(1)  # [T]
        # collapse repeats and remove blanks -> return indices as LongTensor
        prev = -1
        out_list: List[int] = []  # type: ignore[name-defined]
        for i in range(seq.size(0)):
            s = int(seq[i].item())
            if s != prev and s != 0:
                out_list.append(s)
            prev = s
        if len(out_list) == 0:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor(out_list, dtype=torch.int64)


def main():
    if len(sys.argv) < 3:
        print('Usage: python export_model.py <checkpoint.pth> <out.pt>')
        return
    ckpt = torch.load(sys.argv[1], map_location='cpu')
    idx_to_char = ckpt['idx_to_char']
    # Normalize idx_to_char to list for JSON portability
    if isinstance(idx_to_char, dict):
        # Ensure order by key index
        keys = sorted(idx_to_char.keys(), key=lambda k: int(k) if isinstance(k, str) else k)
        idx_to_char = [idx_to_char[k] for k in keys]

    n_classes = len(idx_to_char)

    # Pick architecture matching the checkpoint
    model_name = ckpt.get('model_name', '')
    state_dict = ckpt['model_state_dict']
    use_attention = ('attention' in model_name) or any(k.startswith('attention.') for k in state_dict.keys())

    if use_attention:
        model = CRNNWithAttention(n_classes)
    else:
        model = CRNN(n_classes)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    wrapper = ExportWrapper(model, {i: ch for i, ch in enumerate(idx_to_char)})
    scripted = torch.jit.script(wrapper)
    scripted.save(sys.argv[2])
    # Also write idx_to_char.json next to the scripted model for decoding on-device
    import json
    out_path = Path(sys.argv[2])
    json_path = out_path.with_suffix('.chars.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump({i: ch for i, ch in enumerate(idx_to_char)}, jf, ensure_ascii=False)

    print('Saved scripted model to', sys.argv[2])
    print('Saved character map to', str(json_path))

if __name__ == '__main__':
    main()
