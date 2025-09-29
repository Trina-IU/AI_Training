"""
Export trained CRNN+CTC to a TorchScript module for Android.
Usage:
  python export_model.py best_crnn_ctc.pth exported_model.pt

The exported module will accept a FloatTensor [1,1,H,W] and return a dict { 'text': str }
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
        seq = torch.argmax(probs, dim=2).squeeze(1).cpu().numpy().tolist()
        # collapse repeats and remove blanks -> return indices as LongTensor
        out_inds = []
        prev = -1
        for s in seq:
            if s != prev and s != 0:
                out_inds.append(int(s))
            prev = s
        if len(out_inds) == 0:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor(out_inds, dtype=torch.int64)


def main():
    if len(sys.argv) < 3:
        print('Usage: python export_model.py <checkpoint.pth> <out.pt>')
        return
    ckpt = torch.load(sys.argv[1], map_location='cpu')
    idx_to_char = ckpt['idx_to_char']
    model = CRNN(len(idx_to_char))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    wrapper = ExportWrapper(model, idx_to_char)
    scripted = torch.jit.script(wrapper)
    scripted.save(sys.argv[2])
    # Also write idx_to_char.json next to the scripted model for decoding on-device
    import json
    out_path = Path(sys.argv[2])
    json_path = out_path.with_suffix('.chars.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(idx_to_char, jf, ensure_ascii=False)

    print('Saved scripted model to', sys.argv[2])
    print('Saved character map to', str(json_path))

if __name__ == '__main__':
    main()
