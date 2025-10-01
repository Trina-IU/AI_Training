"""
Batch prediction script for CRNN with Attention model
Compatible with models trained using ocr_ctc_attention.py
"""
import argparse
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """Read image with Unicode path support"""
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels, 7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(channels, 1, 7, padding=3, bias=False)
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool_cat = torch.cat([avg_pool, max_pool], dim=1)
        
        attention = self.conv1(pool_cat)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)
        
        return x * attention


class CRNNWithAttention(nn.Module):
    def __init__(self, n_classes, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # CNN backbone with residual blocks
        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(channels[0], channels[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            ResidualBlock(channels[1], channels[1]),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(channels[1], channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            ResidualBlock(channels[2], channels[2]),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(channels[2], channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True),
            ResidualBlock(channels[3], channels[3]),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        
        # Attention mechanism
        self.attention = AttentionModule(channels[3])
        
        # RNN
        self.rnn = nn.LSTM(channels[3], 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        
        # Output layer
        self.fc = nn.Linear(256 * 2, n_classes)
    
    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        
        # Apply attention
        conv = self.attention(conv)
        
        # Adaptive pooling to ensure consistent height
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        
        # Reshape for RNN
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        
        # RNN processing
        out, _ = self.rnn(conv)
        
        # Output projection
        out = self.fc(out)
        
        # CTC format: [T, N, C]
        out = out.permute(1, 0, 2)
        return out


def preprocess(image_path, target_size=(64, 256)):
    """Preprocess image for model input"""
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = (img - 0.5) / 0.5
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img


def greedy_decode(logits, idx_to_char):
    """CTC greedy decoding"""
    probs = logits.softmax(2).cpu().numpy()
    seq = probs.argmax(axis=2).squeeze(1)  # [T]
    
    # Collapse repeats and remove blanks (0)
    result = []
    prev = -1
    for s in seq:
        if s != prev and s != 0:
            result.append(idx_to_char.get(s, '?'))
        prev = s
    
    return ''.join(result)


def predict_batch(model, image_paths, idx_to_char, device, target_size=(64, 256)):
    """Predict text for batch of images"""
    results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Preprocess
            img_tensor = preprocess(img_path, target_size)
            if img_tensor is None:
                results.append({
                    'filename': img_path.name,
                    'predicted_text': '[ERROR: Could not read image]'
                })
                continue
            
            img_tensor = img_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                logits = model(img_tensor)
            
            # Decode
            text = greedy_decode(logits, idx_to_char)
            
            results.append({
                'filename': img_path.name,
                'predicted_text': text
            })
        
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'predicted_text': f'[ERROR: {str(e)}]'
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch prediction for CRNN with Attention')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--input', required=True, help='Input directory containing images')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--target-h', type=int, default=64, help='Target image height')
    parser.add_argument('--target-w', type=int, default=256, help='Target image width')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)
    
    # Get character mapping
    idx_to_char = checkpoint['idx_to_char']
    n_classes = len(idx_to_char)
    
    print(f"Model info:")
    print(f"  - Vocabulary size: {n_classes}")
    print(f"  - Device: {args.device}")
    
    # Initialize model
    model = CRNNWithAttention(n_classes).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Get image files
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Supported image extensions
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    
    if args.recursive:
        image_files = [f for ext in extensions for f in input_dir.rglob(f'*{ext}')]
    else:
        image_files = [f for ext in extensions for f in input_dir.glob(f'*{ext}')]
    
    print(f"\nFound {len(image_files)} images in {input_dir}")
    
    if len(image_files) == 0:
        print("No images found! Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .tif")
        return
    
    # Run predictions
    print("\nStarting batch prediction...")
    results = predict_batch(
        model, 
        image_files, 
        idx_to_char, 
        args.device,
        target_size=(args.target_h, args.target_w)
    )
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'predicted_text'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Results saved to: {output_path}")
    print(f"📊 Processed {len(results)} images")
    
    # Show sample results
    print("\n📋 Sample predictions (first 5):")
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. {result['filename']}: \"{result['predicted_text']}\"")


if __name__ == '__main__':
    main()
