"""
End-to-End Document OCR Pipeline: YOLO Detection + CRNN Recognition

This script combines:
1. YOLO: Detects text regions in document images
2. CRNN with Attention: Recognizes text in detected regions

Perfect for processing doctor's handwritten prescriptions where you need to:
- Find where the handwriting is on the page
- Read what the handwriting says

Usage:
  # Process a single document
  python pipeline_yolo_ocr.py --yolo-weights best_yolo.pt --ocr-weights best_crnn_attention.pth --image prescription.jpg --output results/

  # Process a folder of documents
  python pipeline_yolo_ocr.py --yolo-weights best_yolo.pt --ocr-weights best_crnn_attention.pth --image-dir ./documents --output results/

  # Visualize results with bounding boxes and recognized text
  python pipeline_yolo_ocr.py --yolo-weights best_yolo.pt --ocr-weights best_crnn_attention.pth --image prescription.jpg --output results/ --visualize
"""

import argparse
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# --------------------- CRNN Model Definition (must match training) ---------------------

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        out = x * attention
        return out


class ResidualBlock(nn.Module):
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


class CRNNWithAttention(nn.Module):
    def __init__(self, n_classes):
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
        conv = F.adaptive_avg_pool2d(conv, (1, conv.size(3)))
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)
        out, _ = self.rnn(conv)
        out = self.fc(out)
        out = out.permute(1, 0, 2)
        return out


# Fallback simple CRNN (if user trained with original ocr_ctc.py)
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


# --------------------- Pipeline Functions ---------------------

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Read image robustly on Windows."""
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


def preprocess_for_detection(img):
    """Enhance image contrast and stroke visibility to improve text detection.
    This is helpful for faint handwriting or low-contrast scans.
    Returns a 3-channel BGR image suitable for YOLO input.
    """
    try:
        # Convert to gray and apply CLAHE (adaptive histogram equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Adaptive threshold to emphasize strokes
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 9)

        # Morphological closing to connect thin strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Convert back to BGR
        colored = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
        # Return preprocessed BGR image plus intermediates for debugging
        return colored, enhanced, thresh, closed
    except Exception:
        # On error return original image and empty intermediates
        return img, None, None, None


def load_ocr_model(checkpoint_path, device='cpu'):
    """Load CRNN model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    idx_to_char = ckpt['idx_to_char']
    n_classes = len(idx_to_char)

    # Try to load with attention model first
    model_name = ckpt.get('model_name', 'crnn_ctc')

    try:
        if 'attention' in model_name:
            model = CRNNWithAttention(n_classes)
        else:
            model = CRNN(n_classes)
        model.load_state_dict(ckpt['model_state_dict'])
    except Exception as e:
        print(f"Warning: Failed to load with {model_name}, trying fallback CRNN")
        model = CRNN(n_classes)
        model.load_state_dict(ckpt['model_state_dict'])

    model.to(device)
    model.eval()

    return model, idx_to_char


def preprocess_crop_for_ocr(crop, target_size=(64, 256)):
    """Preprocess a detected crop for OCR."""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    crop = cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5
    tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return tensor


def greedy_decode(logits, idx_to_char):
    """Greedy CTC decoding."""
    # logits: [T, N, C]
    probs = torch.softmax(logits, dim=2)
    top1 = torch.argmax(probs, dim=2)  # [T, N]
    seq = top1[:, 0].cpu().numpy().tolist()

    # Collapse repeats and remove blanks
    decoded = []
    prev = -1
    for s in seq:
        s_int = int(s)
        if s_int != prev and s_int != 0:
            ch = idx_to_char.get(s_int if isinstance(list(idx_to_char.keys())[0], int) else str(s_int), '?')
            decoded.append(ch)
        prev = s_int

    return ''.join(decoded)


def detect_and_recognize(image_path, yolo_model, ocr_model, idx_to_char, device='cpu',
                        conf_threshold=0.25, iou_threshold=0.45, imgsz=640,
                        target_size=(64, 256), preprocess=False, output_dir=None):
    """
    Main pipeline: detect text regions with YOLO, then recognize with CRNN.

    Returns:
        List of dicts: [{'box': [x1,y1,x2,y2], 'text': 'recognized text', 'confidence': float}, ...]
    """
    # Read image
    img = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return []

    # Optional preprocessing to enhance faint handwriting
    img_for_detect = img
    enhanced = thresh = closed = None
    
    if preprocess:
        processed = preprocess_for_detection(img)
        if isinstance(processed, tuple) and len(processed) == 4:
            img_for_detect, enhanced, thresh, closed = processed
        else:
            img_for_detect = processed
            enhanced = thresh = closed = None

        # Ensure img_for_detect is a valid numpy array for YOLO
        if img_for_detect is None or not isinstance(img_for_detect, np.ndarray):
            print("Warning: preprocessing failed, using original image")
            img_for_detect = img
            enhanced = thresh = closed = None

        # Save debug images if output_dir provided
        if output_dir is not None:
            debug_dir = Path(output_dir) / 'debug'
            debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                if enhanced is not None:
                    cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_enhanced_gray.jpg"), enhanced)
                if thresh is not None:
                    cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_thresh.jpg"), thresh)
                if closed is not None:
                    cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_closed.jpg"), closed)
                if img_for_detect is not None:
                    cv2.imwrite(str(debug_dir / f"{Path(image_path).stem}_preprocessed_for_detect.jpg"), img_for_detect)
            except Exception as e:
                print(f"Warning: could not save debug images: {e}")

    # Step 1: Detect text regions with YOLO
    results = yolo_model.predict(img_for_detect, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        print(f"No text regions detected in {image_path}")
        return []

    detections = []

    # Step 2: For each detected region, run OCR
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        # Extract crop
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocess for OCR
        crop_tensor = preprocess_crop_for_ocr(crop, target_size).to(device)

        # Run OCR
        with torch.no_grad():
            logits = ocr_model(crop_tensor)  # [T, N, C]

        # Decode
        text = greedy_decode(logits, idx_to_char)

        detections.append({
            'box': [x1, y1, x2, y2],
            'text': text,
            'confidence': conf
        })

    return detections


def visualize_results(image_path, detections, output_path):
    """Draw bounding boxes and recognized text on image."""
    img = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return

    for det in detections:
        x1, y1, x2, y2 = det['box']
        text = det['text']
        conf = det['confidence']

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw text background
        label = f"{text} ({conf:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), (0, 255, 0), -1)

        # Draw text
        cv2.putText(img, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imwrite(str(output_path), img)
    print(f"Saved visualization to {output_path}")


def process_image(image_path, yolo_model, ocr_model, idx_to_char, output_dir,
                 device='cpu', visualize=False, conf_threshold=0.25, iou_threshold=0.45, imgsz=640, preprocess=False):
    """Process a single image through the pipeline."""
    print(f"\nProcessing: {image_path}")

    detections = detect_and_recognize(
        image_path, yolo_model, ocr_model, idx_to_char,
        device=device, conf_threshold=conf_threshold, iou_threshold=iou_threshold, imgsz=imgsz,
        preprocess=preprocess, output_dir=output_dir
    )

    # Save results as JSON
    result = {
        'image': str(image_path),
        'detections': detections,
        'num_detections': len(detections)
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{Path(image_path).stem}_result.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Found {len(detections)} text region(s)")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. '{det['text']}' (conf: {det['confidence']:.2f})")

    # Visualize if requested
    if visualize:
        vis_path = output_path / f"{Path(image_path).stem}_visual.jpg"
        visualize_results(image_path, detections, vis_path)

    return result


def main():
    parser = argparse.ArgumentParser(description='YOLO + CRNN OCR Pipeline')
    parser.add_argument('--yolo-weights', required=True, help='Path to trained YOLO weights (.pt)')
    parser.add_argument('--ocr-weights', required=True, help='Path to trained CRNN weights (.pth)')
    parser.add_argument('--image', help='Path to single image')
    parser.add_argument('--image-dir', help='Path to directory of images')
    parser.add_argument('--output', default='./pipeline_results', help='Output directory')
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO IoU threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO image size')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    parser.add_argument('--target-h', type=int, default=64, help='OCR target height')
    parser.add_argument('--target-w', type=int, default=256, help='OCR target width')
    parser.add_argument('--preprocess', action='store_true', help='Apply image enhancement before YOLO detection')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Error: Must specify --image or --image-dir")
        return

    # Check YOLO availability
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load models
    print(f"\nLoading YOLO model from {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)

    print(f"Loading OCR model from {args.ocr_weights}")
    ocr_model, idx_to_char = load_ocr_model(args.ocr_weights, device=device)
    print(f"OCR vocabulary size: {len(idx_to_char)}")

    # Process images
    results = []

    if args.image:
        # Debug: show process_image signature and origin
        try:
            import inspect
            print('DEBUG: process_image signature ->', inspect.signature(process_image))
            print('DEBUG: process_image defined in ->', process_image.__code__.co_filename, 'line', process_image.__code__.co_firstlineno)
        except Exception as _:
            pass

        result = process_image(
            args.image, yolo_model, ocr_model, idx_to_char, args.output,
            device=device, visualize=args.visualize, conf_threshold=args.conf, iou_threshold=args.iou, imgsz=args.imgsz,
            preprocess=args.preprocess
        )
        results.append(result)

    elif args.image_dir:
        img_dir = Path(args.image_dir)
        image_files = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg'))
        print(f"\nFound {len(image_files)} images in {img_dir}")

        for img_path in image_files:
            # Debug: show process_image signature and origin
            try:
                import inspect
                print('DEBUG: process_image signature ->', inspect.signature(process_image))
                print('DEBUG: process_image defined in ->', process_image.__code__.co_filename, 'line', process_image.__code__.co_firstlineno)
            except Exception:
                pass

            result = process_image(
                img_path, yolo_model, ocr_model, idx_to_char, args.output,
                device=device, visualize=args.visualize, conf_threshold=args.conf, iou_threshold=args.iou, imgsz=args.imgsz,
                preprocess=args.preprocess
            )
            results.append(result)

    # Save summary
    summary_path = Path(args.output) / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Pipeline complete!")
    print(f"Results saved to: {args.output}")
    print(f"Summary: {summary_path}")

    total_detections = sum(r['num_detections'] for r in results)
    print(f"Total images processed: {len(results)}")
    print(f"Total text regions detected: {total_detections}")


if __name__ == '__main__':
    main()
