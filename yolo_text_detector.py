"""
YOLO-based Text Region Detection for Handwritten Documents

This script detects and localizes handwritten text regions in document images.
Outputs bounding boxes that will be fed to the CRNN OCR model.

Usage:
  # Train detector
  python yolo_text_detector.py train --data-yaml data.yaml --epochs 100 --img-size 640
  
  # Detect text regions
  python yolo_text_detector.py detect --weights best_yolo.pt --source document.jpg --output ./detected
  
  # Prepare dataset for YOLO (convert annotations to YOLO format)
  python yolo_text_detector.py prepare --input ./raw_dataset --output ./yolo_dataset
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import sys
import shutil

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


def prepare_yolo_dataset(input_dir, output_dir, train_split=0.7, val_split=0.15):
    """
    Prepare dataset for YOLO training.
    
    Expected input structure:
      input_dir/
        images/
          img1.png
          img2.png
        annotations/  (optional - if you have labeled bounding boxes)
          img1.json  # format: {"boxes": [[x1,y1,x2,y2,label], ...]}
    
    Output structure (YOLO format):
      output_dir/
        images/
          train/
          val/
          test/
        labels/
          train/
          val/
          test/
        data.yaml
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get all images
    img_dir = input_path / 'images'
    if not img_dir.exists():
        print(f"Error: {img_dir} not found")
        return
    
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg'))
    print(f"Found {len(images)} images")
    
    if len(images) == 0:
        print("No images found. Please place images in", img_dir)
        return
    
    # Shuffle and split
    import random
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    
    splits = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }
    
    ann_dir = input_path / 'annotations'
    has_annotations = ann_dir.exists()
    
    for split_name, img_list in splits.items():
        print(f"Processing {split_name}: {len(img_list)} images")
        for img_path in img_list:
            # Copy image
            dst_img = output_path / 'images' / split_name / img_path.name
            shutil.copy(str(img_path), str(dst_img))
            
            # Process annotation if exists
            if has_annotations:
                ann_path = ann_dir / f"{img_path.stem}.json"
                if ann_path.exists():
                    with open(ann_path, 'r') as f:
                        ann = json.load(f)
                    
                    # Convert to YOLO format (normalized xywh)
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                    
                    yolo_lines = []
                    for box in ann.get('boxes', []):
                        # box format: [x1, y1, x2, y2, class_id]
                        x1, y1, x2, y2 = box[:4]
                        class_id = int(box[4]) if len(box) > 4 else 0
                        
                        # Convert to YOLO format: class_id x_center y_center width height (normalized)
                        x_center = ((x1 + x2) / 2) / w
                        y_center = ((y1 + y2) / 2) / h
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    
                    # Write YOLO label file
                    label_path = output_path / 'labels' / split_name / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.writelines(yolo_lines)
                else:
                    # No annotation - create empty label file (for auto-annotation)
                    label_path = output_path / 'labels' / split_name / f"{img_path.stem}.txt"
                    label_path.touch()
            else:
                # No annotations directory - create empty labels for unsupervised/weak supervision
                label_path = output_path / 'labels' / split_name / f"{img_path.stem}.txt"
                # For text detection, we can use connected component analysis as weak supervision
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                pseudo_boxes = auto_detect_text_regions(img)
                
                h, w = img.shape
                yolo_lines = []
                for x1, y1, x2, y2 in pseudo_boxes:
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                with open(label_path, 'w') as f:
                    f.writelines(yolo_lines)
    
    # Create data.yaml
    yaml_content = f"""# YOLO Text Detection Dataset
path: {output_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1  # number of classes
names: ['text']  # class names
"""
    
    with open(output_path / 'data.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nDataset prepared successfully!")
    print(f"Output: {output_path}")
    print(f"Data config: {output_path / 'data.yaml'}")
    print(f"\nTo train YOLO:")
    print(f"  python yolo_text_detector.py train --data-yaml {output_path / 'data.yaml'}")


def auto_detect_text_regions(img, min_area=50, max_area=50000):
    """
    Automatic text region detection using connected components.
    Used as weak supervision when manual annotations are not available.
    """
    # Threshold
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to connect nearby components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if min_area < area < max_area:
            boxes.append([x, y, x+w, y+h])
    
    return boxes


def train_yolo_detector(data_yaml, epochs=100, img_size=640, batch=16, weights='yolov8n.pt', device='cpu'):
    """Train YOLO text detector."""
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return
    
    print(f"Training YOLO text detector...")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {img_size}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    
    # Load pretrained YOLO model
    model = YOLO(weights)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        project='yolo_runs',
        name='text_detector',
        patience=20,  # early stopping
        save=True,
        plots=True
    )
    
    print(f"\nTraining complete!")
    print(f"Best model saved to: yolo_runs/text_detector/weights/best.pt")
    
    return results


def detect_text_regions(model_path, source, output_dir, conf_threshold=0.25, iou_threshold=0.45, 
                        device='cpu', save_crops=True):
    """
    Detect text regions in images using trained YOLO model.
    
    Args:
        model_path: Path to trained YOLO weights (.pt file)
        source: Path to image, directory, or video
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device: 'cpu' or 'cuda'
        save_crops: If True, save cropped text regions for OCR
    """
    if not ULTRALYTICS_AVAILABLE:
        print("Error: ultralytics not installed. Install with: pip install ultralytics")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_crops:
        crops_dir = output_path / 'crops'
        crops_dir.mkdir(exist_ok=True)
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        save=True,
        project=str(output_path),
        name='detections'
    )
    
    # Process results
    detections_info = []
    
    for idx, result in enumerate(results):
        img_path = result.path
        img = cv2.imread(img_path)
        
        boxes = result.boxes
        img_detections = []
        
        for i, box in enumerate(boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            img_detections.append({
                'box': [x1, y1, x2, y2],
                'confidence': conf,
                'class': cls
            })
            
            # Save cropped region
            if save_crops:
                crop = img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_name = f"{Path(img_path).stem}_crop_{i}.png"
                    cv2.imwrite(str(crops_dir / crop_name), crop)
        
        detections_info.append({
            'image': img_path,
            'detections': img_detections
        })
    
    # Save detections as JSON
    json_path = output_path / 'detections.json'
    with open(json_path, 'w') as f:
        json.dump(detections_info, f, indent=2)
    
    print(f"\nDetection complete!")
    print(f"Results saved to: {output_path}")
    print(f"Detections JSON: {json_path}")
    if save_crops:
        print(f"Cropped text regions: {crops_dir}")
    
    return detections_info


def main():
    parser = argparse.ArgumentParser(description='YOLO Text Region Detector')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Prepare dataset command
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset for YOLO training')
    prepare_parser.add_argument('--input', required=True, help='Input directory with images')
    prepare_parser.add_argument('--output', required=True, help='Output directory for YOLO dataset')
    prepare_parser.add_argument('--train-split', type=float, default=0.7)
    prepare_parser.add_argument('--val-split', type=float, default=0.15)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train YOLO text detector')
    train_parser.add_argument('--data-yaml', required=True, help='Path to data.yaml')
    train_parser.add_argument('--epochs', type=int, default=100)
    train_parser.add_argument('--img-size', type=int, default=640)
    train_parser.add_argument('--batch', type=int, default=16)
    train_parser.add_argument('--weights', default='yolov8n.pt', help='Pretrained weights')
    train_parser.add_argument('--device', default='cpu', help='cpu or cuda')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect text regions')
    detect_parser.add_argument('--weights', required=True, help='Path to trained YOLO weights')
    detect_parser.add_argument('--source', required=True, help='Image/directory/video source')
    detect_parser.add_argument('--output', default='./yolo_detections', help='Output directory')
    detect_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    detect_parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    detect_parser.add_argument('--device', default='cpu', help='cpu or cuda')
    detect_parser.add_argument('--no-save-crops', action='store_true', help='Do not save cropped regions')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_yolo_dataset(args.input, args.output, args.train_split, args.val_split)
    
    elif args.command == 'train':
        train_yolo_detector(
            args.data_yaml,
            epochs=args.epochs,
            img_size=args.img_size,
            batch=args.batch,
            weights=args.weights,
            device=args.device
        )
    
    elif args.command == 'detect':
        detect_text_regions(
            args.weights,
            args.source,
            args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            save_crops=not args.no_save_crops
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
