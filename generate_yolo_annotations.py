"""
Auto-generate YOLO annotations for text detection in medical prescriptions.
This script creates bounding boxes around text regions automatically.
"""

import cv2
import numpy as np
from pathlib import Path
import os

def create_text_annotations(image_path, output_label_path):
    """
    Auto-generate YOLO format annotations for text regions.
    Uses image processing to detect text areas.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Apply threshold to find text regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and create bounding boxes
    boxes = []
    min_area = (w * h) * 0.001  # Minimum 0.1% of image area
    max_area = (w * h) * 0.8    # Maximum 80% of image area

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, box_w, box_h = cv2.boundingRect(contour)

            # Convert to YOLO format (normalized center coordinates)
            center_x = (x + box_w / 2) / w
            center_y = (y + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            # Only add if box is reasonable size
            if 0.01 < norm_w < 0.95 and 0.01 < norm_h < 0.95:
                boxes.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    # If no boxes found, create a full-image box
    if not boxes:
        boxes = ["0 0.5 0.5 0.8 0.8"]  # Full image text region

    # Write YOLO annotation file
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(boxes))

    return True

def main():
    """Generate annotations for all images in train folder"""

    train_images_dir = Path("yolo_dataset/images/train")
    train_labels_dir = Path("yolo_dataset/labels/train")
    val_images_dir = Path("yolo_dataset/images/val")
    val_labels_dir = Path("yolo_dataset/labels/val")

    # Create directories
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(train_images_dir.glob(f'*{ext}')))
        image_files.extend(list(train_images_dir.glob(f'*{ext.upper()}')))

    print(f"Found {len(image_files)} images")

    if len(image_files) == 0:
        print("No images found in train directory!")
        return

    # Split images: 80% train, 20% val
    split_index = int(len(image_files) * 0.8)
    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    # Move validation images
    val_images_dir.mkdir(parents=True, exist_ok=True)
    for img_path in val_images:
        new_path = val_images_dir / img_path.name
        if not new_path.exists():
            img_path.rename(new_path)

    # Generate annotations for training images
    print(f"Processing {len(train_images)} training images...")
    success_count = 0

    for img_path in train_images:
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if create_text_annotations(img_path, label_path):
            success_count += 1
        else:
            print(f"Failed to process: {img_path}")

    # Generate annotations for validation images
    print(f"Processing {len(val_images)} validation images...")
    for img_path in val_images_dir.glob('*'):
        if img_path.suffix.lower() in image_extensions:
            label_path = val_labels_dir / f"{img_path.stem}.txt"
            if create_text_annotations(img_path, label_path):
                success_count += 1

    print(f"\n✅ Successfully generated annotations for {success_count} images")
    print(f"📁 Training images: {len(train_images)}")
    print(f"📁 Validation images: {len(val_images)}")
    print(f"📄 Dataset ready for YOLO training!")

if __name__ == "__main__":
    main()