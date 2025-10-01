import csv
import cv2
import os
import numpy as np
from pathlib import Path
import argparse
import random
import math
from collections import Counter

# Optional imports for training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
except Exception:
    torch = None


def _augment_image(img):
    """Enhanced augmentations for handwriting images."""
    out = [img]

    # Rotation augmentations (more angles for better generalization)
    for ang in (-15, -10, -5, 5, 10, 15):
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), ang, 1.0)
        rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
        out.append(rot)

    # Perspective transformations (simulate different viewing angles)
    h, w = img.shape
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    # Slight perspective distortions
    for dx, dy in [(5, 2), (-3, 4), (2, -3)]:
        pts2 = np.float32([[dx, dy], [w-dx, dy], [dx, h-dy], [w-dx, h-dy]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(img, M, (w, h), borderValue=255)
        out.append(warped)

    # Brightness/contrast variations (more realistic variations)
    for alpha, beta in [(1.2, -15), (0.8, 15), (1.1, -5), (0.9, 8)]:
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        out.append(adjusted)

    # Gaussian noise
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    out.append(noisy)

    # Morphological operations (simulate pen thickness variations)
    kernel = np.ones((2,2), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    dilated = cv2.dilate(img, kernel, iterations=1)
    out.extend([eroded, dilated])

    return out


def enhanced_preprocess_image(img, target_size=(64, 256), skip_blur=False):
    """Enhanced preprocessing specifically for handwriting recognition."""
    h, w = img.shape
    h_t, w_t = target_size

    # Adaptive thresholding for better binarization of handwriting
    # First, apply Gaussian blur to reduce noise
    if skip_blur:
        blurred = img
    else:
        blurred = cv2.GaussianBlur(img, (3, 3), 0)

    # Use adaptive threshold instead of OTSU for handwriting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize while maintaining aspect ratio
    scale = min(w_t / float(w), h_t / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(cleaned, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to fixed size with white background
    canvas = 255 * np.ones((h_t, w_t), dtype=np.uint8)
    x_offset = (w_t - new_w) // 2
    y_offset = (h_t - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def process_video(video_path, output_dir, fps=1, target_size=(64, 256),
                  sharpness_thresh=10.0, ink_ratio_range=(0.005, 0.8), augment=0,
                  dedupe=True, dedupe_hamming_thresh=12, max_frames_per_video=None, debug=False,
                  video_prefix=None, save_original=False, skip_blur=False):
    """Enhanced video processing for handwriting OCR with better preprocessing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # label inferred from folder name
    label = output_dir.name
    labels_csv = output_dir / "labels.csv"

    # Check file exists
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path} (cwd={Path.cwd()})")
        return 0

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: OpenCV couldn't open the video file: {video_path}")
        return 0
    # Get video FPS and compute sampling interval
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(video_fps / float(max(1, fps)))))
    print(f"Video opened: {video_path} fps={video_fps:.2f} sample_fps={fps} frame_interval={frame_interval}")

    count, saved = 0, 0
    writer = None
    if not labels_csv.exists():
        writer = open(labels_csv, "w", newline="", encoding="utf-8")
        csv.writer(writer).writerow(["filename", "label"])
    else:
        writer = open(labels_csv, "a", newline="", encoding="utf-8")
    csvw = csv.writer(writer)

    h_t, w_t = target_size
    last_hash = None

    # Read frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing (optionally skip the blur)
            processed = enhanced_preprocess_image(gray, target_size, skip_blur=skip_blur)

            # Optionally save the raw grayscale (resized) for human inspection
            if save_original:
                try:
                    # resize grayscale to target for consistency
                    orig = cv2.resize(gray, (w_t, h_t), interpolation=cv2.INTER_AREA)
                    orig_name = f"{vp}_orig_{saved:05d}.png"
                    orig_path = output_dir / orig_name
                    cv2.imwrite(str(orig_path), orig)
                    # Do not write to labels.csv for originals to keep training data unchanged
                except Exception:
                    pass

            # Deduplication using improved hashing
            if dedupe:
                try:
                    small = cv2.resize(processed, (8, 8), interpolation=cv2.INTER_AREA)
                    avg = small.mean()
                    bits = (small > avg).astype(np.uint8)
                    bh = 0
                    for b in bits.flatten():
                        bh = (bh << 1) | int(b)
                    if last_hash is not None:
                        ham = (last_hash ^ bh).bit_count()
                        if ham <= dedupe_hamming_thresh:
                            count += 1
                            continue
                    last_hash = bh
                except Exception:
                    pass

            # Respect per-video maximum
            if max_frames_per_video is not None and saved >= int(max_frames_per_video):
                break

            # Enhanced quality checks
            sharpness = cv2.Laplacian(processed, cv2.CV_64F).var()
            ink_ratio = float((processed < 128).mean())

            # More lenient quality checks for handwriting
            if sharpness < sharpness_thresh:
                count += 1
                continue
            if not (ink_ratio_range[0] <= ink_ratio <= ink_ratio_range[1]):
                count += 1
                continue

            # Save processed frame. Prefer an explicit video_prefix; if not provided
            # use the source video's filename stem so frames are grouped by source
            # video (avoid using the output directory name, which confuses test runs).
            try:
                src_stem = video_path.stem
            except Exception:
                src_stem = None
            if video_prefix:
                vp = video_prefix
            elif src_stem:
                vp = src_stem
            else:
                vp = output_dir.name
            base_name = f"{vp}_frame_{saved:05d}.png"
            filepath = output_dir / base_name
            cv2.imwrite(str(filepath), processed)
            csvw.writerow([str(filepath.name), label])
            saved += 1

            # Enhanced augmentations
            if augment > 0:
                aug_images = _augment_image(processed)
                # Save more augmentations but limit to prevent overfitting
                max_aug = min(augment, len(aug_images) - 1)
                for i, aug_img in enumerate(aug_images[1:max_aug+1], start=1):
                    aug_name = f"{vp}_frame_{saved:05d}_aug{i}.png"
                    aug_path = output_dir / aug_name
                    cv2.imwrite(str(aug_path), aug_img)
                    csvw.writerow([str(aug_path.name), label])
                    saved += 1

        count += 1

    cap.release()
    writer.close()
    print(f"Extracted & processed {saved} frames → {output_dir}")
    return saved


def process_videos_in_dir(input_dir, dataset_root, fps=1, augment=0, target_size=(64, 256),
                          dedupe=True, dedupe_hamming_thresh=12, max_frames_per_video=None,
                          sharpness_thresh=10.0, ink_ratio_range=(0.005, 0.8), debug=False,
                          save_original=False, skip_blur=False):
    """Process all videos with enhanced target size for handwriting."""
    input_dir = Path(input_dir)
    dataset_root = Path(dataset_root)
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    exts = {'.mp4', '.mov', '.avi', '.mkv'}
    processed = 0
    skipped = 0

    subdirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if subdirs:
        for d in subdirs:
            out = dataset_root / d.name
            out.mkdir(parents=True, exist_ok=True)
            any_video = False
            for v in sorted(d.iterdir()):
                if v.is_file() and v.suffix.lower() in exts:
                    any_video = True
                    saved = process_video(
                        v,
                        out,
                        fps=fps,
                        target_size=target_size,
                        augment=augment,
                        dedupe=dedupe,
                        dedupe_hamming_thresh=dedupe_hamming_thresh,
                        max_frames_per_video=max_frames_per_video,
                        sharpness_thresh=sharpness_thresh,
                        ink_ratio_range=ink_ratio_range,
                            debug=debug,
                            video_prefix=v.stem,
                            save_original=save_original,
                            skip_blur=skip_blur,
                    )
                    print(f"Processed {d.name}/{v.name} -> saved {saved} frames")
                    processed += 1
            if not any_video:
                print(f"No videos found in subdir: {d}")
                skipped += 1
    else:
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out = dataset_root / p.stem
                if out.exists() and any(out.glob('*.png')):
                    print(f"Skipping {p.name}, output already exists: {out}")
                    skipped += 1
                    continue
                saved = process_video(
                    p,
                    out,
                    fps=fps,
                    target_size=target_size,
                    augment=augment,
                    dedupe=dedupe,
                    dedupe_hamming_thresh=dedupe_hamming_thresh,
                    max_frames_per_video=max_frames_per_video,
                    sharpness_thresh=sharpness_thresh,
                    ink_ratio_range=ink_ratio_range,
                    debug=debug,
                    video_prefix=p.stem,
                    save_original=save_original,
                    skip_blur=skip_blur,
                )
                print(f"Processed {p.name} -> saved {saved} frames")
                processed += 1
    print(f"process_videos_in_dir: processed={processed} skipped={skipped}")
    return processed, skipped


# Enhanced CNN architectures for handwriting OCR
class EnhancedCNN(nn.Module):
    """Enhanced CNN architecture specifically designed for handwriting OCR."""
    def __init__(self, n_classes, dropout_rate=0.5):
        super().__init__()

        # Feature extraction layers with batch normalization
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class HandwritingCRNN(nn.Module):
    """CRNN (Convolutional Recurrent Neural Network) for sequence recognition."""
    def __init__(self, n_classes, hidden_size=256):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1))
        )

        # RNN layers
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # CNN features
        conv = self.cnn(x)  # [B, C, H, W]

        # Prepare for RNN: collapse height and transpose
        b, c, h, w = conv.size()
        conv = conv.view(b, c * h, w).permute(0, 2, 1)  # [B, W, C*H]

        # RNN
        output, _ = self.rnn(conv)  # [B, W, hidden_size*2]

        # Classification
        output = self.classifier(output)  # [B, W, n_classes]

        # For single character classification, take mean over sequence
        output = output.mean(dim=1)  # [B, n_classes]

        return output


# Module-level dataset classes to avoid multiprocessing pickling issues on Windows
if torch is not None:
    class OCRDataset(Dataset):
        def __init__(self, items, label_to_idx, target_size, is_training=False):
            self.items = items
            self.label_to_idx = label_to_idx
            self.target_size = target_size
            self.is_training = is_training

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {path}")

            # Ensure target size (height, width)
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                               interpolation=cv2.INTER_AREA)

            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            img = (img - 0.5) / 0.5

            if self.is_training and random.random() < 0.3:
                noise = torch.randn_like(img) * 0.1
                img = img + noise
                img = torch.clamp(img, -1, 1)

            return img, self.label_to_idx[label]

    class EvalDataset(Dataset):
        def __init__(self, items, label_to_idx, target_size):
            self.items = items
            self.label_to_idx = label_to_idx
            self.target_size = target_size

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image {path}")

            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_AREA)
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            img = (img - 0.5) / 0.5

            return img, self.label_to_idx[label], path
else:
    # Placeholders so module can be imported for preprocessing without torch
    OCRDataset = None
    EvalDataset = None


def run_training(dataset_root, epochs=50, batch_size=16, lr=1e-3, device_str="cpu",
                val_split=0.15, test_split=0.15, model_name="enhanced_cnn",
                target_size=(64, 256), early_stopping_patience=10, split_by_video=False,
                resume_from=None):
    """Enhanced training function with better hyperparameters and techniques."""
    if torch is None:
        raise RuntimeError("PyTorch not installed. Install packages from requirements.txt before training.")

    # Build dataset index
    def build_index(root):
        root = Path(root)
        items = []
        for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
            csvf = subdir / "labels.csv"
            if csvf.exists():
                with open(csvf, newline='', encoding='utf-8') as f:
                    r = csv.reader(f)
                    header = next(r, None)
                    for row in r:
                        if not row:
                            continue
                        fname = row[0]
                        label = row[1] if len(row) > 1 else subdir.name
                        path = subdir / fname
                        if path.exists():
                            items.append((str(path), label))
            else:
                for img in subdir.glob("*.png"):
                    items.append((str(img), subdir.name))
        return items

    items = build_index(dataset_root)
    if len(items) == 0:
        raise RuntimeError("No dataset images found under " + str(dataset_root))

    # Analyze class distribution
    label_counts = Counter(label for _, label in items)
    print(f"Class distribution: {dict(label_counts)}")

    # Create balanced splits. Optionally split by video (folder-level) to avoid leakage
    labels = sorted({lab for _, lab in items})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    train_items, val_items, test_items = [], [], []

    if split_by_video:
        # Group items by label -> video prefix (assumes filenames start with '<video>_')
        label_video_map = {}
        for path, label in items:
            name = Path(path).name
            # infer video id from filename prefix before first '_frame_' or first '_'
            vid = None
            if '_frame_' in name:
                vid = name.split('_frame_')[0]
            else:
                vid = name.split('_')[0]

            label_video_map.setdefault(label, {}).setdefault(vid, []).append((path, label))

        for label, vid_map in label_video_map.items():
            vids = list(vid_map.keys())
            random.shuffle(vids)
            n_vids = len(vids)
            n_test_vids = max(1, int(n_vids * test_split))
            n_val_vids = max(1, int(n_vids * val_split))
            n_train_vids = n_vids - n_val_vids - n_test_vids

            train_vids = vids[:n_train_vids]
            val_vids = vids[n_train_vids:n_train_vids + n_val_vids]
            test_vids = vids[n_train_vids + n_val_vids:]

            for v in train_vids:
                train_items.extend(vid_map[v])
            for v in val_vids:
                val_items.extend(vid_map[v])
            for v in test_vids:
                test_items.extend(vid_map[v])
    else:
        # Stratified splitting at image level
        label_items = {}
        for item in items:
            path, label = item
            label_items.setdefault(label, []).append(item)

        for label, label_item_list in label_items.items():
            random.shuffle(label_item_list)
            n = len(label_item_list)
            n_test = max(1, int(n * test_split))
            n_val = max(1, int(n * val_split))
            n_train = n - n_val - n_test

            train_items.extend(label_item_list[:n_train])
            val_items.extend(label_item_list[n_train:n_train + n_val])
            test_items.extend(label_item_list[n_train + n_val:])

    print(f"Dataset: {len(items)} images — train={len(train_items)}, val={len(val_items)}, test={len(test_items)} — classes={len(labels)}")

    # Create datasets using module-level classes (avoids Windows multiprocessing pickling issues)
    train_ds = OCRDataset(train_items, label_to_idx, target_size, is_training=True)
    val_ds = OCRDataset(val_items, label_to_idx, target_size, is_training=False)
    test_ds = OCRDataset(test_items, label_to_idx, target_size, is_training=False)

    device = torch.device(device_str)
    print(f"Using device: {device}")

    # DataLoader worker/pin settings: on Windows use num_workers=0 to avoid spawn/pickle issues
    is_windows = os.name == 'nt'
    num_workers = 0 if is_windows else 2
    pin_memory = True if (device.type == 'cuda') else False

    # Enhanced data loaders with safer defaults
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    # Build enhanced model
    if model_name == "enhanced_cnn":
        model = EnhancedCNN(len(labels), dropout_rate=0.4).to(device)
    elif model_name == "crnn":
        model = HandwritingCRNN(len(labels), hidden_size=256).to(device)
    elif model_name == "resnet18":
        try:
            from torchvision import models
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, len(labels))
            model = model.to(device)
        except Exception:
            print("Warning: torchvision not available — falling back to enhanced_cnn")
            model = EnhancedCNN(len(labels)).to(device)
    else:
        # Fallback to simple CNN but improved
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

        model = ImprovedSimpleCNN(len(labels)).to(device)

    # Enhanced optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Avoid passing verbose to keep compatibility with older PyTorch versions
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    print(f"Starting training with {model.__class__.__name__} for {epochs} epochs...")

    # Resume support: if a checkpoint path is provided, attempt to restore model/optimizer/scheduler
    start_epoch = 1
    if resume_from:
        try:
            print(f"Loading checkpoint to resume from: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: failed to load optimizer state: {e}")
            if 'scheduler_state_dict' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as e:
                    print(f"Warning: failed to load scheduler state: {e}")

            best_val_acc = ckpt.get('val_acc', best_val_acc)
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f"Resuming from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")
        except Exception as e:
            print(f"Failed to load checkpoint '{resume_from}': {e}. Starting from scratch.")

    for epoch in range(start_epoch, epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:3d}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Early stopping and model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'labels': labels,
                'val_acc': val_acc,
                'epoch': epoch,
                'model_name': model_name
            }, "best_handwriting_ocr_model.pth")
            print(f"New best validation accuracy: {val_acc:.4f} - Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (patience: {early_stopping_patience})")
                break

    # Final test evaluation with best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load("best_handwriting_ocr_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    test_correct = 0
    test_total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)

            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    test_acc = test_correct / test_total
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy analysis
    from collections import defaultdict
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, true in zip(predictions, true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1

    print("\nPer-class accuracy:")
    for class_idx in sorted(class_total.keys()):
        class_name = labels[class_idx]
        acc = class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0
        print(f"{class_name}: {acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

    return best_val_acc, test_acc


def dataset_summary(dataset_root):
    """Print a short summary of the dataset directory: per-label image counts and CSV checks."""
    root = Path(dataset_root)
    if not root.exists() or not root.is_dir():
        print(f"Dataset root does not exist: {root}")
        return

    total = 0
    print(f"Scanning dataset root: {root}\n")
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        pngs = list(subdir.glob('*.png'))
        csvf = subdir / 'labels.csv'
        n_png = len(pngs)
        total += n_png
        has_csv = csvf.exists()
        # Quick duplicate filename check (within folder)
        names = [p.name for p in pngs]
        dup_count = len(names) - len(set(names))
        print(f"Label: {subdir.name}: {n_png} images; labels.csv={'yes' if has_csv else 'no'}; dup_filenames={dup_count}")

    print(f"\nTotal images: {total}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced handwriting OCR preprocessing and training.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_pre = sub.add_parser("preprocess", help="Extract frames from videos into dataset folders")
    p_pre.add_argument("video", help="Path to input video file")
    p_pre.add_argument("out", help="Output directory for processed frames (folder name used as label)")
    p_pre.add_argument("--fps", type=float, default=1.0)
    p_pre.add_argument("--augment", type=int, default=5, help="Number of augmentations per frame")
    p_pre.add_argument("--target-size", nargs=2, type=int, default=[64, 256], help="Target image size (height width)")
    p_pre.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Disable deduplication during preprocessing")
    p_pre.add_argument("--dedupe-thresh", dest="dedupe_thresh", type=int, default=12, help="Hamming threshold for dedupe")
    p_pre.add_argument("--max-frames-per-video", dest="max_frames", type=int, default=None, help="Cap saved frames per video")
    p_pre.add_argument("--save-original", dest="save_original", action="store_true", help="Save raw grayscale frame alongside processed image")
    p_pre.add_argument("--skip-blur", dest="skip_blur", action="store_true", help="Skip Gaussian blur in preprocessing (crisper output)")

    p_train = sub.add_parser("train", help="Train enhanced handwriting OCR classifier")
    p_train.add_argument("dataset_root", help="Path to dataset root (each subfolder=label or CSVs)")
    p_train.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu", help="Device to use")
    p_train.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation set")
    p_train.add_argument("--test-split", type=float, default=0.15, help="Fraction for test set")
    p_train.add_argument("--model", choices=["enhanced_cnn", "crnn", "simple_cnn", "resnet18"],
                        default="enhanced_cnn", help="Model architecture to use")
    p_train.add_argument("--target-size", nargs=2, type=int, default=[64, 256], help="Target image size (height width)")
    p_train.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience")
    p_train.add_argument("--split-by-video", action="store_true", help="Split dataset by source video (avoid leakage)")
    p_train.add_argument("--resume-from", dest="resume_from", help="Path to checkpoint (.pth) to resume training from")

    p_pipe = sub.add_parser("pipeline", help="Complete pipeline: preprocess videos and train model")
    p_pipe.add_argument("input_dir", help="Directory containing videos to process")
    p_pipe.add_argument("dataset_root", help="Directory where processed dataset will be created")
    p_pipe.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate")
    p_pipe.add_argument("--augment", type=int, default=5, help="Number of augmentations per frame")
    p_pipe.add_argument("--target-size", nargs=2, type=int, default=[64, 256], help="Target image size (height width)")
    p_pipe.add_argument("--no-dedupe", dest="dedupe", action="store_false", help="Disable deduplication")
    p_pipe.add_argument("--dedupe-thresh", dest="dedupe_thresh", type=int, default=12, help="Hamming threshold for dedupe")
    p_pipe.add_argument("--max-frames-per-video", dest="max_frames", type=int, default=None, help="Cap frames per video")
    p_pipe.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p_pipe.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    p_pipe.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_pipe.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu", help="Device to use")
    p_pipe.add_argument("--val-split", type=float, default=0.15, help="Fraction for validation set")
    p_pipe.add_argument("--test-split", type=float, default=0.15, help="Fraction for test set")
    p_pipe.add_argument("--model", choices=["enhanced_cnn", "crnn", "simple_cnn", "resnet18"],
                        default="enhanced_cnn", help="Model architecture to use")
    p_pipe.add_argument("--early-stopping", type=int, default=10, help="Early stopping patience")
    p_pipe.add_argument("--force", action="store_true", help="Force overwrite existing dataset")
    p_pipe.add_argument("--sharpness-thresh", type=float, default=10.0, help="Sharpness threshold (lower = more lenient)")
    p_pipe.add_argument("--ink-ratio-min", type=float, default=0.005, help="Minimum ink ratio")
    p_pipe.add_argument("--ink-ratio-max", type=float, default=0.8, help="Maximum ink ratio")
    p_pipe.add_argument("--split-by-video", action="store_true", help="Split dataset by source video (avoid leakage)")
    p_pipe.add_argument("--save-original", dest="save_original", action="store_true", help="Save raw grayscale frame alongside processed image")
    p_pipe.add_argument("--skip-blur", dest="skip_blur", action="store_true", help="Skip Gaussian blur in preprocessing (crisper output)")
    p_pipe.add_argument("--resume-from", dest="resume_from", help="Path to checkpoint (.pth) to resume training from")

    p_eval = sub.add_parser("evaluate", help="Evaluate a trained model on test data")
    p_eval.add_argument("model_path", help="Path to saved model (.pth file)")
    p_eval.add_argument("dataset_root", help="Path to dataset root")
    p_eval.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    p_eval.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu", help="Device to use")
    p_eval.add_argument("--target-size", nargs=2, type=int, default=[64, 256], help="Target image size (height width)")

    p_test = sub.add_parser("test-video", help="Test video processing on a single video with verbose output")
    p_test.add_argument("video", help="Path to input video file")
    p_test.add_argument("out", help="Output directory for processed frames")
    p_test.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate")
    p_test.add_argument("--target-size", nargs=2, type=int, default=[64, 256], help="Target image size (height width)")
    p_test.add_argument("--sharpness-thresh", type=float, default=5.0, help="Sharpness threshold (lower = more lenient)")
    p_test.add_argument("--ink-ratio-min", type=float, default=0.001, help="Minimum ink ratio")
    p_test.add_argument("--ink-ratio-max", type=float, default=0.9, help="Maximum ink ratio")
    p_test.add_argument("--max-frames", type=int, default=50, help="Maximum frames to process for testing")
    p_test.add_argument("--save-original", dest="save_original", action="store_true", help="Save raw grayscale frame alongside processed image")
    p_test.add_argument("--skip-blur", dest="skip_blur", action="store_true", help="Skip Gaussian blur in preprocessing (crisper output)")

    p_ds = sub.add_parser("dataset-summary", help="Show per-label image counts and quick dataset checks")
    p_ds.add_argument("dataset_root", help="Path to dataset root to summarize")

    args = parser.parse_args()

    # Handle target_size conversion
    if hasattr(args, 'target_size') and args.target_size:
        target_size = tuple(args.target_size)
    else:
        target_size = (64, 256)

    if args.cmd == "preprocess":
        # Use the video filename stem as the prefix so frames are grouped by source video
        video_prefix = Path(args.video).stem
        saved_frames = process_video(
            args.video, args.out,
            fps=args.fps,
            target_size=target_size,
            augment=args.augment,
            dedupe=args.dedupe,
            dedupe_hamming_thresh=args.dedupe_thresh,
            max_frames_per_video=args.max_frames,
            video_prefix=video_prefix,
            save_original=getattr(args, 'save_original', False),
            skip_blur=getattr(args, 'skip_blur', False),
        )
        print(f"Preprocessing completed. Saved {saved_frames} frames.")

    elif args.cmd == "train":
        print("Starting enhanced handwriting OCR training...")
        print(f"Using model: {args.model}")
        print(f"Target image size: {target_size}")
        print(f"Device: {args.device}")

        best_val_acc, test_acc = run_training(
            args.dataset_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            val_split=args.val_split,
            test_split=args.test_split,
            model_name=args.model,
            target_size=target_size,
            early_stopping_patience=args.early_stopping,
            split_by_video=getattr(args, 'split_by_video', False),
            resume_from=getattr(args, 'resume_from', None)
        )

        print(f"\n=== Training Complete ===")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")

    elif args.cmd == "pipeline":
        print("Starting complete handwriting OCR pipeline...")

        # Clear dataset if force flag is set
        if args.force:
            import shutil
            dataset_path = Path(args.dataset_root)
            if dataset_path.exists():
                print(f"Removing existing dataset: {dataset_path}")
                shutil.rmtree(dataset_path)

        # Preprocessing phase
        print("\n=== Phase 1: Video Preprocessing ===")
        processed, skipped = process_videos_in_dir(
            args.input_dir,
            args.dataset_root,
            fps=args.fps,
            augment=args.augment,
            target_size=target_size,
            dedupe=args.dedupe,
            dedupe_hamming_thresh=args.dedupe_thresh,
            max_frames_per_video=args.max_frames,
            sharpness_thresh=args.sharpness_thresh,
            ink_ratio_range=(args.ink_ratio_min, args.ink_ratio_max),
            debug=False,
            save_original=getattr(args, 'save_original', False),
            skip_blur=getattr(args, 'skip_blur', False),
        )

        if processed == 0:
            print("No videos processed. Exiting pipeline.")
            exit(1)

        print(f"Preprocessing complete: {processed} videos processed, {skipped} skipped")

        # Training phase
        print("\n=== Phase 2: Model Training ===")
        print(f"Using model: {args.model}")
        print(f"Target image size: {target_size}")
        print(f"Device: {args.device}")

        best_val_acc, test_acc = run_training(
            args.dataset_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device_str=args.device,
            val_split=args.val_split,
            test_split=args.test_split,
            model_name=args.model,
            target_size=target_size,
            early_stopping_patience=args.early_stopping,
            split_by_video=getattr(args, 'split_by_video', False),
            resume_from=getattr(args, 'resume_from', None)
        )

        print(f"\n=== Pipeline Complete ===")
        print(f"Videos processed: {processed}")
        print(f"Best Validation Accuracy: {best_val_acc:.4f}")
        print(f"Final Test Accuracy: {test_acc:.4f}")
        print(f"Model saved as: best_handwriting_ocr_model.pth")

    elif args.cmd == "evaluate":
        print("Evaluating trained model...")

        if torch is None:
            raise RuntimeError("PyTorch not installed.")

        # Load model
        checkpoint = torch.load(args.model_path, map_location=args.device)
        labels = checkpoint['labels']
        model_name = checkpoint.get('model_name', 'enhanced_cnn')

        # Recreate model architecture
        if model_name == "enhanced_cnn":
            model = EnhancedCNN(len(labels))
        elif model_name == "crnn":
            model = HandwritingCRNN(len(labels))
        else:
            # Fallback
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

            model = ImprovedSimpleCNN(len(labels))

        model.load_state_dict(checkpoint['model_state_dict'])
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()

        print(f"Model loaded: {model_name}")
        print(f"Number of classes: {len(labels)}")
        print(f"Classes: {labels}")

        # Build test dataset (using all data as test for evaluation)
        def build_index(root):
            root = Path(root)
            items = []
            for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
                csvf = subdir / "labels.csv"
                if csvf.exists():
                    with open(csvf, newline='', encoding='utf-8') as f:
                        r = csv.reader(f)
                        header = next(r, None)
                        for row in r:
                            if not row:
                                continue
                            fname = row[0]
                            label = row[1] if len(row) > 1 else subdir.name
                            path = subdir / fname
                            if path.exists():
                                items.append((str(path), label))
                else:
                    for img in subdir.glob("*.png"):
                        items.append((str(img), subdir.name))
            return items

        items = build_index(args.dataset_root)
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        class EvalDataset(Dataset):
            def __init__(self, items, label_to_idx, target_size):
                self.items = items
                self.label_to_idx = label_to_idx
                self.target_size = target_size

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                path, label = self.items[idx]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read image {path}")

                img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                               interpolation=cv2.INTER_AREA)
                img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
                img = (img - 0.5) / 0.5

                return img, self.label_to_idx[label], path

        eval_ds = EvalDataset(items, label_to_idx, target_size)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

        # Evaluation
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        misclassified = []

        with torch.no_grad():
            for data, target, paths in eval_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)

                correct += pred.eq(target).sum().item()
                total += target.size(0)

                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

                # Track misclassified samples
                for i, (p, t, path) in enumerate(zip(pred.cpu(), target.cpu(), paths)):
                    if p != t:
                        misclassified.append((path, labels[t], labels[p]))

        accuracy = correct / total
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{total})")

        # Per-class accuracy
        from collections import defaultdict
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for pred, true in zip(predictions, true_labels):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1

        print("\nPer-class accuracy:")
        for class_idx in sorted(class_total.keys()):
            class_name = labels[class_idx]
            acc = class_correct[class_idx] / class_total[class_idx] if class_total[class_idx] > 0 else 0
            print(f"{class_name}: {acc:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")

        # Show some misclassified examples
        if misclassified:
            print(f"\nSample misclassifications (showing first 10 of {len(misclassified)}):")
            for path, true_label, pred_label in misclassified[:10]:
                print(f"  {Path(path).name}: True={true_label}, Predicted={pred_label}")

    elif args.cmd == "test-video":
        print("Testing video processing with verbose output...")
        target_size = tuple(args.target_size) if hasattr(args, 'target_size') and args.target_size else (64, 256)

        saved_frames = process_video(
            args.video, args.out,
            fps=args.fps,
            target_size=target_size,
            sharpness_thresh=args.sharpness_thresh,
            ink_ratio_range=(args.ink_ratio_min, args.ink_ratio_max),
            augment=0,  # No augmentation for testing
            dedupe=False,  # No deduplication for testing
            max_frames_per_video=args.max_frames,
            debug=True,  # Enable debug output for test-video
            save_original=getattr(args, 'save_original', False),
            skip_blur=getattr(args, 'skip_blur', False),
        )
        print(f"Test completed. Saved {saved_frames} frames to {args.out}")

        # Show some sample images if any were saved
        out_path = Path(args.out)
        sample_images = list(out_path.glob("*.png"))[:5]
        if sample_images:
            print(f"\nSample images saved:")
            for img_path in sample_images:
                print(f"  {img_path}")
        else:
            print("\nNo images were saved. Try adjusting the thresholds:")
            print(f"  --sharpness-thresh (current: {args.sharpness_thresh}, try lower like 1.0)")
            print(f"  --ink-ratio-min (current: {args.ink_ratio_min}, try lower like 0.0001)")
            print(f"  --ink-ratio-max (current: {args.ink_ratio_max}, try higher like 0.95)")

    elif args.cmd == "dataset-summary":
        dataset_summary(args.dataset_root)

    else:
        print("No command provided or invalid command.")
        print("Available commands: preprocess, train, pipeline, evaluate, test-video")
        print("Use --help for detailed usage information.")
        print("\nExample usage:")
        print("  # Test a single video first to check if processing works:")
        print("  python train.py test-video Abbreviation/some_video.mp4 test_output/")
        print("  ")
        print("  # Run full pipeline:")
        print("  python train.py pipeline . dataset/ --model enhanced_cnn --epochs 50")
        print("  ")
        print("  # Train only:")
        print("  python train.py train dataset/ --model enhanced_cnn --batch-size 16")
        print("  ")
        print("  # Evaluate a model:")
        print("  python train.py evaluate best_handwriting_ocr_model.pth dataset/")