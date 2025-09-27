import csv
import cv2
import os
import numpy as np
from pathlib import Path
import argparse
import random
import math

# Optional imports for training
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as T
except Exception:
    torch = None


def _augment_image(img):
    """Yield simple augmentations for a grayscale image (numpy array).
    Returns a list of augmented images (including the original).
    """
    out = [img]
    # small rotations
    for ang in (-5, 5):
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), ang, 1.0)
        rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)
        out.append(rot)
    # brightness/contrast (linear)
    for alpha, beta in ((1.1, -10), (0.9, 10)):
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        out.append(adjusted)
    return out


def process_video(video_path, output_dir, fps=1, target_size=(32, 128),
                  sharpness_thresh=50.0, ink_ratio_range=(0.005, 0.6), augment=0):
    """Extract frames from a video and prepare them for OCR training.

    Args:
      video_path: path to input video file.
      output_dir: directory to write processed PNGs and a labels.csv.
      fps: target frames-per-second sampling (1 means one frame per second of video).
      target_size: (height, width) tuple to resize/pad images to. Default (32,128).
      sharpness_thresh: variance of Laplacian minimum threshold to keep a frame.
      ink_ratio_range: acceptable range (min,max) of fraction of dark pixels.
      augment: number of augmentation variants to save per kept frame (0 = none).

    Notes:
      - This function assumes each call processes a single word/video pair and
        will infer the label from the output_dir basename. It appends rows to
        <output_dir>/labels.csv as (filename,label).
    """
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

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if video_fps <= 0:
        # fallback if FPS couldn't be read
        video_fps = 30.0

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            # ---- PREPROCESS ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize while keeping aspect ratio and pad to target_size
            h, w = gray.shape
            scale = min(w_t / float(w), h_t / float(h))
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Pad to fixed size (height, width)
            canvas = 255 * np.ones((h_t, w_t), dtype=np.uint8)  # white background
            x_offset = (w_t - new_w) // 2
            y_offset = (h_t - new_h) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

            # Threshold (binarization)
            _, thresh = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Basic quality checks
            sharpness = cv2.Laplacian(thresh, cv2.CV_64F).var()
            ink_ratio = float((thresh < 128).mean())
            if sharpness < sharpness_thresh:
                # too blurry
                count += 1
                continue
            if not (ink_ratio_range[0] <= ink_ratio <= ink_ratio_range[1]):
                # too empty or too full; likely not a good sample
                count += 1
                continue

            # Save processed frame and optional augmentations
            base_name = f"frame_{saved:05d}.png"
            filepath = output_dir / base_name
            cv2.imwrite(str(filepath), thresh)
            csvw.writerow([str(filepath.name), label])
            saved += 1

            if augment > 0:
                aug_images = _augment_image(thresh)
                # skip first because it's the original which we already saved
                for i, aug_img in enumerate(aug_images[1:augment+1], start=1):
                    aug_name = f"frame_{saved:05d}_aug{i}.png"
                    aug_path = output_dir / aug_name
                    cv2.imwrite(str(aug_path), aug_img)
                    csvw.writerow([str(aug_path.name), label])
                    saved += 1

        count += 1

    cap.release()
    writer.close()
    print(f"Extracted & processed {saved} frames → {output_dir}")
    return saved


def process_videos_in_dir(input_dir, dataset_root, fps=1, augment=0, target_size=(32,128)):
    """Process all videos in input_dir. Each video becomes a subfolder of dataset_root named after the filename stem."""
    input_dir = Path(input_dir)
    dataset_root = Path(dataset_root)
    if not input_dir.exists() or not input_dir.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    dataset_root.mkdir(parents=True, exist_ok=True)
    exts = {'.mp4', '.mov', '.avi', '.mkv'}
    processed = 0
    skipped = 0
    # If input_dir has subdirectories, treat each subdir as a label and process videos inside
    subdirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if subdirs:
        for d in subdirs:
            out = dataset_root / d.name
            out.mkdir(parents=True, exist_ok=True)
            any_video = False
            for v in sorted(d.iterdir()):
                if v.is_file() and v.suffix.lower() in exts:
                    any_video = True
                    saved = process_video(v, out, fps=fps, target_size=target_size, augment=augment)
                    print(f"Processed {d.name}/{v.name} -> saved {saved} frames")
                    processed += 1
            if not any_video:
                print(f"No videos found in subdir: {d}")
                skipped += 1
    else:
        # No subdirectories: treat files in input_dir as individual labeled videos
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out = dataset_root / p.stem
                # Skip if already has images
                if out.exists() and any(out.glob('*.png')):
                    print(f"Skipping {p.name}, output already exists: {out}")
                    skipped += 1
                    continue
                saved = process_video(p, out, fps=fps, target_size=target_size, augment=augment)
                print(f"Processed {p.name} -> saved {saved} frames")
                processed += 1
    print(f"process_videos_in_dir: processed={processed} skipped={skipped}")
    return processed, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video frames and/or train a simple OCR classifier.")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_pre = sub.add_parser("preprocess", help="Extract frames from videos into dataset folders")
    p_pre.add_argument("video", help="Path to input video file")
    p_pre.add_argument("out", help="Output directory for processed frames (folder name used as label)")
    p_pre.add_argument("--fps", type=float, default=1.0)
    p_pre.add_argument("--augment", type=int, default=0, help="Number of augmentations per frame")

    p_train = sub.add_parser("train", help="Train a simple classifier on a dataset root")
    p_train.add_argument("dataset_root", help="Path to dataset root (each subfolder=label or CSVs)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu")
    p_train.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation set")
    p_train.add_argument("--test-split", type=float, default=0.1, help="Fraction for test set")

    p_pipe = sub.add_parser("pipeline", help="Process all videos in a folder and then train end-to-end")
    p_pipe.add_argument("input_dir", help="Directory containing videos to process")
    p_pipe.add_argument("dataset_root", help="Directory where processed dataset subfolders will be created")
    p_pipe.add_argument("--fps", type=float, default=1.0)
    p_pipe.add_argument("--augment", type=int, default=0)
    p_pipe.add_argument("--epochs", type=int, default=10)
    p_pipe.add_argument("--batch-size", type=int, default=32)
    p_pipe.add_argument("--lr", type=float, default=1e-3)
    p_pipe.add_argument("--device", default="cuda" if torch and torch.cuda.is_available() else "cpu")
    p_pipe.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation set")
    p_pipe.add_argument("--test-split", type=float, default=0.1, help="Fraction for test set")

    args = parser.parse_args()

    if args.cmd == "preprocess":
        process_video(args.video, args.out, fps=args.fps, target_size=(32, 128), augment=args.augment)
        print("Preprocessing completed.")
        exit(0)

    def run_training(dataset_root, epochs=10, batch_size=32, lr=1e-3, device_str="cpu", val_split=0.1, test_split=0.1):
        if torch is None:
            raise RuntimeError("PyTorch not installed. Install packages from requirements.txt before training.")

        # Build dataset index
        def build_index(root):
            root = Path(root)
            items = []
            # If there are CSV files in label folders, prefer those
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
                    # fallback: all image files in folder
                    for img in subdir.glob("*.png"):
                        items.append((str(img), subdir.name))
            return items

        items = build_index(dataset_root)
        if len(items) == 0:
            raise RuntimeError("No dataset images found under " + str(dataset_root))

        # label -> idx
        labels = sorted({lab for _, lab in items})
        label_to_idx = {lab: i for i, lab in enumerate(labels)}

        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(math.floor(n * test_split)))
        n_val = max(1, int(math.floor(n * val_split)))
        n_train = n - n_val - n_test
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        print(f"Dataset: {n} images — train={len(train_items)}, val={len(val_items)}, test={len(test_items)} — classes={len(labels)}")

        class OCRDataset(Dataset):
            def __init__(self, items, label_to_idx, transform=None):
                self.items = items
                self.label_to_idx = label_to_idx
                self.transform = transform

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                path, label = self.items[idx]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise RuntimeError(f"Failed to read image {path}")
                # ensure target size
                img = cv2.resize(img, (128, 32), interpolation=cv2.INTER_AREA)
                # convert to 1xHxW float tensor
                if self.transform:
                    img = self.transform(img)
                else:
                    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
                return img, self.label_to_idx[label]

        # simple transforms
        def numpy_to_tensor(img):
            return torch.from_numpy(img).unsqueeze(0).float() / 255.0

        train_ds = OCRDataset(train_items, label_to_idx, transform=numpy_to_tensor)
        val_ds = OCRDataset(val_items, label_to_idx, transform=numpy_to_tensor)
        test_ds = OCRDataset(test_items, label_to_idx, transform=numpy_to_tensor)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        device = torch.device(device_str)

        # Simple CNN classifier
        class SimpleCNN(nn.Module):
            def __init__(self, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, n_classes)
                )

            def forward(self, x):
                return self.net(x)

        model = SimpleCNN(len(labels)).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        for epoch in range(1, epochs + 1):
            # train
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                out = model(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()
                running_loss += loss.item() * xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
            train_loss = running_loss / total
            train_acc = correct / total

            # validation
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    out = model(xb)
                    loss = crit(out, yb)
                    v_loss += loss.item() * xb.size(0)
                    preds = out.argmax(dim=1)
                    v_correct += (preds == yb).sum().item()
                    v_total += xb.size(0)
            val_loss = v_loss / max(1, v_total)
            val_acc = v_correct / max(1, v_total)

            print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # checkpoint best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model_state": model.state_dict(), "labels": labels}, "best_model.pth")

        # final test evaluation
        model.eval()
        t_correct = 0
        t_total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                preds = out.argmax(dim=1)
                t_correct += (preds == yb).sum().item()
                t_total += xb.size(0)
        print(f"Test accuracy: {t_correct}/{t_total} = {t_correct / max(1, t_total):.4f}")

        print("Training complete. Best val acc:", best_val_acc)

    # train command now calls run_training
    if args.cmd == "train":
        run_training(args.dataset_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     device_str=args.device, val_split=args.val_split, test_split=args.test_split)

    # pipeline: preprocess all videos in input_dir then train
    if args.cmd == "pipeline":
        processed, skipped = process_videos_in_dir(args.input_dir, args.dataset_root, fps=args.fps, augment=args.augment)
        if processed == 0:
            print("No videos processed. Exiting.")
        else:
            run_training(args.dataset_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                         device_str=args.device, val_split=args.val_split, test_split=args.test_split)

    else:
        print("No command provided. Use --help for usage. Example: preprocess or train")
