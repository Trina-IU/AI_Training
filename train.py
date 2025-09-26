import csv
import cv2
import os
import numpy as np
from pathlib import Path


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

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if video_fps <= 0:
        # fallback if FPS couldn't be read
        video_fps = 30.0

    frame_interval = max(1, int(round(video_fps / float(max(1, fps)))))

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


if __name__ == "__main__":
    # Example usage (uncomment and adapt paths):
    # process_video("videos/amoxicillin.mp4", "dataset/amoxicillin", fps=1, target_size=(32,128), augment=2)
    pass
