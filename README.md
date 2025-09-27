# AI_Training

This workspace contains a preprocessing and simple training pipeline for OCR from videos.

Files
- `train.py` - Preprocess video into image frames and a small PyTorch training flow for a simple CNN classifier.
- `requirements.txt` - Python packages required (install with pip).

Quickstart

1. Install dependencies (use a virtualenv):

```powershell
python -m pip install -r requirements.txt
```

2. Preprocess a video into a labeled folder (folder name used as label):

```powershell
python train.py preprocess videos/amoxicillin.mp4 dataset/amoxicillin --fps 1 --augment 2
```

3. Train on the dataset root (each subfolder is treated as a label)

```powershell
python train.py train dataset --epochs 10 --batch-size 32 --val-split 0.1 --test-split 0.1
```

Notes and next steps
- This repository currently provides a preprocessing pipeline and a simple classification training baseline. For a production handwriting OCR system you will want to:
  - Use a sequence model (CRNN or Transformer) and CTC loss for word/line recognition.
  - Improve augmentation, segmentation, and labeling workflows.
  - Collect diverse handwriting samples and separate writer hold-out splits for evaluation.
