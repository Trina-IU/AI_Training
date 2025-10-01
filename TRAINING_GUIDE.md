# Complete Guide: YOLO + CRNN for Doctor's Handwriting OCR

## Overview

This system uses a **two-stage pipeline** for accurate handwritten text recognition:

1. **YOLO (Stage 1)**: Detects and localizes handwritten text regions in document images
2. **CRNN with Attention (Stage 2)**: Recognizes text within detected regions

This is the industry-standard approach for document OCR, especially for challenging handwriting like medical prescriptions.

---

## Why This Architecture?

### YOLO for Text Detection
- Finds WHERE text is located in the document
- Handles multiple text regions (medication names, dosages, instructions)
- Robust to document layout variations
- Real-time processing capability

### CRNN with Attention for Recognition
- Reads WHAT the text says
- Attention mechanism focuses on relevant parts of messy handwriting
- Handles variable-length text sequences
- Residual connections improve feature learning

---

## Installation

### 1. Install Dependencies

```powershell
# Core dependencies
pip install torch torchvision opencv-python numpy

# For YOLO (ultralytics YOLOv8)
pip install ultralytics

# Optional: GPU support (if you have NVIDIA GPU with CUDA)
# Follow instructions at: https://pytorch.org/get-started/locally/
```

### 2. Verify Installation

```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
```

---

## Dataset Preparation

### Option A: If You Have Labeled Bounding Boxes

Your dataset should be structured as:

```
raw_dataset/
  images/
    prescription1.jpg
    prescription2.jpg
    ...
  annotations/
    prescription1.json
    prescription2.json
    ...
```

**Annotation format** (`prescription1.json`):
```json
{
  "boxes": [
    [x1, y1, x2, y2, 0],  // 0 = class "text"
    [x1, y1, x2, y2, 0],
    ...
  ]
}
```

Then run:
```powershell
python yolo_text_detector.py prepare --input ./raw_dataset --output ./yolo_dataset
```

### Option B: Auto-Detection (No Manual Labels)

If you don't have bounding box labels, the script will automatically detect text regions using connected component analysis as weak supervision:

```powershell
# Just put images in a folder
mkdir raw_dataset\images
# Copy your images there

python yolo_text_detector.py prepare --input ./raw_dataset --output ./yolo_dataset
```

This creates pseudo-labels automatically. For doctor's handwriting, manual refinement is recommended for best accuracy.

---

## Training

### Stage 1: Train YOLO Text Detector

```powershell
# Train YOLO (use GPU if available)
python yolo_text_detector.py train --data-yaml ./yolo_dataset/data.yaml --epochs 100 --img-size 640 --batch 16 --device cuda

# If no GPU:
python yolo_text_detector.py train --data-yaml ./yolo_dataset/data.yaml --epochs 100 --img-size 640 --batch 8 --device cpu
```

**Training time estimate:**
- GPU (RTX 3060): ~1-2 hours for 100 epochs
- CPU: 8-12 hours for 100 epochs

**Output:**
- Best model: `yolo_runs/text_detector/weights/best.pt`

### Stage 2: Train CRNN with Attention

Your OCR dataset should be structured as:

```
dataset/
  medication1/
    labels.csv      # filename,transcript
    image1.png
    image2.png
    ...
  medication2/
    labels.csv
    image1.png
    ...
```

**Train the enhanced CRNN:**

```powershell
# With GPU (recommended)
python ocr_ctc_attention.py --dataset ./dataset --epochs 100 --batch-size 32 --lr 1e-3 --device cuda --save-path best_crnn_attention.pth --patience 15

# With CPU (slower)
python ocr_ctc_attention.py --dataset ./dataset --epochs 100 --batch-size 16 --lr 1e-3 --device cpu --save-path best_crnn_attention.pth --patience 15
```

**Training time estimate:**
- GPU (RTX 3060): 2-4 hours for 100 epochs
- CPU: 12-24 hours for 100 epochs

**Key features:**
- Automatic data augmentation (rotation, perspective, brightness, noise)
- Early stopping (stops if no improvement for 15 epochs)
- Resume support (if training is interrupted)
- Learning rate scheduling

**Resume training if interrupted:**
```powershell
python ocr_ctc_attention.py --dataset ./dataset --epochs 100 --batch-size 32 --device cuda --save-path best_crnn_attention.pth --resume-from best_crnn_attention.last.pth --patience 15
```

---

## Inference (Using the Complete Pipeline)

### Process Single Document

```powershell
python pipeline_yolo_ocr.py --yolo-weights yolo_runs/text_detector/weights/best.pt --ocr-weights best_crnn_attention.pth --image prescription.jpg --output results/ --visualize --device cuda
```

### Process Multiple Documents

```powershell
python pipeline_yolo_ocr.py --yolo-weights yolo_runs/text_detector/weights/best.pt --ocr-weights best_crnn_attention.pth --image-dir ./prescriptions --output results/ --visualize --device cuda
```

### Output

The pipeline generates:
- `{image_name}_result.json`: Detected boxes and recognized text
- `{image_name}_visual.jpg`: Annotated image with boxes and text (if --visualize)
- `summary.json`: Complete results for all images

**Example output** (`prescription1_result.json`):
```json
{
  "image": "prescription1.jpg",
  "num_detections": 3,
  "detections": [
    {
      "box": [100, 50, 400, 120],
      "text": "Amoxicillin 500mg",
      "confidence": 0.92
    },
    {
      "box": [100, 150, 350, 200],
      "text": "3 times daily",
      "confidence": 0.87
    },
    ...
  ]
}
```

---

## Alternative: Train Only CRNN (No YOLO)

If your images are already cropped to text regions (single line/word per image), you can skip YOLO and just train CRNN:

```powershell
# Use the enhanced attention model
python ocr_ctc_attention.py --dataset ./dataset --epochs 100 --batch-size 32 --device cuda --save-path best_crnn_attention.pth

# Or use the original basic CRNN
python ocr_ctc.py --dataset ./dataset --epochs 100 --batch-size 32 --device cuda --save-path best_crnn_ctc.pth
```

For inference on cropped images:
```powershell
# Single image
python predict_text_ctc.py best_crnn_attention.pth image.png

# Batch processing
python batch_predict_ctc.py --checkpoint best_crnn_attention.pth --input-dir ./test_images --output predictions.csv
```

---

## Performance Tips

### For Better Accuracy

1. **More Training Data**
   - Aim for 1000+ labeled samples per medication/text class
   - Include variations: different doctors, lighting conditions, paper types

2. **Data Augmentation**
   - Already enabled in `ocr_ctc_attention.py`
   - Simulates rotation, perspective, brightness, pen thickness variations

3. **Fine-tuning**
   - Start with pretrained YOLO weights (already done: `yolov8n.pt`)
   - Train CRNN for 100-200 epochs
   - Use learning rate scheduling (already enabled)

4. **Ensemble Models**
   - Train multiple CRNN models with different random seeds
   - Average predictions for better robustness

### For Faster Training

1. **Use GPU**
   - 10-20x faster than CPU
   - Reduce batch size if you run out of memory

2. **Mixed Precision Training** (optional)
   - Can add to training script for ~2x speedup
   - Requires PyTorch with AMP support

3. **Reduce Image Size**
   - YOLO: Try `--img-size 416` instead of 640
   - CRNN: Try `--target-h 48 --target-w 192` instead of 64x256

---

## Troubleshooting

### "ultralytics could not be resolved"
```powershell
pip install ultralytics
```

### "CUDA out of memory"
Reduce batch size:
```powershell
# YOLO
python yolo_text_detector.py train ... --batch 8

# CRNN
python ocr_ctc_attention.py ... --batch-size 16
```

### "No text regions detected"
- Lower YOLO confidence threshold: `--conf 0.1`
- Check if YOLO model is trained properly
- Verify image quality and contrast

### Empty Predictions from CRNN
- Model is undertrained (train for more epochs)
- Check that checkpoint has been saved properly
- Verify image preprocessing matches training

### Training Interrupted
Resume from last checkpoint:
```powershell
python ocr_ctc_attention.py ... --resume-from best_crnn_attention.last.pth
```

---

## Quick Start: 3-Day Training Plan

### Day 1: Setup & YOLO Training
1. Install dependencies (30 min)
2. Prepare YOLO dataset (2 hours)
3. Start YOLO training overnight (8-12 hours)

### Day 2: CRNN Training
1. Verify YOLO model (30 min)
2. Prepare OCR dataset (2 hours)
3. Start CRNN training overnight (12-24 hours)

### Day 3: Testing & Fine-tuning
1. Run end-to-end pipeline (1 hour)
2. Evaluate results (2 hours)
3. Fine-tune if needed (4+ hours)
4. **Demo ready!**

---

## Files Created

| File | Purpose |
|------|---------|
| `yolo_text_detector.py` | YOLO training & detection |
| `ocr_ctc_attention.py` | Enhanced CRNN with attention |
| `pipeline_yolo_ocr.py` | End-to-end YOLO+CRNN pipeline |
| `ocr_ctc.py` | Original CRNN (backup) |
| `inspect_crnn_ctc.py` | Debug tool for CRNN |
| `batch_predict_ctc.py` | Batch OCR inference |

---

## Next Steps

Choose your path:

**Path A: Full Pipeline (YOLO + CRNN)**
1. Prepare YOLO dataset with document images
2. Train YOLO detector
3. Prepare OCR dataset with text crops
4. Train CRNN with attention
5. Run end-to-end pipeline

**Path B: OCR Only (if images are pre-cropped)**
1. Prepare OCR dataset
2. Train CRNN with attention
3. Run batch prediction

**Path C: Use Existing CRNN, Add YOLO**
1. Prepare YOLO dataset
2. Train YOLO detector
3. Run pipeline with existing CRNN checkpoint

---

## Questions?

Run any script with `--help` to see all options:
```powershell
python yolo_text_detector.py --help
python ocr_ctc_attention.py --help
python pipeline_yolo_ocr.py --help
```

Good luck with your demo! 🚀
