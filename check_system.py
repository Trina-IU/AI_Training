"""
System Check: Verify YOLO + CRNN OCR Setup

This script checks if everything is properly installed and configured.
Run this before starting training to catch issues early.

Usage:
  python check_system.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("YOLO + CRNN OCR System Check")
print("=" * 60)
print()

all_ok = True
warnings = []

# Check Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)")
    all_ok = False

# Check PyTorch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print("   ⚠ CUDA not available (training will be slower on CPU)")
        warnings.append("No GPU detected - training will take much longer")
except ImportError:
    print("   ✗ PyTorch not installed")
    print("      Install: pip install torch torchvision")
    all_ok = False

# Check OpenCV
print("\n3. Checking OpenCV...")
try:
    import cv2
    print(f"   ✓ OpenCV {cv2.__version__}")
except ImportError:
    print("   ✗ OpenCV not installed")
    print("      Install: pip install opencv-python")
    all_ok = False

# Check NumPy
print("\n4. Checking NumPy...")
try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError:
    print("   ✗ NumPy not installed")
    print("      Install: pip install numpy")
    all_ok = False

# Check Ultralytics (YOLO)
print("\n5. Checking Ultralytics (YOLO)...")
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"   ✓ Ultralytics {ultralytics.__version__}")
except ImportError:
    print("   ✗ Ultralytics not installed")
    print("      Install: pip install ultralytics")
    all_ok = False

# Check required files
print("\n6. Checking required files...")
required_files = [
    'yolo_text_detector.py',
    'ocr_ctc_attention.py',
    'ocr_ctc.py',
    'pipeline_yolo_ocr.py',
    'requirements.txt',
    'TRAINING_GUIDE.md',
    'NEXT_STEPS.md'
]

for filename in required_files:
    if Path(filename).exists():
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} missing")
        all_ok = False

# Check optional but recommended packages
print("\n7. Checking optional packages...")
optional_packages = {
    'pandas': 'Data handling',
    'matplotlib': 'Visualization',
    'tqdm': 'Progress bars',
    'PIL': ('Pillow', 'Image processing')  # PIL is the import name for Pillow
}

for import_name, info in optional_packages.items():
    if isinstance(info, tuple):
        package_name, description = info
    else:
        package_name = import_name
        description = info
    
    try:
        __import__(import_name)
        print(f"   ✓ {package_name} ({description})")
    except ImportError:
        print(f"   ⚠ {package_name} not installed ({description})")
        warnings.append(f"{package_name} not installed - {description}")

# Check disk space
print("\n8. Checking disk space...")
try:
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (2**30)
    if free_gb > 10:
        print(f"   ✓ {free_gb} GB free")
    else:
        print(f"   ⚠ Only {free_gb} GB free (recommend 10+ GB)")
        warnings.append(f"Low disk space: {free_gb} GB (recommend 10+ GB for models and datasets)")
except Exception as e:
    print(f"   ⚠ Could not check disk space: {e}")

# Check for dataset directories
print("\n9. Checking for dataset directories...")
dataset_paths = ['./dataset', './yolo_dataset', './raw_dataset']
found_datasets = False
for path in dataset_paths:
    if Path(path).exists():
        print(f"   ✓ Found: {path}")
        found_datasets = True

if not found_datasets:
    print("   ⚠ No dataset directories found")
    warnings.append("No dataset directories found - you'll need to prepare your data before training")

# Summary
print("\n" + "=" * 60)
if all_ok and len(warnings) == 0:
    print("✓ System Check PASSED - Ready to train!")
elif all_ok and len(warnings) > 0:
    print("✓ System Check PASSED with warnings")
else:
    print("✗ System Check FAILED - Please fix errors above")

if warnings:
    print("\nWarnings:")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")

print("=" * 60)

# Recommendations
print("\nRecommendations:")
print("  • Use GPU for training (10-20x faster)")
print("  • Prepare datasets before starting (see TRAINING_GUIDE.md)")
print("  • Train YOLO for 100+ epochs")
print("  • Train CRNN for 100+ epochs")
print("  • Keep 10+ GB disk space free for models")

print("\nNext steps:")
print("  1. If errors above, install missing packages:")
print("     pip install -r requirements.txt")
print("  2. Prepare your dataset")
print("  3. Run training:")
print("     .\\quick_start.ps1  (Windows)")
print("     or see NEXT_STEPS.md for manual commands")

print("\n" + "=" * 60)

sys.exit(0 if all_ok else 1)
