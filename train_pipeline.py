"""
Integrated YOLO + CRNN Training Script
Trains both models sequentially for complete OCR pipeline
"""

import subprocess
import time
from pathlib import Path

def train_yolo_crnn_pipeline():
    """Train complete YOLO + CRNN pipeline"""
    
    print("🚀 YOLO + CRNN Training Pipeline")
    print("=" * 50)
    
    # Check prerequisites
    yolo_data_yaml = Path('yolo_data/data.yaml')
    crnn_dataset = Path('dataset')
    
    if not yolo_data_yaml.exists():
        print("❌ YOLO data.yaml not found!")
        print("   Run: python prepare_yolo_data.py first")
        return False
        
    if not crnn_dataset.exists():
        print("❌ CRNN dataset not found!")
        return False
    
    print("✅ Prerequisites check passed")
    print()
    
    # Phase 1: Train YOLO (Text Detection)
    print("Phase 1: Training YOLO for Text Detection")
    print("-" * 40)
    
    yolo_cmd = [
        "python", "yolo_text_detector.py", 
        "train",
        "--data-yaml", str(yolo_data_yaml),
        "--epochs", "100",
        "--batch", "16", 
        "--device", "cuda"
    ]
    
    print(f"Command: {' '.join(yolo_cmd)}")
    print("Estimated time: 3-4 hours")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(yolo_cmd, check=True, capture_output=True, text=True)
        yolo_time = time.time() - start_time
        print(f"✅ YOLO training completed in {yolo_time/3600:.1f} hours")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO training failed: {e}")
        print("Error output:", e.stderr)
        return False
    
    # Phase 2: Train Enhanced CRNN (Text Recognition)
    print("\nPhase 2: Training Enhanced CRNN for Text Recognition")
    print("-" * 50)
    
    crnn_cmd = [
        "python", "ocr_ctc_attention.py",
        "--dataset", str(crnn_dataset),
        "--epochs", "200",
        "--batch-size", "16",
        "--lr", "0.0005",
        "--target-h", "128",
        "--target-w", "512", 
        "--save-path", "best_crnn_high_accuracy.pth",
        "--patience", "20",
        "--device", "cuda"
    ]
    
    print(f"Command: {' '.join(crnn_cmd)}")
    print("Estimated time: 8-10 hours")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(crnn_cmd, check=True)
        crnn_time = time.time() - start_time
        print(f"✅ CRNN training completed in {crnn_time/3600:.1f} hours")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CRNN training failed: {e}")
        return False
    
    # Phase 3: Test Pipeline
    print("\nPhase 3: Testing Complete Pipeline")
    print("-" * 40)
    
    print("✅ Training complete!")
    print("\nTrained models:")
    print("  - YOLO: yolo_runs/text_detector/weights/best.pt")
    print("  - CRNN: best_crnn_high_accuracy.pth")
    print()
    print("Test pipeline with:")
    print("python pipeline_yolo_ocr.py --yolo-weights yolo_runs/text_detector/weights/best.pt --ocr-weights best_crnn_high_accuracy.pth --image test_document.jpg --output results --visualize")
    
    return True

def train_crnn_only():
    """Train only enhanced CRNN for 95% accuracy"""
    
    print("🎯 Enhanced CRNN Training (Target: 95% accuracy)")
    print("=" * 50)
    
    crnn_cmd = [
        "python", "ocr_ctc_attention.py",
        "--dataset", "./dataset",
        "--epochs", "200",
        "--batch-size", "16",
        "--lr", "0.0005",
        "--target-h", "128", 
        "--target-w", "512",
        "--save-path", "best_crnn_high_accuracy.pth",
        "--patience", "20",
        "--device", "cuda"
    ]
    
    print(f"Command: {' '.join(crnn_cmd)}")
    print("Estimated time: 8-10 hours")
    print("Target: 95%+ accuracy on medical handwriting")
    print()
    
    return ' '.join(crnn_cmd)

if __name__ == '__main__':
    print("Choose training option:")
    print("1. CRNN only (95% accuracy focus)")
    print("2. Full YOLO + CRNN pipeline")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        cmd = train_crnn_only()
        print("Run this command:")
        print(cmd)
    elif choice == "2":
        train_yolo_crnn_pipeline()
    else:
        print("Invalid choice")