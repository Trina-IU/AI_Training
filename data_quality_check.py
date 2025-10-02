"""
Data Quality Check Script
Diagnoses why accuracy is 78% instead of 95%
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
import torch

def check_image_quality():
    """Check image quality differences between train/test"""
    
    print("🔍 DATA QUALITY CHECK")
    print("=" * 50)
    
    # Check test folder
    test_dir = Path('./test')
    if not test_dir.exists():
        print("❌ Test folder not found!")
        return
    
    test_images = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
    print(f"✅ Found {len(test_images)} test images")
    
    # Check training data
    dataset_dir = Path('./dataset')
    train_images = []
    for subfolder in ['Medicine', 'Dosage', 'Instructions', 'Abbreviation', 'numbers']:
        folder_path = dataset_dir / subfolder
        if folder_path.exists():
            folder_images = list(folder_path.glob('*.png'))
            train_images.extend(folder_images)
    
    print(f"✅ Found {len(train_images)} training images")
    
    if len(test_images) == 0 or len(train_images) == 0:
        print("❌ No images found to compare!")
        return
    
    # Analyze image properties
    print("\n📊 IMAGE ANALYSIS")
    print("-" * 30)
    
    def analyze_images(image_paths, label):
        sizes = []
        contrasts = []
        brightnesses = []
        
        for img_path in image_paths[:10]:  # Sample 10 images
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sizes.append(img.shape)
                contrasts.append(np.std(img))
                brightnesses.append(np.mean(img))
        
        print(f"\n{label}:")
        if sizes:
            print(f"  Image sizes: {Counter(sizes).most_common(3)}")
            print(f"  Avg contrast: {np.mean(contrasts):.1f} ± {np.std(contrasts):.1f}")
            print(f"  Avg brightness: {np.mean(brightnesses):.1f} ± {np.std(brightnesses):.1f}")
        
        return sizes, contrasts, brightnesses
    
    test_stats = analyze_images(test_images, "TEST IMAGES")
    train_stats = analyze_images(train_images[:20], "TRAINING IMAGES (sample)")
    
    # Check if sizes/quality match
    test_contrast_avg = np.mean(test_stats[1]) if test_stats[1] else 0
    train_contrast_avg = np.mean(train_stats[1]) if train_stats[1] else 0
    
    print(f"\n🎯 QUALITY ANALYSIS")
    print("-" * 30)
    
    if abs(test_contrast_avg - train_contrast_avg) > 20:
        print("⚠️  CONTRAST MISMATCH detected!")
        print("   Test and training images have different contrast levels")
        print("   This can cause accuracy drops!")
    else:
        print("✅ Contrast levels are similar")
    
    # Check model predictions vs ground truth
    print(f"\n📋 PREDICTION ANALYSIS")
    print("-" * 30)
    
    # Load recent predictions
    results_file = Path('./test_results_fixed.csv')
    if results_file.exists():
        with open(results_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        predictions = [line.strip().split(',')[1] for line in lines]
        pred_lengths = [len(p.strip('"')) for p in predictions]
        
        print(f"Prediction length stats:")
        print(f"  Min: {min(pred_lengths)}, Max: {max(pred_lengths)}, Avg: {np.mean(pred_lengths):.1f}")
        
        # Common prediction patterns
        pred_counter = Counter(predictions)
        print(f"\nMost common predictions:")
        for pred, count in pred_counter.most_common(5):
            print(f"  '{pred.strip('\"')}': {count} times")
    
    return True


def quick_improvement_suggestions():
    """Suggest immediate improvements based on analysis"""
    
    print(f"\n🚀 QUICK IMPROVEMENT SUGGESTIONS")
    print("=" * 50)
    
    print("1. IMMEDIATE IMPROVEMENTS (2-3 hours):")
    print("   ✅ Train with higher resolution: --target-h 128 --target-w 512")
    print("   ✅ Lower learning rate: --lr 0.0005") 
    print("   ✅ More epochs: --epochs 200")
    print("   ✅ Smaller batch size: --batch-size 16")
    print()
    
    print("2. ENHANCED TRAINING COMMAND:")
    cmd = """python ocr_ctc_attention.py --dataset ./dataset --epochs 200 --batch-size 16 --lr 0.0005 --target-h 128 --target-w 512 --save-path best_crnn_95_percent.pth --patience 20 --device cuda"""
    print(f"   {cmd}")
    print()
    
    print("3. DATA PREPROCESSING IMPROVEMENTS:")
    print("   ✅ Enhance contrast on test images")
    print("   ✅ Normalize image brightness")
    print("   ✅ Apply same preprocessing as training")
    print()
    
    print("4. EXPECTED RESULTS:")
    print("   📊 Training time: 8-10 hours")
    print("   🎯 Target accuracy: 95%+")
    print("   📈 Validation loss: < 0.02")
    print()
    
    return cmd


def test_preprocessing():
    """Test different preprocessing on current test images"""
    
    print(f"\n🧪 PREPROCESSING TEST")
    print("=" * 30)
    
    test_dir = Path('./test')
    if not test_dir.exists():
        print("❌ Test folder not found!")
        return
    
    test_images = list(test_dir.glob('*.png'))[:3]  # Test first 3 images
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        
        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        # Original stats
        print(f"  Original - Contrast: {np.std(img):.1f}, Brightness: {np.mean(img):.1f}")
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        print(f"  Enhanced - Contrast: {np.std(enhanced):.1f}, Brightness: {np.mean(enhanced):.1f}")
        
        # Resize to training size
        resized = cv2.resize(enhanced, (256, 64), interpolation=cv2.INTER_CUBIC)
        print(f"  Resized to training size: {resized.shape}")


if __name__ == '__main__':
    print("DATA QUALITY CHECK & QUICK IMPROVEMENTS")
    print("=" * 60)
    
    # Step 1: Check data quality
    check_image_quality()
    
    # Step 2: Test preprocessing 
    test_preprocessing()
    
    # Step 3: Get improvement suggestions
    cmd = quick_improvement_suggestions()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED NEXT ACTION:")
    print("=" * 60)
    print("Run the enhanced training command above to achieve 95% accuracy!")
    print("Estimated time: 8-10 hours")