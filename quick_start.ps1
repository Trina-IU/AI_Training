# Quick Start: Automated Training Script for Doctor's Handwriting OCR
# This script automates the training process for both YOLO and CRNN

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Doctor's Handwriting OCR - Quick Start" -ForegroundColor Cyan
Write-Host "YOLO Detection + CRNN Recognition" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Select training mode:" -ForegroundColor Yellow
Write-Host "  1. Full Pipeline (YOLO + CRNN) - Recommended for document images"
Write-Host "  2. CRNN Only - For pre-cropped text images"
Write-Host "  3. YOLO Only - For text detection"
Write-Host "  4. Resume Training (continue interrupted training)"
Write-Host "  5. Run Inference (use trained models)"
Write-Host ""

$mode = Read-Host "Enter choice (1-5)"

# Detect GPU
$hasGPU = $false
try {
    $gpuCheck = python -c "import torch; print(torch.cuda.is_available())" 2>&1
    if ($gpuCheck -eq "True") {
        $hasGPU = $true
        Write-Host "✓ CUDA GPU detected - Training will be fast!" -ForegroundColor Green
        $device = "cuda"
    } else {
        Write-Host "! No CUDA GPU detected - Training will use CPU (slower)" -ForegroundColor Yellow
        $device = "cpu"
    }
} catch {
    Write-Host "! GPU detection failed - Using CPU" -ForegroundColor Yellow
    $device = "cpu"
}

Write-Host ""

switch ($mode) {
    "1" {
        Write-Host "=== Full Pipeline Training ===" -ForegroundColor Cyan
        Write-Host ""
        
        # Step 1: Check YOLO dataset
        Write-Host "Step 1: Prepare YOLO dataset" -ForegroundColor Yellow
        $yoloDataset = Read-Host "Enter path to raw dataset (or press Enter to skip if already prepared)"
        
        if ($yoloDataset -ne "") {
            $yoloOutput = Read-Host "Enter output path for YOLO dataset (default: ./yolo_dataset)"
            if ($yoloOutput -eq "") { $yoloOutput = "./yolo_dataset" }
            
            Write-Host "Preparing YOLO dataset..." -ForegroundColor Cyan
            python yolo_text_detector.py prepare --input $yoloDataset --output $yoloOutput
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ YOLO dataset prepared" -ForegroundColor Green
            } else {
                Write-Host "✗ Failed to prepare YOLO dataset" -ForegroundColor Red
                exit 1
            }
        }
        
        # Step 2: Train YOLO
        Write-Host ""
        Write-Host "Step 2: Train YOLO text detector" -ForegroundColor Yellow
        $yoloDataYaml = Read-Host "Enter path to data.yaml (default: ./yolo_dataset/data.yaml)"
        if ($yoloDataYaml -eq "") { $yoloDataYaml = "./yolo_dataset/data.yaml" }
        
        $yoloEpochs = Read-Host "Enter number of epochs (default: 100)"
        if ($yoloEpochs -eq "") { $yoloEpochs = "100" }
        
        $yoloBatch = if ($hasGPU) { "16" } else { "8" }
        $yoloBatch = Read-Host "Enter batch size (default: $yoloBatch)"
        if ($yoloBatch -eq "") { $yoloBatch = if ($hasGPU) { "16" } else { "8" } }
        
        Write-Host "Starting YOLO training (this will take several hours)..." -ForegroundColor Cyan
        Write-Host "Command: python yolo_text_detector.py train --data-yaml $yoloDataYaml --epochs $yoloEpochs --batch $yoloBatch --device $device" -ForegroundColor Gray
        
        python yolo_text_detector.py train --data-yaml $yoloDataYaml --epochs $yoloEpochs --batch $yoloBatch --device $device
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ YOLO training complete!" -ForegroundColor Green
            Write-Host "Model saved to: yolo_runs/text_detector/weights/best.pt" -ForegroundColor Green
        } else {
            Write-Host "✗ YOLO training failed" -ForegroundColor Red
            exit 1
        }
        
        # Step 3: Train CRNN
        Write-Host ""
        Write-Host "Step 3: Train CRNN with Attention" -ForegroundColor Yellow
        $crnnDataset = Read-Host "Enter path to OCR dataset (default: ./dataset)"
        if ($crnnDataset -eq "") { $crnnDataset = "./dataset" }
        
        $crnnEpochs = Read-Host "Enter number of epochs (default: 100)"
        if ($crnnEpochs -eq "") { $crnnEpochs = "100" }
        
        $crnnBatch = if ($hasGPU) { "32" } else { "16" }
        $crnnBatch = Read-Host "Enter batch size (default: $crnnBatch)"
        if ($crnnBatch -eq "") { $crnnBatch = if ($hasGPU) { "32" } else { "16" } }
        
        Write-Host "Starting CRNN training (this will take several hours)..." -ForegroundColor Cyan
        Write-Host "Command: python ocr_ctc_attention.py --dataset $crnnDataset --epochs $crnnEpochs --batch-size $crnnBatch --device $device --save-path best_crnn_attention.pth --patience 15" -ForegroundColor Gray
        
        python ocr_ctc_attention.py --dataset $crnnDataset --epochs $crnnEpochs --batch-size $crnnBatch --device $device --save-path best_crnn_attention.pth --patience 15
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ CRNN training complete!" -ForegroundColor Green
            Write-Host "Model saved to: best_crnn_attention.pth" -ForegroundColor Green
        } else {
            Write-Host "✗ CRNN training failed" -ForegroundColor Red
            exit 1
        }
        
        Write-Host ""
        Write-Host "=====================================" -ForegroundColor Cyan
        Write-Host "✓ Full pipeline training complete!" -ForegroundColor Green
        Write-Host "=====================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor Yellow
        Write-Host "  Run inference: ./quick_start.ps1 (select option 5)"
        Write-Host "  Or manually: python pipeline_yolo_ocr.py --yolo-weights yolo_runs/text_detector/weights/best.pt --ocr-weights best_crnn_attention.pth --image your_image.jpg --output results/ --visualize"
    }
    
    "2" {
        Write-Host "=== CRNN Training Only ===" -ForegroundColor Cyan
        Write-Host ""
        
        $crnnDataset = Read-Host "Enter path to OCR dataset (default: ./dataset)"
        if ($crnnDataset -eq "") { $crnnDataset = "./dataset" }
        
        $crnnEpochs = Read-Host "Enter number of epochs (default: 100)"
        if ($crnnEpochs -eq "") { $crnnEpochs = "100" }
        
        $crnnBatch = if ($hasGPU) { "32" } else { "16" }
        $crnnBatch = Read-Host "Enter batch size (default: $crnnBatch)"
        if ($crnnBatch -eq "") { $crnnBatch = if ($hasGPU) { "32" } else { "16" } }
        
        Write-Host "Starting CRNN training with attention mechanism..." -ForegroundColor Cyan
        Write-Host "Command: python ocr_ctc_attention.py --dataset $crnnDataset --epochs $crnnEpochs --batch-size $crnnBatch --device $device --save-path best_crnn_attention.pth --patience 15" -ForegroundColor Gray
        
        python ocr_ctc_attention.py --dataset $crnnDataset --epochs $crnnEpochs --batch-size $crnnBatch --device $device --save-path best_crnn_attention.pth --patience 15 --checkpoint-batches 100
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Training complete!" -ForegroundColor Green
            Write-Host "Model saved to: best_crnn_attention.pth" -ForegroundColor Green
        }
    }
    
    "3" {
        Write-Host "=== YOLO Training Only ===" -ForegroundColor Cyan
        Write-Host ""
        
        $yoloDataYaml = Read-Host "Enter path to data.yaml"
        
        if ($yoloDataYaml -eq "") {
            Write-Host "Error: data.yaml path is required" -ForegroundColor Red
            exit 1
        }
        
        $yoloEpochs = Read-Host "Enter number of epochs (default: 100)"
        if ($yoloEpochs -eq "") { $yoloEpochs = "100" }
        
        $yoloBatch = if ($hasGPU) { "16" } else { "8" }
        $yoloBatch = Read-Host "Enter batch size (default: $yoloBatch)"
        if ($yoloBatch -eq "") { $yoloBatch = if ($hasGPU) { "16" } else { "8" } }
        
        Write-Host "Starting YOLO training..." -ForegroundColor Cyan
        python yolo_text_detector.py train --data-yaml $yoloDataYaml --epochs $yoloEpochs --batch $yoloBatch --device $device
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Training complete!" -ForegroundColor Green
            Write-Host "Model saved to: yolo_runs/text_detector/weights/best.pt" -ForegroundColor Green
        }
    }
    
    "4" {
        Write-Host "=== Resume Training ===" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Which model to resume?" -ForegroundColor Yellow
        Write-Host "  1. CRNN"
        Write-Host "  2. YOLO"
        $resumeChoice = Read-Host "Enter choice (1-2)"
        
        if ($resumeChoice -eq "1") {
            $checkpoint = Read-Host "Enter checkpoint path (default: best_crnn_attention.last.pth)"
            if ($checkpoint -eq "") { $checkpoint = "best_crnn_attention.last.pth" }
            
            $crnnDataset = Read-Host "Enter dataset path (default: ./dataset)"
            if ($crnnDataset -eq "") { $crnnDataset = "./dataset" }
            
            $crnnEpochs = Read-Host "Enter total epochs (default: 100)"
            if ($crnnEpochs -eq "") { $crnnEpochs = "100" }
            
            Write-Host "Resuming CRNN training..." -ForegroundColor Cyan
            python ocr_ctc_attention.py --dataset $crnnDataset --epochs $crnnEpochs --batch-size 32 --device $device --save-path best_crnn_attention.pth --resume-from $checkpoint --patience 15
        }
        elseif ($resumeChoice -eq "2") {
            Write-Host "YOLO automatically resumes from last.pt if available" -ForegroundColor Yellow
            $yoloDataYaml = Read-Host "Enter data.yaml path"
            python yolo_text_detector.py train --data-yaml $yoloDataYaml --epochs 100 --device $device --resume
        }
    }
    
    "5" {
        Write-Host "=== Run Inference ===" -ForegroundColor Cyan
        Write-Host ""
        
        $yoloWeights = Read-Host "Enter YOLO weights path (default: yolo_runs/text_detector/weights/best.pt)"
        if ($yoloWeights -eq "") { $yoloWeights = "yolo_runs/text_detector/weights/best.pt" }
        
        $ocrWeights = Read-Host "Enter OCR weights path (default: best_crnn_attention.pth)"
        if ($ocrWeights -eq "") { $ocrWeights = "best_crnn_attention.pth" }
        
        Write-Host "Process:" -ForegroundColor Yellow
        Write-Host "  1. Single image"
        Write-Host "  2. Folder of images"
        $inferenceChoice = Read-Host "Enter choice (1-2)"
        
        if ($inferenceChoice -eq "1") {
            $imagePath = Read-Host "Enter image path"
            $output = Read-Host "Enter output directory (default: ./results)"
            if ($output -eq "") { $output = "./results" }
            
            Write-Host "Running pipeline..." -ForegroundColor Cyan
            python pipeline_yolo_ocr.py --yolo-weights $yoloWeights --ocr-weights $ocrWeights --image $imagePath --output $output --visualize --device $device
        }
        elseif ($inferenceChoice -eq "2") {
            $imageDir = Read-Host "Enter images directory"
            $output = Read-Host "Enter output directory (default: ./results)"
            if ($output -eq "") { $output = "./results" }
            
            Write-Host "Running pipeline on all images..." -ForegroundColor Cyan
            python pipeline_yolo_ocr.py --yolo-weights $yoloWeights --ocr-weights $ocrWeights --image-dir $imageDir --output $output --visualize --device $device
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Inference complete!" -ForegroundColor Green
            Write-Host "Check results in: $output" -ForegroundColor Green
        }
    }
    
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Done! Press any key to exit..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
