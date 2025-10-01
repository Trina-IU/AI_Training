# Simple Training Launcher for Doctor's Handwriting OCR
# This is a simplified version without complex error handling

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Doctor's Handwriting OCR - Training" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Check GPU
Write-Host "Checking GPU..." -ForegroundColor Yellow
$gpuAvailable = python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')"
if ($gpuAvailable -eq "Yes") {
    Write-Host "GPU detected! Training will be fast." -ForegroundColor Green
    $device = "cuda"
} else {
    Write-Host "No GPU detected. Training will be slower on CPU." -ForegroundColor Yellow
    $device = "cpu"
}

Write-Host ""
Write-Host "Select what to train:" -ForegroundColor Yellow
Write-Host "  1. CRNN Only (text recognition)"
Write-Host "  2. YOLO Only (text detection)"
Write-Host "  3. Both YOLO + CRNN"
Write-Host "  4. Test/Inference"
Write-Host ""

$choice = Read-Host "Enter choice (1-4)"

Write-Host ""

if ($choice -eq "1") {
    Write-Host "=== Training CRNN (Text Recognition) ===" -ForegroundColor Cyan
    Write-Host ""

    $dataset = Read-Host "Dataset path (default: .\dataset)"
    if ($dataset -eq "") { $dataset = ".\dataset" }

    $epochs = Read-Host "Epochs (default: 100)"
    if ($epochs -eq "") { $epochs = "100" }

    $batch = Read-Host "Batch size (default: 32 for GPU, 16 for CPU)"
    if ($batch -eq "") {
        if ($device -eq "cuda") { $batch = "32" } else { $batch = "16" }
    }

    Write-Host ""
    Write-Host "Starting CRNN training..." -ForegroundColor Green
    Write-Host "This will take 3-6 hours on GPU, 12-24 hours on CPU" -ForegroundColor Yellow
    Write-Host ""

    python ocr_ctc_attention.py --dataset $dataset --epochs $epochs --batch-size $batch --device $device --save-path best_crnn_attention.pth --patience 10

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Training complete! Model saved to: best_crnn_attention.pth" -ForegroundColor Green
    }
}
elseif ($choice -eq "2") {
    Write-Host "=== Training YOLO (Text Detection) ===" -ForegroundColor Cyan
    Write-Host ""

    $dataYaml = Read-Host "Path to data.yaml"
    if ($dataYaml -eq "") {
        Write-Host "Error: data.yaml path required" -ForegroundColor Red
        exit
    }

    $epochs = Read-Host "Epochs (default: 100)"
    if ($epochs -eq "") { $epochs = "100" }

    $batch = Read-Host "Batch size (default: 16 for GPU, 8 for CPU)"
    if ($batch -eq "") {
        if ($device -eq "cuda") { $batch = "16" } else { $batch = "8" }
    }

    Write-Host ""
    Write-Host "Starting YOLO training..." -ForegroundColor Green
    Write-Host "This will take 2-4 hours on GPU, 12-24 hours on CPU" -ForegroundColor Yellow
    Write-Host ""

    python yolo_text_detector.py train --data-yaml $dataYaml --epochs $epochs --batch $batch --device $device

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "Training complete! Model saved to: yolo_runs/text_detector/weights/best.pt" -ForegroundColor Green
    }
}
elseif ($choice -eq "3") {
    Write-Host "=== Training Both YOLO + CRNN ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This will train YOLO first, then CRNN" -ForegroundColor Yellow
    Write-Host "Total time: 5-10 hours on GPU, 24-48 hours on CPU" -ForegroundColor Yellow
    Write-Host ""

    # YOLO first
    $yoloData = Read-Host "YOLO data.yaml path"
    if ($yoloData -ne "") {
        Write-Host "Training YOLO..." -ForegroundColor Green
        python yolo_text_detector.py train --data-yaml $yoloData --epochs 100 --batch 16 --device $device
    }

    # Then CRNN
    $crnnDataset = Read-Host "CRNN dataset path (default: .\dataset)"
    if ($crnnDataset -eq "") { $crnnDataset = ".\dataset" }

    Write-Host "Training CRNN..." -ForegroundColor Green
    python ocr_ctc_attention.py --dataset $crnnDataset --epochs 100 --batch-size 32 --device $device --save-path best_crnn_attention.pth --patience 10

    Write-Host ""
    Write-Host "Both models trained!" -ForegroundColor Green
}
elseif ($choice -eq "4") {
    Write-Host "=== Test/Inference ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Test options:" -ForegroundColor Yellow
    Write-Host "  1. Test CRNN on single image"
    Write-Host "  2. Test CRNN on folder"
    Write-Host "  3. Test full pipeline (YOLO + CRNN)"
    Write-Host ""

    $testChoice = Read-Host "Enter choice (1-3)"

    if ($testChoice -eq "1") {
        $model = Read-Host "CRNN model path (default: best_crnn_attention.pth)"
        if ($model -eq "") { $model = "best_crnn_attention.pth" }

        $image = Read-Host "Image path"

        python predict_text_ctc.py $model $image --device $device
    }
    elseif ($testChoice -eq "2") {
        $model = Read-Host "CRNN model path (default: best_crnn_attention.pth)"
        if ($model -eq "") { $model = "best_crnn_attention.pth" }

        $folder = Read-Host "Folder path"
        $output = Read-Host "Output CSV (default: predictions.csv)"
        if ($output -eq "") { $output = "predictions.csv" }

        python batch_predict_ctc.py --checkpoint $model --input-dir $folder --output $output --device $device
    }
    elseif ($testChoice -eq "3") {
        $yolo = Read-Host "YOLO weights path"
        $crnn = Read-Host "CRNN weights path"
        $image = Read-Host "Image path"
        $output = Read-Host "Output dir (default: .\results)"
        if ($output -eq "") { $output = ".\results" }

        python pipeline_yolo_ocr.py --yolo-weights $yolo --ocr-weights $crnn --image $image --output $output --visualize --device $device
    }
}
else {
    Write-Host "Invalid choice" -ForegroundColor Red
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Cyan
