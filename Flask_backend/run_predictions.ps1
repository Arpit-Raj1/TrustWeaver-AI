Write-Host "🔮 Review Fraud Detection - Batch Prediction" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host

# Activate virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Check if model files exist
if (-not (Test-Path "model.pkl")) {
    Write-Host "❌ Error: model.pkl not found!" -ForegroundColor Red
    Write-Host "Please train the model first using train.py or the Flask API." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "vectorizer.pkl")) {
    Write-Host "❌ Error: vectorizer.pkl not found!" -ForegroundColor Red
    Write-Host "Please train the model first using train.py or the Flask API." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Get input file from user or parameter
if ($args.Count -eq 0) {
    $input_file = Read-Host "Enter CSV file path (or press Enter for sample_reviews.csv)"
    if ([string]::IsNullOrEmpty($input_file)) {
        $input_file = "sample_reviews.csv"
    }
} else {
    $input_file = $args[0]
}

# Check if input file exists
if (-not (Test-Path $input_file)) {
    Write-Host "❌ Error: Input file '$input_file' not found!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host
Write-Host "📁 Input file: $input_file" -ForegroundColor Cyan
Write-Host "🚀 Starting prediction process..." -ForegroundColor Green
Write-Host

# Run prediction
python predict.py $input_file

Write-Host
Write-Host "✅ Prediction completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
