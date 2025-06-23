@echo off
echo 🔮 Review Fraud Detection - Batch Prediction
echo ============================================
echo.

REM Activate virtual environment if it exists
if exist "venv" (
    echo Activating virtual environment...
    call venv\Scripts\activate
)

REM Check if model files exist
if not exist "model.pkl" (
    echo ❌ Error: model.pkl not found!
    echo Please train the model first using train.py or the Flask API.
    pause
    exit /b 1
)

if not exist "vectorizer.pkl" (
    echo ❌ Error: vectorizer.pkl not found!
    echo Please train the model first using train.py or the Flask API.
    pause
    exit /b 1
)

REM Get input file from user
if "%1"=="" (
    set /p input_file="Enter CSV file path (or press Enter for sample_reviews.csv): "
    if "!input_file!"=="" set input_file=sample_reviews.csv
) else (
    set input_file=%1
)

REM Check if input file exists
if not exist "%input_file%" (
    echo ❌ Error: Input file '%input_file%' not found!
    pause
    exit /b 1
)

echo.
echo 📁 Input file: %input_file%
echo 🚀 Starting prediction process...
echo.

REM Run prediction
python predict.py "%input_file%"

echo.
echo ✅ Prediction completed!
pause
