@echo off
echo Starting Review Fraud Detection Flask API...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Start Flask application
echo.
echo Starting Flask application on http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
