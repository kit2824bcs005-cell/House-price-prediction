@echo off
echo ============================================================
echo   House Price Predictor Pro — Setup Script
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r backend\requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: pip install failed. Make sure Python and pip are installed.
    pause
    exit /b 1
)
echo.

echo [2/3] Checking for dataset...
if not exist "data\train.csv" (
    echo WARNING: data\train.csv not found!
    echo Please download the Ames Housing dataset and place train.csv in the data\ folder.
    echo Download: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset
    echo.
) else (
    echo Found data\train.csv
    echo.
    echo [3/3] Training ML models (this may take 2-5 minutes)...
    python backend\train_model.py
    if %errorlevel% neq 0 (
        echo ERROR: Training failed.
        pause
        exit /b 1
    )
    echo.
    echo ============================================================
    echo   Training complete! Now starting Flask API...
    echo   Open frontend\index.html in your browser.
    echo ============================================================
    echo.
    python backend\app.py
)

pause
