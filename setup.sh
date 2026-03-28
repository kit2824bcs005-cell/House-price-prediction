#!/bin/bash
# setup.sh — House Price Predictor Pro
echo "============================================================"
echo "  House Price Predictor Pro — Setup Script"
echo "============================================================"

echo "[1/3] Installing Python dependencies..."
pip install -r backend/requirements.txt || { echo "ERROR: pip install failed"; exit 1; }

if [ ! -f "data/train.csv" ]; then
    echo "WARNING: data/train.csv not found!"
    echo "Download from: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset"
    exit 1
fi

echo "[2/3] Training ML models (2-5 minutes)..."
python backend/train_model.py || { echo "ERROR: Training failed"; exit 1; }

echo "[3/3] Starting Flask API on http://localhost:5000"
echo "Open frontend/index.html in your browser."
python backend/app.py
