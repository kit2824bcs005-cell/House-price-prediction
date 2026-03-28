"""
app.py
======
Flask REST API for House Price Predictor Pro.

Endpoints:
  POST /predict  — Returns predicted price, category, confidence
  GET  /metrics  — Returns all model evaluation metrics
  GET  /features — Returns Random Forest feature importances
  GET  /plots    — Returns available plot filenames

Author: House Price Predictor Pro
"""

import os
import sys
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "..", "frontend", "plots")
MODEL_PKL = os.path.join(BASE_DIR, "model.pkl")
METRICS_JSON = os.path.join(BASE_DIR, "metrics.json")
FEATURES_JSON = os.path.join(BASE_DIR, "feature_importance.json")

sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
CORS(app)  # Allow all origins (for frontend dev)

# ── Load model bundle ─────────────────────────────────────────────────────────
bundle = None
metrics_cache = None
features_cache = None


def load_bundle():
    """Lazy-load the trained model bundle."""
    global bundle
    if bundle is None:
        if not os.path.exists(MODEL_PKL):
            raise FileNotFoundError(
                f"model.pkl not found. Run: python train_model.py"
            )
        bundle = joblib.load(MODEL_PKL)
    return bundle


def load_metrics():
    global metrics_cache
    if metrics_cache is None and os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            metrics_cache = json.load(f)
    return metrics_cache or []


def load_features():
    global features_cache
    if features_cache is None and os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON) as f:
            features_cache = json.load(f)
    return features_cache or []


# ── Utility: price category ────────────────────────────────────────────────────

def inr_category(price_usd: float) -> dict:
    """Convert USD to INR and determine housing category.

    Conversion rate: 1 USD ≈ 83 INR
    Budget  : < 20 Lakhs INR
    Mid     : 20–40 Lakhs INR
    Luxury  : > 40 Lakhs INR
    """
    INR_RATE = 83.0
    LAKH = 100_000

    price_inr = price_usd * INR_RATE
    lakhs = price_inr / LAKH

    if lakhs < 20:
        category = "Budget"
        color = "#22c55e"
        icon = "💚"
    elif lakhs <= 40:
        category = "Mid"
        color = "#f59e0b"
        icon = "🧡"
    else:
        category = "Luxury"
        color = "#a855f7"
        icon = "💜"

    return {
        "price_usd": round(price_usd, 2),
        "price_inr": round(price_inr, 2),
        "price_lakhs": round(lakhs, 2),
        "price_formatted": f"₹{lakhs:.2f} Lakhs",
        "category": category,
        "category_color": color,
        "category_icon": icon,
    }


def compute_confidence(metrics: list, best_name: str) -> float:
    """Compute confidence score from best model's R² and CV stability."""
    for m in metrics:
        if m["name"] == best_name:
            r2 = max(0, m["r2"])
            cv_std = m.get("cv_r2_std", 0.05)
            # Penalise high CV std (instability)
            stability = max(0, 1 - cv_std * 5)
            confidence = round(r2 * stability * 100, 1)
            return min(confidence, 99.9)
    return 75.0


# ═════════════════════════════════════════════════════════════════════════════
# Routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "House Price Predictor Pro API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/metrics", "/features", "/plots/<filename>"],
    })


# ── POST /predict ─────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """Predict house price from input features.

    Request JSON:
        {
          "LotArea": 8450,
          "OverallQual": 7,
          "YearBuilt": 2003,
          "GrLivArea": 1710,
          "TotalBsmtSF": 856,
          "GarageCars": 2,
          "FullBath": 2,
          "BedroomAbvGr": 3
        }

    Response JSON:
        {
          "success": true,
          "price_formatted": "₹18.45 Lakhs",
          "price_usd": 22229.0,
          "price_inr": 1845094.0,
          "price_lakhs": 18.45,
          "category": "Budget",
          "category_color": "#22c55e",
          "category_icon": "💚",
          "confidence": 84.2,
          "model_used": "Gradient Boosting",
          "model_metrics": {...}
        }
    """
    try:
        b = load_bundle()
        data = request.get_json(force=True)

        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        # Required fields
        required = [
            "LotArea", "OverallQual", "YearBuilt", "GrLivArea",
            "TotalBsmtSF", "GarageCars", "FullBath", "BedroomAbvGr",
        ]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {missing}",
            }), 400

        # Validate numeric values
        for field in required:
            try:
                float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    "success": False,
                    "error": f"Field '{field}' must be numeric",
                }), 400

        # Preprocess + predict
        pipeline = b["pipeline"]
        model = b["model"]
        best_name = b["best_model_name"]

        X = pipeline.transform_input(data)
        price_usd = float(model.predict(X)[0])
        price_usd = max(price_usd, 10_000)  # Clamp unrealistic negatives

        # Build response
        result = inr_category(price_usd)
        metrics = load_metrics()
        confidence = compute_confidence(metrics, best_name)

        # Best model's metrics
        model_metrics = next(
            (m for m in metrics if m["name"] == best_name), {}
        )

        return jsonify({
            "success": True,
            **result,
            "confidence": confidence,
            "model_used": best_name,
            "model_metrics": model_metrics,
        })

    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /metrics ──────────────────────────────────────────────────────────────

@app.route("/metrics", methods=["GET"])
def metrics():
    """Return performance metrics for all trained models."""
    try:
        data = load_metrics()
        if not data:
            return jsonify({"success": False, "error": "metrics.json not found. Train the model first."}), 503
        b = load_bundle()
        return jsonify({
            "success": True,
            "best_model": b["best_model_name"],
            "models": data,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /features ─────────────────────────────────────────────────────────────

@app.route("/features", methods=["GET"])
def features():
    """Return Random Forest feature importances."""
    try:
        data = load_features()
        if not data:
            return jsonify({"success": False, "error": "feature_importance.json not found."}), 503
        return jsonify({"success": True, "features": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── GET /plots/<filename> ─────────────────────────────────────────────────────

@app.route("/plots/<path:filename>", methods=["GET"])
def serve_plot(filename):
    """Serve generated plot images."""
    return send_from_directory(PLOTS_DIR, filename)


# ── GET /sample ───────────────────────────────────────────────────────────────

@app.route("/sample", methods=["GET"])
def sample():
    """Return a sample input for demo/testing."""
    return jsonify({
        "success": True,
        "sample": {
            "LotArea": 8450,
            "OverallQual": 7,
            "YearBuilt": 2003,
            "GrLivArea": 1710,
            "TotalBsmtSF": 856,
            "GarageCars": 2,
            "FullBath": 2,
            "BedroomAbvGr": 3,
        },
        "description": "Average Ames Housing property (actual: ~$208,500)",
    })


# ═════════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🏠 House Price Predictor Pro API starting...")
    print("   📡 Running on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
