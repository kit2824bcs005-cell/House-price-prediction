"""
train_model.py
==============
Trains and compares 6 ML models on the Ames Housing dataset.
Implements GridSearchCV hyperparameter tuning, cross-validation,
and saves the best model as model.pkl.

Models:
  1. Linear Regression (baseline)
  2. Ridge Regression
  3. Lasso Regression
  4. KNN Regressor (k=5)
  5. Random Forest Regressor
  6. Gradient Boosting Regressor

Author: House Price Predictor Pro
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")
MODEL_PKL = os.path.join(BASE_DIR, "model.pkl")
METRICS_JSON = os.path.join(BASE_DIR, "metrics.json")
FEATURES_JSON = os.path.join(BASE_DIR, "feature_importance.json")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "frontend", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Import pipeline ───────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from preprocessing import PreprocessingPipeline, NUMERIC_FEATURES


# ═════════════════════════════════════════════════════════════════════════════
# 1. Load & split data
# ═════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load Ames Housing CSV and return train/test splits."""
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    pipeline = PreprocessingPipeline()
    X, y = pipeline.fit_transform(df)

    print(f"   ✓ PCA reduced to {pipeline.n_pca_components} components "
          f"({pipeline.explained_variance_ratio.sum()*100:.1f}% variance retained)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, pipeline, df


# ═════════════════════════════════════════════════════════════════════════════
# 2. Model definitions + hyperparameter grids
# ═════════════════════════════════════════════════════════════════════════════

def get_model_configs():
    """Return list of (name, estimator, param_grid) tuples."""
    return [
        (
            "Linear Regression",
            LinearRegression(),
            {},  # No hyperparams to tune
        ),
        (
            "Ridge Regression",
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        (
            "Lasso Regression",
            Lasso(max_iter=10000),
            {"alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        (
            "KNN Regressor",
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]},
        ),
        (
            "Random Forest",
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5],
            },
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 3. Train + evaluate all models
# ═════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train all models, run GridSearchCV, and collect metrics."""
    configs = get_model_configs()
    results = []
    best_model = None
    best_r2 = -np.inf
    best_model_name = ""

    for name, estimator, param_grid in configs:
        print(f"\n🔧 Training: {name}")
        t0 = time.time()

        if param_grid:
            cv_search = GridSearchCV(
                estimator,
                param_grid,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )
            cv_search.fit(X_train, y_train)
            model = cv_search.best_estimator_
            print(f"   Best params: {cv_search.best_params_}")
        else:
            model = estimator
            model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Cross-validation R² on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = round(time.time() - t0, 2)

        print(f"   ✓ R²={r2:.4f} | RMSE={rmse:.0f} | MAE={mae:.0f} | "
              f"CV R²={cv_scores.mean():.4f} | {elapsed}s")

        results.append({
            "name": name,
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "cv_r2_mean": round(cv_scores.mean(), 4),
            "cv_r2_std": round(cv_scores.std(), 4),
            "train_time_s": elapsed,
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

    print(f"\n🏆 Best Model: {best_model_name} (R²={best_r2:.4f})")
    return results, best_model, best_model_name


# ═════════════════════════════════════════════════════════════════════════════
# 4. Feature importance
# ═════════════════════════════════════════════════════════════════════════════

def extract_feature_importance(df: pd.DataFrame, pipeline: PreprocessingPipeline):
    """Train a dedicated Random Forest on original features to get feature importance."""
    from sklearn.impute import SimpleImputer

    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    X_raw = df[feature_cols].values
    y_raw = df["SalePrice"].values

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_imp, y_raw)

    importances = rf.feature_importances_
    importance_list = [
        {"feature": f, "importance": round(float(v), 6)}
        for f, v in sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    ]
    return importance_list


# ═════════════════════════════════════════════════════════════════════════════
# 5. Generate plots
# ═════════════════════════════════════════════════════════════════════════════

def generate_plots(df, pipeline, metrics, importance_list, X_test, y_test, best_model):
    """Generate and save all visualization plots."""
    print("\n📊 Generating plots...")

    # ── Color palette ────────────────────────────────────────────────────────
    DARK_BG = "#0f0f1a"
    CARD_BG = "#1a1a2e"
    ACCENT = "#6c63ff"
    ACCENT2 = "#ff6584"
    TEXT = "#e0e0e0"
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })

    # ── 1. Correlation Heatmap ───────────────────────────────────────────────
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr_cols = feature_cols + (["SalePrice"] if "SalePrice" in df.columns else [])
    corr_df = df[corr_cols].dropna()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(DARK_BG)
    mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
    sns.heatmap(
        corr_df.corr(), mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold",
                 color="#6c63ff", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ Correlation heatmap")

    # ── 2. PCA Scree Plot ────────────────────────────────────────────────────
    evr = pipeline.explained_variance_ratio
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.bar(range(1, len(evr)+1), evr*100, color=ACCENT, alpha=0.85, edgecolor="none")
    ax.plot(range(1, len(evr)+1), np.cumsum(evr)*100, color=ACCENT2, marker="o",
            linewidth=2.5, markersize=7, label="Cumulative Variance")
    ax.axhline(95, color="#ffd700", linestyle="--", alpha=0.7, label="95% threshold")
    ax.set_xlabel("Principal Component", fontsize=13)
    ax.set_ylabel("Explained Variance (%)", fontsize=13)
    ax.set_title("PCA Scree Plot", fontsize=16, fontweight="bold", color="#6c63ff", pad=15)
    ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_scree.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ PCA scree plot")

    # ── 3. Feature Importance ────────────────────────────────────────────────
    feat_df = pd.DataFrame(importance_list).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)
    colors = sns.color_palette("cool", len(feat_df))
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors, edgecolor="none")
    ax.set_xlabel("Importance", fontsize=13)
    ax.set_title("Random Forest Feature Importance", fontsize=16, fontweight="bold",
                 color="#6c63ff", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ Feature importance")

    # ── 4. Actual vs Predicted ───────────────────────────────────────────────
    y_pred = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.scatter(y_test, y_pred, alpha=0.5, color=ACCENT, edgecolors="none", s=40)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=ACCENT2, linewidth=2, linestyle="--", label="Perfect Prediction")
    ax.set_xlabel("Actual Price (USD)", fontsize=13)
    ax.set_ylabel("Predicted Price (USD)", fontsize=13)
    ax.set_title("Actual vs Predicted Prices", fontsize=16, fontweight="bold",
                 color="#6c63ff", pad=15)
    ax.legend(facecolor=CARD_BG, edgecolor="none", labelcolor=TEXT)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ Actual vs Predicted")

    # ── 5. Residual Plot ─────────────────────────────────────────────────────
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.scatter(y_pred, residuals, alpha=0.5, color=ACCENT2, edgecolors="none", s=40)
    ax.axhline(0, color="#ffd700", linewidth=2, linestyle="--")
    ax.set_xlabel("Predicted Price (USD)", fontsize=13)
    ax.set_ylabel("Residuals", fontsize=13)
    ax.set_title("Residual Plot", fontsize=16, fontweight="bold", color="#6c63ff", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "residuals.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ Residual plot")

    # ── 6. Model Comparison Bar Chart ────────────────────────────────────────
    mdf = pd.DataFrame(metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes:
        ax.set_facecolor(CARD_BG)

    axes[0].barh(mdf["name"], mdf["r2"], color=ACCENT, edgecolor="none")
    axes[0].set_title("R² Score", fontsize=13, color=ACCENT)
    axes[0].set_xlim(0, 1)

    axes[1].barh(mdf["name"], mdf["rmse"], color=ACCENT2, edgecolor="none")
    axes[1].set_title("RMSE (lower is better)", fontsize=13, color=ACCENT2)

    axes[2].barh(mdf["name"], mdf["mae"], color="#ffd700", edgecolor="none")
    axes[2].set_title("MAE (lower is better)", fontsize=13, color="#ffd700")

    for ax in axes:
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold",
                 color="#6c63ff", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("   ✓ Model comparison chart")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  🏠 House Price Predictor Pro — Model Training")
    print("=" * 60)

    # Load & preprocess
    X_train, X_test, y_train, y_test, pipeline, df = load_data()

    # Train all models
    metrics, best_model, best_model_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Feature importance (on raw features)
    print("\n📈 Extracting feature importances...")
    importance_list = extract_feature_importance(df, pipeline)

    # Generate plots
    generate_plots(df, pipeline, metrics, importance_list, X_test, y_test, best_model)

    # Save model + pipeline bundle
    bundle = {
        "model": best_model,
        "pipeline": pipeline,
        "best_model_name": best_model_name,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PKL)
    print(f"\n💾 Saved model bundle → {MODEL_PKL}")

    # Save metrics JSON
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics → {METRICS_JSON}")

    # Save feature importance JSON
    with open(FEATURES_JSON, "w") as f:
        json.dump(importance_list, f, indent=2)
    print(f"💾 Saved feature importances → {FEATURES_JSON}")

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete! Best model: {best_model_name}")
    best = next(m for m in metrics if m["name"] == best_model_name)
    print(f"     R²={best['r2']} | RMSE={best['rmse']:.0f} | MAE={best['mae']:.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
