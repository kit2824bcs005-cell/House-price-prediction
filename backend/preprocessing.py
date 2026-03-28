"""
preprocessing.py
================
Preprocessing pipeline for House Price Predictor Pro.
Implements best practices from top Ames Housing GitHub projects:
  - Feature selection (8 key predictors)
  - Missing value imputation (median strategy)
  - Feature engineering (HouseAge, TotalSF)
  - MinMaxScaler normalization
  - PCA dimensionality reduction

Author: House Price Predictor Pro
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ── Core feature columns ──────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
    "BedroomAbvGr",
]

TARGET_COLUMN = "SalePrice"

# PCA: retain 95% variance
PCA_N_COMPONENTS = 0.95


# ── Helper: derived features ──────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: HouseAge and TotalSF.

    Args:
        df: Raw dataframe with at least YearBuilt and TotalBsmtSF/GrLivArea.

    Returns:
        DataFrame with additional columns HouseAge and TotalSF.
    """
    current_year = 2024
    df = df.copy()

    if "YearBuilt" in df.columns:
        df["HouseAge"] = current_year - df["YearBuilt"]

    if "TotalBsmtSF" in df.columns and "GrLivArea" in df.columns:
        df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]

    return df


# ── Main preprocessing pipeline ───────────────────────────────────────────────

class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for the Ames Housing dataset.

    Steps:
        1. Select relevant features
        2. Impute missing values with median
        3. Add engineered features (HouseAge, TotalSF)
        4. MinMaxScaler normalization
        5. PCA dimensionality reduction (95% variance)

    Usage:
        pipeline = PreprocessingPipeline()
        X_train_pca, y_train = pipeline.fit_transform(train_df)
        X_test_pca = pipeline.transform(test_df)
        user_input = pipeline.transform_input({'LotArea': 8450, ...})
    """

    def __init__(self, n_components: float = PCA_N_COMPONENTS):
        self.n_components = n_components
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.feature_names_: list = []
        self.is_fitted_ = False

    # ── Fit + Transform ───────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """Fit the pipeline on training data and return transformed array.

        Args:
            df: Training dataframe including the TARGET_COLUMN.

        Returns:
            Tuple (X_transformed, y) where X_transformed is PCA-reduced.
        """
        df = self._validate_and_select(df)
        y = df[TARGET_COLUMN].values if TARGET_COLUMN in df.columns else None
        df = add_derived_features(df)

        # Update feature names after engineering
        feature_cols = self._get_feature_cols(df)
        self.feature_names_ = feature_cols

        X = df[feature_cols].values

        # Fit imputer, scaler, PCA on training data
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X)

        self.is_fitted_ = True
        return X_pca, y

    # ── Transform only ────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using already-fitted pipeline.

        Args:
            df: DataFrame with feature columns (no target required).

        Returns:
            PCA-reduced numpy array.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline not fitted. Call fit_transform() first.")

        df = add_derived_features(df)
        feature_cols = self._get_feature_cols(df)

        X = df[feature_cols].values
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return self.pca.transform(X)

    # ── Transform single user input dict ──────────────────────────────────────

    def transform_input(self, input_dict: dict) -> np.ndarray:
        """Transform a single user input dictionary.

        Args:
            input_dict: Keys matching NUMERIC_FEATURES.

        Returns:
            PCA-reduced numpy array of shape (1, n_pca_components).
        """
        row = {col: [input_dict.get(col, np.nan)] for col in NUMERIC_FEATURES}
        df = pd.DataFrame(row)
        return self.transform(df)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _validate_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select available feature columns from dataframe."""
        available = [c for c in NUMERIC_FEATURES if c in df.columns]
        cols = available + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])
        return df[cols].copy()

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Get all feature columns (numeric + engineered) excluding target."""
        base = [c for c in NUMERIC_FEATURES if c in df.columns]
        engineered = [c for c in ["HouseAge", "TotalSF"] if c in df.columns]
        return base + engineered

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """PCA explained variance ratio for scree plot."""
        return self.pca.explained_variance_ratio_

    @property
    def n_pca_components(self) -> int:
        """Number of PCA components retained."""
        return self.pca.n_components_
