"""Preprocessing components with leakage-safe train-fold fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ExperimentConfig


class Winsorizer(BaseEstimator, TransformerMixin):
    """Quantile-based winsorization learned on training data only."""

    def __init__(self, lower_q: float = 0.01, upper_q: float = 0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.lower_bounds_ = X_df.quantile(self.lower_q)
        self.upper_bounds_ = X_df.quantile(self.upper_q)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].clip(self.lower_bounds_[col], self.upper_bounds_[col])
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"x{i}" for i in range(len(self.lower_bounds_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    X: pd.DataFrame,
    cfg: ExperimentConfig,
    standardize_numeric: bool,
) -> ColumnTransformer:
    """
    Create a leakage-safe preprocessing pipeline.
    This object is fit inside each CV/train fold, never on the full dataset.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(cfg.winsor_lower_q, cfg.winsor_upper_q)),
    ]
    if standardize_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
