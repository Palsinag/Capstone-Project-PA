"""Modeling and evaluation workflow for cost and schedule overrun prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import ExperimentConfig
from src.preprocessing import build_preprocessor


@dataclass
class FitArtifacts:
    best_estimators: Dict[str, Dict[str, Pipeline]]
    cv_scores: pd.DataFrame
    best_hyperparameters: pd.DataFrame


def _rmse(y_true, y_pred) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": _safe_mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int,
    random_state: int,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    samples = {"rmse": [], "mae": [], "mape": [], "r2": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        m = compute_regression_metrics(yt, yp)
        for k, v in m.items():
            samples[k].append(v)

    ci = {}
    for k, vals in samples.items():
        low, high = np.quantile(vals, [0.025, 0.975])
        ci[k] = (float(low), float(high))
    return ci


def early_warning_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        yt = (y_true >= threshold).astype(int)
        yp = (y_pred >= threshold).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return pd.DataFrame(rows)


def train_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: ExperimentConfig,
) -> Tuple[FitArtifacts, Dict[str, Dict[str, np.ndarray]], pd.DataFrame, pd.DataFrame]:
    """Train and tune linear, random forest, and XGBoost models for each target."""
    # Hold-out test split is untouched during tuning to avoid optimistic bias.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )

    cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    targets = ["cost_overrun_pct", "schedule_overrun_pct"]
    models = ["linear_regression", "random_forest", "xgboost"]

    best_estimators: Dict[str, Dict[str, Pipeline]] = {t: {} for t in targets}
    cv_rows: List[dict] = []
    hyperparameter_rows: List[dict] = []

    for target in targets:
        y_target = y_train[target]

        # Linear baseline for methodological interpretability and comparison.
        linear_pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, cfg, standardize_numeric=True)),
                ("model", LinearRegression()),
            ]
        )
        lin_scores = cross_val_score(
            linear_pipe,
            X_train,
            y_target,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        linear_pipe.fit(X_train, y_target)
        best_estimators[target]["linear_regression"] = linear_pipe
        cv_rows.append(
            {
                "target": target,
                "model": "linear_regression",
                "cv_rmse_mean": float(-lin_scores.mean()),
                "cv_rmse_std": float(lin_scores.std()),
            }
        )
        hyperparameter_rows.append(
            {
                "target": target,
                "model": "linear_regression",
                "best_hyperparameters_json": json.dumps({"tuned": False}),
            }
        )

        # Random Forest captures nonlinear interactions with robust variance control.
        rf_pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, cfg, standardize_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(random_state=cfg.random_state, n_jobs=-1),
                ),
            ]
        )
        rf_search = RandomizedSearchCV(
            estimator=rf_pipe,
            param_distributions={
                "model__n_estimators": randint(200, 700),
                "model__max_depth": randint(4, 20),
                "model__min_samples_split": randint(2, 20),
                "model__min_samples_leaf": randint(1, 12),
                "model__max_features": ["sqrt", "log2", None],
            },
            n_iter=cfg.rf_random_search_iter,
            cv=cv,
            random_state=cfg.random_state,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rf_search.fit(X_train, y_target)
        best_estimators[target]["random_forest"] = rf_search.best_estimator_
        cv_rows.append(
            {
                "target": target,
                "model": "random_forest",
                "cv_rmse_mean": float(-rf_search.best_score_),
                "cv_rmse_std": float(rf_search.cv_results_["std_test_score"][rf_search.best_index_]),
            }
        )
        hyperparameter_rows.append(
            {
                "target": target,
                "model": "random_forest",
                "best_hyperparameters_json": json.dumps(rf_search.best_params_, sort_keys=True),
            }
        )

        # XGBoost handles higher-order nonlinearities with regularized boosting.
        xgb_pipe = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, cfg, standardize_numeric=False)),
                (
                    "model",
                    XGBRegressor(
                        random_state=cfg.random_state,
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        xgb_search = RandomizedSearchCV(
            estimator=xgb_pipe,
            param_distributions={
                "model__n_estimators": randint(150, 700),
                "model__max_depth": randint(3, 12),
                "model__learning_rate": uniform(0.01, 0.25),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_alpha": uniform(0.0, 1.0),
                "model__reg_lambda": uniform(0.2, 2.0),
            },
            n_iter=cfg.xgb_random_search_iter,
            cv=cv,
            random_state=cfg.random_state,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        xgb_search.fit(X_train, y_target)
        best_estimators[target]["xgboost"] = xgb_search.best_estimator_
        cv_rows.append(
            {
                "target": target,
                "model": "xgboost",
                "cv_rmse_mean": float(-xgb_search.best_score_),
                "cv_rmse_std": float(xgb_search.cv_results_["std_test_score"][xgb_search.best_index_]),
            }
        )
        hyperparameter_rows.append(
            {
                "target": target,
                "model": "xgboost",
                "best_hyperparameters_json": json.dumps(xgb_search.best_params_, sort_keys=True),
            }
        )

    cv_df = pd.DataFrame(cv_rows).sort_values(["target", "cv_rmse_mean"], ascending=[True, True])
    hyperparameter_df = pd.DataFrame(hyperparameter_rows).sort_values(["target", "model"])

    # Predict using all trained models on test set for full comparison.
    test_predictions: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in targets}
    for target in targets:
        for model_name in models:
            test_predictions[target][model_name] = best_estimators[target][model_name].predict(X_test)

    return (
        FitArtifacts(
            best_estimators=best_estimators,
            cv_scores=cv_df,
            best_hyperparameters=hyperparameter_df,
        ),
        test_predictions,
        y_test,
        X_test,
    )
