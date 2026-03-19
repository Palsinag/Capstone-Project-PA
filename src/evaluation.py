"""Evaluation and robustness metrics for thesis experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import ExperimentConfig
from src.modeling import bootstrap_metric_cis, compute_regression_metrics, early_warning_metrics


def assign_risk_category(values: np.ndarray, risk_band_edges: list[float]) -> np.ndarray:
    """Map continuous overrun predictions into thesis risk bands."""
    low_edge, high_edge = risk_band_edges
    return np.where(
        values < low_edge,
        "low_risk",
        np.where(values < high_edge, "moderate_risk", "high_risk"),
    )


def evaluate_all_models(
    y_test: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for target, pred_by_model in predictions.items():
        y_true = y_test[target].to_numpy()
        for model_name, y_pred in pred_by_model.items():
            metrics = compute_regression_metrics(y_true, y_pred)
            cis = bootstrap_metric_cis(
                y_true,
                y_pred,
                n_boot=cfg.bootstrap_iterations,
                random_state=cfg.random_state,
            )
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    **metrics,
                    "rmse_ci_low": cis["rmse"][0],
                    "rmse_ci_high": cis["rmse"][1],
                    "mae_ci_low": cis["mae"][0],
                    "mae_ci_high": cis["mae"][1],
                    "mape_ci_low": cis["mape"][0],
                    "mape_ci_high": cis["mape"][1],
                    "r2_ci_low": cis["r2"][0],
                    "r2_ci_high": cis["r2"][1],
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "rmse"], ascending=[True, True])


def evaluate_early_warning(
    y_test: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for target, pred_by_model in predictions.items():
        y_true = y_test[target].to_numpy()
        for model_name, y_pred in pred_by_model.items():
            ew = early_warning_metrics(y_true, y_pred, cfg.high_risk_eval_thresholds)
            ew["target"] = target
            ew["model"] = model_name
            rows.append(ew)
    return pd.concat(rows, ignore_index=True)


def summarize_risk_categories(
    y_test: pd.DataFrame,
    predictions: Dict[str, Dict[str, np.ndarray]],
    cfg: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create decision-oriented risk tiers from actual and predicted overruns."""
    detailed_rows = []
    summary_rows = []

    for target, pred_by_model in predictions.items():
        actual_values = y_test[target].to_numpy()
        actual_labels = assign_risk_category(actual_values, cfg.risk_band_edges)
        for model_name, pred_values in pred_by_model.items():
            predicted_labels = assign_risk_category(pred_values, cfg.risk_band_edges)

            detail = pd.DataFrame(
                {
                    "target": target,
                    "model": model_name,
                    "actual_overrun_pct": actual_values,
                    "predicted_overrun_pct": pred_values,
                    "actual_risk_category": actual_labels,
                    "predicted_risk_category": predicted_labels,
                }
            )
            detailed_rows.append(detail)

            summary = (
                detail.groupby("predicted_risk_category", as_index=False)
                .size()
                .rename(columns={"size": "n_projects"})
            )
            summary["share_projects"] = summary["n_projects"] / len(detail)
            summary["target"] = target
            summary["model"] = model_name
            summary_rows.append(summary)

    detailed_df = pd.concat(detailed_rows, ignore_index=True)
    summary_df = pd.concat(summary_rows, ignore_index=True)
    return detailed_df, summary_df


def feature_importance_stability(
    best_estimators,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Compute fold-based feature-importance rank stability for RF and XGBoost."""
    records = []
    cv = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    targets = ["cost_overrun_pct", "schedule_overrun_pct"]

    for target in targets:
        y_target = y[target].to_numpy()
        for model_name in ["random_forest", "xgboost"]:
            pipeline_template = best_estimators[target][model_name]
            rank_tables = []

            for fold_idx, (tr_idx, _va_idx) in enumerate(cv.split(X), start=1):
                X_tr = X.iloc[tr_idx]
                y_tr = y_target[tr_idx]
                fitted = pipeline_template.fit(X_tr, y_tr)

                pre = fitted.named_steps["preprocessor"]
                model = fitted.named_steps["model"]
                names = pre.get_feature_names_out()
                importances = model.feature_importances_
                rank_df = pd.DataFrame(
                    {
                        "feature": names,
                        "importance": importances,
                        "rank": pd.Series(importances).rank(ascending=False, method="dense"),
                        "fold": fold_idx,
                    }
                )
                rank_tables.append(rank_df)

            all_ranks = pd.concat(rank_tables, ignore_index=True)
            grouped = all_ranks.groupby("feature", as_index=False).agg(
                mean_rank=("rank", "mean"),
                std_rank=("rank", "std"),
                mean_importance=("importance", "mean"),
            )
            grouped["target"] = target
            grouped["model"] = model_name
            records.append(grouped)

    out = pd.concat(records, ignore_index=True)
    return out.sort_values(["target", "model", "mean_rank"])


def compare_model_rankings(
    base_cv_df: pd.DataFrame,
    alt_cv_df: pd.DataFrame,
    base_label: str,
    alt_label: str,
) -> pd.DataFrame:
    """Compare model ranking stability across two experimental sample sizes."""
    base_rank = base_cv_df.copy()
    alt_rank = alt_cv_df.copy()

    base_rank["rank"] = base_rank.groupby("target")["cv_rmse_mean"].rank(method="dense")
    alt_rank["rank"] = alt_rank.groupby("target")["cv_rmse_mean"].rank(method="dense")

    merged = base_rank.merge(
        alt_rank[["target", "model", "cv_rmse_mean", "rank"]],
        on=["target", "model"],
        suffixes=(f"_{base_label}", f"_{alt_label}"),
    )
    merged["rank_shift"] = merged[f"rank_{alt_label}"] - merged[f"rank_{base_label}"]
    return merged.sort_values(["target", f"rank_{base_label}"])
