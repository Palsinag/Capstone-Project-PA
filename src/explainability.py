"""Explainability outputs (SHAP) for trained tree-based models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_shap_values(best_estimators, X_test: pd.DataFrame, out_dir: str) -> None:
    """Generate SHAP summary plots for tree models if SHAP is installed."""
    try:
        import shap
    except ImportError:
        return

    for target in ["cost_overrun_pct", "schedule_overrun_pct"]:
        for model_name in ["random_forest", "xgboost"]:
            pipe = best_estimators[target][model_name]
            pre = pipe.named_steps["preprocessor"]
            model = pipe.named_steps["model"]
            X_transformed = pre.transform(X_test)
            feature_names = pre.get_feature_names_out()

            # Cap rows for SHAP to keep runtime reasonable and reproducible.
            X_shap = X_transformed[:500] if X_transformed.shape[0] > 500 else X_transformed

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_shap,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f"SHAP summary: {model_name} | {target}")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/shap_{model_name}_{target}.png", dpi=200)
            plt.close()


def compute_tree_feature_importance_rankings(best_estimators) -> pd.DataFrame:
    """Collect global feature importance rankings from final RF/XGBoost models."""
    rows = []
    for target in ["cost_overrun_pct", "schedule_overrun_pct"]:
        for model_name in ["random_forest", "xgboost"]:
            pipe = best_estimators[target][model_name]
            pre = pipe.named_steps["preprocessor"]
            model = pipe.named_steps["model"]
            feature_names = pre.get_feature_names_out()
            importances = model.feature_importances_

            frame = pd.DataFrame(
                {
                    "target": target,
                    "model": model_name,
                    "feature": feature_names,
                    "importance": importances,
                }
            )
            frame["rank"] = frame["importance"].rank(ascending=False, method="dense")
            rows.append(frame)

    return pd.concat(rows, ignore_index=True).sort_values(["target", "model", "rank"])


def compute_shap_outputs(
    best_estimators,
    X_test: pd.DataFrame,
    out_dir: str,
    top_n_dependence: int,
) -> pd.DataFrame:
    """
    Generate SHAP summaries and dependence plots for final tree-based models.
    Returns mean absolute SHAP rankings as a global importance table.
    """
    try:
        import shap
    except ImportError:
        return pd.DataFrame()

    rows = []
    for target in ["cost_overrun_pct", "schedule_overrun_pct"]:
        for model_name in ["random_forest", "xgboost"]:
            pipe = best_estimators[target][model_name]
            pre = pipe.named_steps["preprocessor"]
            model = pipe.named_steps["model"]
            X_transformed = pre.transform(X_test)
            feature_names = pre.get_feature_names_out()
            X_shap = X_transformed[:500] if X_transformed.shape[0] > 500 else X_transformed

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            mean_abs = np.abs(shap_values).mean(axis=0)

            rank_df = pd.DataFrame(
                {
                    "target": target,
                    "model": model_name,
                    "feature": feature_names,
                    "mean_abs_shap": mean_abs,
                }
            ).sort_values("mean_abs_shap", ascending=False)
            rank_df["rank"] = np.arange(1, len(rank_df) + 1)
            rows.append(rank_df)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_shap,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f"SHAP summary: {model_name} | {target}")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/shap_{model_name}_{target}.png", dpi=200)
            plt.close()

            top_features = rank_df["feature"].head(top_n_dependence).tolist()
            for feature in top_features:
                feature_index = list(feature_names).index(feature)
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    feature_index,
                    shap_values,
                    X_shap,
                    feature_names=feature_names,
                    show=False,
                )
                plt.title(f"SHAP dependence: {feature} | {model_name} | {target}")
                safe_feature = feature.replace("/", "_").replace(" ", "_")
                plt.tight_layout()
                plt.savefig(
                    f"{out_dir}/shap_dependence_{model_name}_{target}_{safe_feature}.png",
                    dpi=200,
                )
                plt.close()

    return pd.concat(rows, ignore_index=True).sort_values(["target", "model", "rank"])


def summarize_driver_overlap(importance_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Summarize overlap in top drivers between cost and schedule outcomes."""
    rows = []
    if importance_df.empty:
        return pd.DataFrame()

    sort_col = "rank" if "rank" in importance_df.columns else "importance"
    ascending = sort_col == "rank"

    for model_name in importance_df["model"].unique():
        cost_top = (
            importance_df[
                (importance_df["model"] == model_name) & (importance_df["target"] == "cost_overrun_pct")
            ]
            .sort_values(sort_col, ascending=ascending)
        )
        schedule_top = (
            importance_df[
                (importance_df["model"] == model_name)
                & (importance_df["target"] == "schedule_overrun_pct")
            ]
            .sort_values(sort_col, ascending=ascending)
        )
        cost_features = cost_top["feature"].head(top_n).tolist()
        schedule_features = schedule_top["feature"].head(top_n).tolist()
        overlap = sorted(set(cost_features).intersection(schedule_features))
        rows.append(
            {
                "model": model_name,
                "top_n": top_n,
                "n_overlapping_features": len(overlap),
                "overlapping_features": ", ".join(overlap),
            }
        )

    return pd.DataFrame(rows)
