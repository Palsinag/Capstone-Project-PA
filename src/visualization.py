"""Visualization and explainability outputs for thesis reporting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def plot_metric_bars(metrics_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    for i, target in enumerate(metrics_df["target"].unique()):
        subset = metrics_df[metrics_df["target"] == target].copy()
        sns.barplot(
            data=subset,
            x="model",
            y="rmse",
            hue="model",
            legend=False,
            ax=axes[i],
            palette="deep",
        )
        axes[i].set_title(f"RMSE by model: {target}")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_early_warning(early_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    targets = early_df["target"].unique().tolist()
    for i, target in enumerate(targets):
        subset = early_df[early_df["target"] == target].copy()
        subset = subset.sort_values("threshold")
        sns.lineplot(data=subset, x="threshold", y="f1", marker="o", ax=axes[i], label="F1")
        sns.lineplot(data=subset, x="threshold", y="precision", marker="o", ax=axes[i], label="Precision")
        sns.lineplot(data=subset, x="threshold", y="recall", marker="o", ax=axes[i], label="Recall")
        axes[i].set_title(f"Early-warning sensitivity ({target})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_strata_heatmap(df: pd.DataFrame, out_path: str) -> None:
    ctab = pd.crosstab(df["project_scale_tier"], df["project_complexity_tier"], normalize="index")
    plt.figure(figsize=(8, 6))
    sns.heatmap(ctab, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.title("Complexity distribution within each scale tier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_distributions(df: pd.DataFrame, out_path: str) -> None:
    cols = ["planned_budget", "planned_duration", "cost_overrun_pct", "schedule_overrun_pct"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=40)
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_selected_model_diagnostics(
    y_test: pd.DataFrame,
    selected_predictions: pd.DataFrame,
    out_dir: str,
) -> None:
    """Residual diagnostics for final selected models."""
    for target in selected_predictions["target"].unique():
        subset = selected_predictions[selected_predictions["target"] == target].copy()
        y_true = subset["actual"].to_numpy()
        y_pred = subset["predicted"].to_numpy()
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(residuals, kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f"Residual histogram ({target})")

        axes[1].scatter(y_true, y_pred, alpha=0.6)
        line_min = min(np.min(y_true), np.min(y_pred))
        line_max = max(np.max(y_true), np.max(y_pred))
        axes[1].plot([line_min, line_max], [line_min, line_max], color="black", linestyle="--")
        axes[1].set_xlabel("Actual")
        axes[1].set_ylabel("Predicted")
        axes[1].set_title(f"Predicted vs actual ({target})")

        axes[2].scatter(y_pred, residuals, alpha=0.6)
        axes[2].axhline(0.0, color="black", linestyle="--")
        axes[2].set_xlabel("Fitted values")
        axes[2].set_ylabel("Residuals")
        axes[2].set_title(f"Residuals vs fitted ({target})")

        plt.tight_layout()
        plt.savefig(f"{out_dir}/selected_model_diagnostics_{target}.png", dpi=200)
        plt.close()


def plot_feature_importance_rankings(
    importance_df: pd.DataFrame,
    value_col: str,
    out_path: str,
    top_n: int = 10,
) -> None:
    """Plot top feature rankings for each tree model and target."""
    if importance_df.empty:
        return

    plot_df = (
        importance_df.sort_values(["target", "model", value_col], ascending=[True, True, False])
        .groupby(["target", "model"], as_index=False)
        .head(top_n)
        .copy()
    )

    n_panels = len(plot_df[["target", "model"]].drop_duplicates())
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (target, model_name) in zip(
        axes,
        plot_df[["target", "model"]].drop_duplicates().itertuples(index=False, name=None),
    ):
        subset = plot_df[(plot_df["target"] == target) & (plot_df["model"] == model_name)].copy()
        subset = subset.sort_values(value_col, ascending=True)
        sns.barplot(
            data=subset,
            x=value_col,
            y="feature",
            hue="feature",
            legend=False,
            ax=ax,
            palette="deep",
        )
        ax.set_title(f"Top {top_n} features: {model_name} | {target}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
