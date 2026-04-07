"""Variance Inflation Factor analysis for the Linear Regression feature matrix."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig
from src.preprocessing import Winsorizer
from src.utils import ensure_directories


TARGET_COLUMNS = ["cost_overrun_pct", "schedule_overrun_pct"]


def _format_feature_names(feature_names: np.ndarray) -> list[str]:
    """Remove transformer prefixes to keep the output table thesis-friendly."""
    cleaned = []
    for name in feature_names.astype(str):
        if "__" in name:
            cleaned.append(name.split("__", 1)[1])
        else:
            cleaned.append(name)
    return cleaned


def load_training_features(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Reconstruct the same training feature matrix used for model fitting.
    Only the training split is used to avoid any leakage from the test set.
    """
    dataset_path = cfg.data_processed_dir / "final_synthetic_dataset.csv"
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=TARGET_COLUMNS)
    y = df[TARGET_COLUMNS]

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    return X_train


def build_linear_regression_design_matrix(
    X_train: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """
    Recreate the Linear Regression design matrix for VIF analysis using
    training-only fitting and dropping one categorical reference level.

    This keeps preprocessing aligned with the thesis pipeline while avoiding
    perfect multicollinearity from full dummy encoding.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(cfg.winsor_lower_q, cfg.winsor_upper_q)),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    X_train_processed = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(
        X_train_processed,
        columns=_format_feature_names(feature_names),
        index=X_train.index,
    )


def compute_vif_table(X_design: pd.DataFrame) -> pd.DataFrame:
    """Compute VIF values for each column in the final training design matrix."""
    vif_rows = []
    values = X_design.to_numpy(dtype=float)

    for idx, feature_name in enumerate(X_design.columns):
        vif_rows.append(
            {
                "feature": feature_name,
                "vif": float(variance_inflation_factor(values, idx)),
            }
        )

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def summarize_multicollinearity(vif_df: pd.DataFrame) -> str:
    """Create a short interpretation suitable for thesis reporting."""
    max_vif = float(vif_df["vif"].max())

    if max_vif < 5:
        concern = "No material multicollinearity concern is indicated."
        level = "low"
    elif max_vif <= 10:
        concern = "Moderate multicollinearity may be present and should be acknowledged."
        level = "moderate"
    else:
        concern = "High multicollinearity is present and may affect coefficient stability."
        level = "high"

    return (
        f"Maximum VIF: {max_vif:.3f}\n"
        f"Interpretation: the highest observed multicollinearity falls in the {level} range.\n"
        f"Conclusion: {concern}"
    )


def plot_vif_results(vif_df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """
    Plot the highest VIF values in a thesis-friendly horizontal bar chart.
    Infinite VIF values are clipped for plotting and labeled explicitly.
    """
    plot_df = vif_df.head(top_n).copy()
    finite_vifs = plot_df.loc[np.isfinite(plot_df["vif"]), "vif"]
    clip_value = float(finite_vifs.max()) if not finite_vifs.empty else 10.0
    clip_value = max(clip_value, 10.0)
    plot_df["vif_plot"] = plot_df["vif"].replace(np.inf, clip_value * 1.1)
    plot_df["vif_label"] = plot_df["vif"].apply(lambda value: "inf" if np.isinf(value) else f"{value:.2f}")

    plt.figure(figsize=(10, 7))
    bars = plt.barh(plot_df["feature"], plot_df["vif_plot"], color="#4C78A8")
    plt.xlabel("Variance Inflation Factor (VIF)")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} VIF Values in the Linear Regression Training Matrix")
    plt.gca().invert_yaxis()

    for bar, label in zip(bars, plot_df["vif_label"]):
        plt.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_vif_analysis(output_path: Path | None = None) -> pd.DataFrame:
    """Run the full VIF workflow and save the resulting table."""
    cfg = ExperimentConfig()
    exploratory_tables_dir = cfg.output_root / "exploratory" / "tables"
    exploratory_figures_dir = cfg.output_root / "exploratory" / "figures"
    thesis_tables_dir = cfg.output_root / "thesis_tables"
    thesis_figures_dir = cfg.output_root / "thesis_figures"
    ensure_directories(
        [
            cfg.output_tables_dir,
            exploratory_tables_dir,
            exploratory_figures_dir,
            thesis_tables_dir,
            thesis_figures_dir,
        ]
    )

    if output_path is None:
        output_path = exploratory_tables_dir / "vif_results.csv"
    figure_path = exploratory_figures_dir / "vif_results_top15.png"
    thesis_table_path = thesis_tables_dir / "table_4_6_vif_results.csv"
    thesis_figure_png_path = thesis_figures_dir / "figure_4_12_vif_results.png"

    X_train = load_training_features(cfg)
    X_design = build_linear_regression_design_matrix(X_train, cfg)
    vif_df = compute_vif_table(X_design)
    vif_df.to_csv(output_path, index=False)
    vif_df.to_csv(thesis_table_path, index=False)
    plot_vif_results(vif_df, figure_path)
    plot_vif_results(vif_df, thesis_figure_png_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(vif_df)
    print()
    print(summarize_multicollinearity(vif_df))
    print(f"Saved VIF results to: {output_path}")
    print(f"Saved VIF figure to: {figure_path}")
    print(f"Saved thesis VIF table to: {thesis_table_path}")
    print(f"Saved thesis VIF figure to: {thesis_figure_png_path}")

    return vif_df


if __name__ == "__main__":
    run_vif_analysis()
