"""Utility helpers for reproducibility, structure, and guardrails."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd


def set_global_seed(seed: int, python_hash_seed: int) -> None:
    """Set deterministic seeds for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(python_hash_seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(paths: Iterable[str]) -> None:
    """Create output directories if they do not already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def assert_no_post_initiation_features(X: pd.DataFrame) -> None:
    """
    Guardrail for thesis scope:
    only planning-phase variables are allowed in the feature matrix.
    """
    banned_tokens = [
        "actual_",
        "earned_value",
        "cpi",
        "spi",
        "progress",
        "percent_complete",
        "post_initiation",
    ]
    offending = []
    for col in X.columns:
        lower = col.lower()
        if any(token in lower for token in banned_tokens):
            offending.append(col)
    if offending:
        raise ValueError(
            "Detected non-planning (post-initiation) feature columns: "
            + ", ".join(offending)
        )


def build_variable_dictionary(include_optional_planning_features: bool) -> pd.DataFrame:
    """Create a thesis-friendly variable dictionary for generated data."""
    rows = [
        ("project_scale_tier", "categorical", "Structural project scale tier used for stratified generation."),
        ("project_complexity_tier", "categorical", "Structural project complexity tier used for stratified generation."),
        ("planned_budget", "continuous", "Planned project budget observed at initiation."),
        ("planned_duration", "continuous", "Planned project duration observed at initiation."),
        ("scope_variability_index", "continuous_[0,1]", "Early-stage scope variability proxy."),
        ("stakeholder_multiplicity_index", "continuous_[0,1]", "Proxy for stakeholder coordination complexity."),
        ("financial_uncertainty", "continuous_[0,1]", "Financial uncertainty index available at planning stage."),
        ("technical_uncertainty", "continuous_[0,1]", "Technical uncertainty index available at planning stage."),
        ("environmental_uncertainty", "continuous_[0,1]", "Environmental uncertainty index available at planning stage."),
        ("planning_maturity_score", "continuous_[0,1]", "Planning/governance quality score."),
        ("contract_type", "categorical", "Planned contractual regime."),
        ("log_planned_budget", "engineered_continuous", "Log transform of planned budget."),
        ("log_planned_duration", "engineered_continuous", "Log transform of planned duration."),
        ("complexity_x_uncertainty", "engineered_continuous", "Interaction between overall complexity and uncertainty."),
        ("scale_x_planning_maturity", "engineered_continuous", "Interaction between project scale and planning maturity."),
        ("stakeholder_x_governance", "engineered_continuous", "Interaction between stakeholder multiplicity and weaker governance."),
        ("scope_x_technical_uncertainty", "engineered_continuous", "Interaction between scope variability and technical uncertainty."),
        ("composite_risk_index", "engineered_continuous", "Average of the three uncertainty indices."),
        ("cost_overrun_pct", "target_continuous", "Percentage cost overrun relative to planned cost."),
        ("schedule_overrun_pct", "target_continuous", "Percentage schedule overrun relative to planned duration."),
    ]

    if include_optional_planning_features:
        rows.extend(
            [
                ("contingency_buffer_ratio", "continuous_[0,1]", "Planned contingency allowance ratio."),
                ("team_experience_score", "continuous_[0,1]", "Planning-stage project team experience score."),
                ("supplier_dependency_index", "continuous_[0,1]", "Dependence on key suppliers at planning stage."),
                ("regulatory_complexity_score", "continuous_[0,1]", "Expected regulatory complexity before execution."),
            ]
        )

    return pd.DataFrame(rows, columns=["variable", "type", "description"])


def build_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate compact summary statistics suitable for a thesis appendix/table."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.agg(["mean", "std", "min", "median", "max"]).T.reset_index()
    stats = stats.rename(columns={"index": "variable"})
    stats["missing_rate"] = numeric_df.isna().mean().values
    stats["skewness"] = numeric_df.skew().values
    return stats


def save_run_configuration(cfg, out_path: str) -> None:
    """Persist experiment parameters in JSON format."""
    serialized = {}
    cfg_dict = asdict(cfg)
    base_dir = cfg_dict.get("base_dir")
    for key, value in cfg_dict.items():
        if isinstance(value, Path):
            try:
                serialized[key] = str(value.relative_to(base_dir))
            except Exception:
                serialized[key] = str(value)
        else:
            serialized[key] = value
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, sort_keys=True)
