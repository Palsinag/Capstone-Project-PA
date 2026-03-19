"""Data and model diagnostics for sanity checks and thesis transparency."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import skew


def dataset_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Run structural checks on generated synthetic data."""
    findings: List[Dict[str, object]] = []

    def add_check(name: str, status: str, detail: str) -> None:
        findings.append({"check": name, "status": status, "detail": detail})

    add_check(
        "row_count",
        "PASS" if len(df) == 2500 else "FAIL",
        f"n_rows={len(df)} (expected 2500)",
    )

    for col in [
        "scope_variability_index",
        "stakeholder_multiplicity_index",
        "financial_uncertainty",
        "technical_uncertainty",
        "environmental_uncertainty",
        "planning_maturity_score",
    ]:
        valid = df[col].between(0.0, 1.0).all()
        add_check(
            f"range_{col}",
            "PASS" if valid else "FAIL",
            f"min={df[col].min():.4f}, max={df[col].max():.4f}",
        )

    cross_tab = pd.crosstab(df["project_scale_tier"], df["project_complexity_tier"])
    all_cells_nonzero = (cross_tab > 0).all().all()
    add_check(
        "strata_coverage",
        "PASS" if all_cells_nonzero else "FAIL",
        f"min_cell_count={int(cross_tab.values.min())}",
    )

    row_props = cross_tab.div(cross_tab.sum(axis=1), axis=0)
    overlap_ok = (row_props.max(axis=1) < 0.90).all()
    add_check(
        "non_deterministic_overlap",
        "PASS" if overlap_ok else "WARN",
        "largest within-row complexity share must stay < 0.90",
    )

    for col in ["planned_budget", "planned_duration", "cost_overrun_pct", "schedule_overrun_pct"]:
        col_skew = float(skew(df[col].to_numpy(), bias=False))
        add_check(
            f"right_skew_{col}",
            "PASS" if col_skew > 0 else "WARN",
            f"skew={col_skew:.4f}",
        )

    corr = float(df[["planned_budget", "financial_uncertainty"]].corr().iloc[0, 1])
    add_check(
        "weak_size_uncertainty_correlation",
        "PASS" if abs(corr) < 0.35 else "WARN",
        f"corr(planned_budget, financial_uncertainty)={corr:.4f}",
    )

    return pd.DataFrame(findings)


def linear_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Optional residual diagnostics table using statsmodels if available."""
    residuals = y_true - y_pred
    rows = [
        {"metric": "residual_mean", "value": float(np.mean(residuals))},
        {"metric": "residual_std", "value": float(np.std(residuals, ddof=1))},
        {"metric": "residual_skew", "value": float(skew(residuals, bias=False))},
    ]

    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import jarque_bera
        import statsmodels.api as sm

        exog = sm.add_constant(y_pred)
        bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog)
        jb_stat, jb_p, _, _ = jarque_bera(residuals)

        rows.extend(
            [
                {"metric": "breusch_pagan_lm", "value": float(bp_lm)},
                {"metric": "breusch_pagan_lm_pvalue", "value": float(bp_lm_p)},
                {"metric": "breusch_pagan_f", "value": float(bp_f)},
                {"metric": "breusch_pagan_f_pvalue", "value": float(bp_f_p)},
                {"metric": "jarque_bera_stat", "value": float(jb_stat)},
                {"metric": "jarque_bera_pvalue", "value": float(jb_p)},
            ]
        )
    except Exception:
        rows.append(
            {
                "metric": "statsmodels_diagnostics",
                "value": np.nan,
                "note": "statsmodels unavailable; only descriptive residual checks computed",
            }
        )

    return pd.DataFrame(rows)
