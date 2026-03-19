"""Synthetic project dataset generation aligned with thesis assumptions."""

from __future__ import annotations

from itertools import product
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import COMPLEXITY_EFFECTS, TIER_EFFECTS, ExperimentConfig


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def _sample_beta_with_shift(
    rng: np.random.Generator,
    size: int,
    base_a: float,
    base_b: float,
    shift: np.ndarray,
) -> np.ndarray:
    raw = rng.beta(base_a, base_b, size=size)
    return _clip01(raw + shift)


def _sample_joint_tiers(cfg: ExperimentConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample tier combinations from a joint probability matrix."""
    scale_levels: List[str] = ["small", "medium", "large"]
    complexity_levels: List[str] = ["low", "moderate", "high"]

    probs = np.array(cfg.scale_complexity_joint_probs, dtype=float)
    if probs.shape != (3, 3):
        raise ValueError("scale_complexity_joint_probs must be a 3x3 matrix")
    if np.any(probs < 0):
        raise ValueError("scale_complexity_joint_probs must be non-negative")
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError("scale_complexity_joint_probs must sum to 1.0")

    combos = list(product(scale_levels, complexity_levels))
    probs_flat = probs.reshape(-1)
    sampled_idx = rng.choice(len(combos), size=cfg.n_projects, p=probs_flat)

    sampled_pairs = [combos[i] for i in sampled_idx]
    project_scale_tier = np.array([pair[0] for pair in sampled_pairs], dtype=object)
    project_complexity_tier = np.array([pair[1] for pair in sampled_pairs], dtype=object)
    return project_scale_tier, project_complexity_tier


def generate_synthetic_projects(cfg: ExperimentConfig) -> pd.DataFrame:
    """Generate synthetic projects with planning-phase predictors only."""
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.n_projects

    project_scale_tier, project_complexity_tier = _sample_joint_tiers(cfg, rng)

    tier_budget_mu = np.array([TIER_EFFECTS[t]["budget_mu"] for t in project_scale_tier])
    tier_duration_mu = np.array([TIER_EFFECTS[t]["duration_mu"] for t in project_scale_tier])
    tier_complexity_shift = np.array([TIER_EFFECTS[t]["complexity_shift"] for t in project_scale_tier])

    complexity_beta_shift = np.array([COMPLEXITY_EFFECTS[t]["beta_shift"] for t in project_complexity_tier])
    uncertainty_shift = np.array(
        [COMPLEXITY_EFFECTS[t]["uncertainty_shift"] for t in project_complexity_tier]
    )

    # Scale indicators (log-normal right-skew; larger scale shifts mean upwards, with overlap).
    planned_budget = rng.lognormal(mean=tier_budget_mu, sigma=0.52, size=n)
    planned_duration = rng.lognormal(mean=tier_duration_mu, sigma=0.40, size=n)

    # Complexity proxies (beta on [0, 1], shifted by complexity tier with overlap preserved).
    scope_variability_index = _sample_beta_with_shift(
        rng,
        n,
        base_a=2.3,
        base_b=5.2,
        shift=complexity_beta_shift + 0.6 * tier_complexity_shift,
    )
    stakeholder_multiplicity_index = _sample_beta_with_shift(
        rng,
        n,
        base_a=2.6,
        base_b=4.3,
        shift=0.9 * complexity_beta_shift,
    )

    # Uncertainty indices (beta on [0, 1], weakly linked to complexity; mostly independent of size).
    financial_uncertainty = _sample_beta_with_shift(
        rng,
        n,
        base_a=2.0,
        base_b=3.8,
        shift=0.85 * uncertainty_shift + 0.03 * np.tanh((tier_budget_mu - tier_budget_mu.mean())),
    )
    technical_uncertainty = _sample_beta_with_shift(
        rng,
        n,
        base_a=2.1,
        base_b=3.6,
        shift=0.9 * uncertainty_shift,
    )
    environmental_uncertainty = _sample_beta_with_shift(
        rng,
        n,
        base_a=1.9,
        base_b=4.0,
        shift=0.8 * uncertainty_shift,
    )

    # Governance / planning quality.
    planning_maturity_score = _sample_beta_with_shift(
        rng,
        n,
        base_a=4.3,
        base_b=2.2,
        shift=-(0.7 * complexity_beta_shift),
    )
    contract_type = rng.choice(cfg.contract_types, size=n, p=cfg.contract_probs)

    data = {
        "project_scale_tier": project_scale_tier,
        "project_complexity_tier": project_complexity_tier,
        "planned_budget": planned_budget,
        "planned_duration": planned_duration,
        "scope_variability_index": scope_variability_index,
        "stakeholder_multiplicity_index": stakeholder_multiplicity_index,
        "financial_uncertainty": financial_uncertainty,
        "technical_uncertainty": technical_uncertainty,
        "environmental_uncertainty": environmental_uncertainty,
        "planning_maturity_score": planning_maturity_score,
        "contract_type": contract_type,
    }

    if cfg.include_optional_planning_features:
        data["contingency_buffer_ratio"] = _sample_beta_with_shift(
            rng,
            n,
            base_a=2.0,
            base_b=6.5,
            shift=0.08 * complexity_beta_shift,
        )
        data["team_experience_score"] = _sample_beta_with_shift(
            rng,
            n,
            base_a=4.5,
            base_b=2.4,
            shift=-(0.07 * complexity_beta_shift),
        )
        data["supplier_dependency_index"] = _sample_beta_with_shift(
            rng,
            n,
            base_a=2.2,
            base_b=3.8,
            shift=0.06 * complexity_beta_shift,
        )
        data["regulatory_complexity_score"] = _sample_beta_with_shift(
            rng,
            n,
            base_a=2.4,
            base_b=3.5,
            shift=0.12 * uncertainty_shift,
        )

    df = pd.DataFrame(data)

    # Dual-target nonlinear, stochastic overrun generation with right-skewed disturbances.
    # Design goal: moderate realism and explainability, not maximizing any single model class.
    log_budget = np.log(df["planned_budget"])
    log_duration = np.log(df["planned_duration"])
    complexity = (df["scope_variability_index"] + df["stakeholder_multiplicity_index"]) / 2.0
    uncertainty = (
        df["financial_uncertainty"] + df["technical_uncertainty"] + df["environmental_uncertainty"]
    ) / 3.0
    governance_penalty = 1.0 - df["planning_maturity_score"]

    contract_risk_map = {
        "fixed_price": 0.00,
        "cost_reimbursable": 0.07,
        "target_cost": -0.01,
        "hybrid_other": 0.03,
    }
    contract_risk = df["contract_type"].map(contract_risk_map).to_numpy()

    optional_component = 0.0
    if cfg.include_optional_planning_features:
        optional_component = (
            0.045 * df["supplier_dependency_index"]
            + 0.040 * df["regulatory_complexity_score"]
            - 0.050 * df["team_experience_score"]
            - 0.030 * df["contingency_buffer_ratio"]
        )

    complexity_x_uncertainty = complexity * uncertainty
    scale_x_maturity = (0.5 * log_budget + 0.5 * log_duration) * df["planning_maturity_score"]
    stakeholder_x_governance = (
        df["stakeholder_multiplicity_index"] * (1.0 - df["planning_maturity_score"])
    )
    scope_x_technical = df["scope_variability_index"] * df["technical_uncertainty"]

    base_signal = (
        0.020
        + 0.038 * (log_budget - log_budget.mean())
        + 0.034 * (log_duration - log_duration.mean())
        + 0.092 * uncertainty
        + 0.074 * complexity
        + 0.070 * complexity_x_uncertainty
        + 0.042 * stakeholder_x_governance
        + 0.036 * scope_x_technical
        - 0.020 * scale_x_maturity
        + 0.040 * np.power(complexity, 2)
        + contract_risk
        + optional_component
    )

    # Right-skewed (non-Gaussian-only) noise to emulate asymmetric overrun behavior.
    skew_noise_cost = rng.gamma(shape=2.0, scale=0.018, size=n) - (2.0 * 0.018)
    skew_noise_schedule = rng.gamma(shape=2.2, scale=0.022, size=n) - (2.2 * 0.022)

    latent_cost_overrun = np.clip(base_signal + skew_noise_cost, -0.15, 1.25)
    latent_schedule_overrun = np.clip(
        1.08 * base_signal
        + 0.032 * stakeholder_x_governance
        + 0.030 * scope_x_technical
        + skew_noise_schedule,
        -0.18,
        1.45,
    )

    # Generate implicit actuals and compute overruns via thesis formula:
    # Y_cost = (ActualCost - PlannedCost) / PlannedCost
    # Y_schedule = (ActualDuration - PlannedDuration) / PlannedDuration
    actual_cost = df["planned_budget"] * (1.0 + latent_cost_overrun)
    actual_duration = df["planned_duration"] * (1.0 + latent_schedule_overrun)
    df["cost_overrun_pct"] = (actual_cost - df["planned_budget"]) / df["planned_budget"]
    df["schedule_overrun_pct"] = (actual_duration - df["planned_duration"]) / df["planned_duration"]

    return df


def inject_mcar_missingness(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """Inject low-rate MCAR missingness into continuous predictors only."""
    rng = np.random.default_rng(cfg.random_state + 7)
    out = df.copy()

    continuous_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    outcome_cols = ["cost_overrun_pct", "schedule_overrun_pct"]
    cols = [c for c in continuous_cols if c not in outcome_cols]

    for col in cols:
        mask = rng.random(len(out)) < cfg.missing_rate
        out.loc[mask, col] = np.nan

    return out


def inject_predictor_outliers(df: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Inject a small amount of realistic extreme values into selected predictors only.
    Targets are explicitly excluded.
    """
    rng = np.random.default_rng(cfg.random_state + 11)
    out = df.copy()

    outlier_cols = [
        "planned_budget",
        "planned_duration",
        "scope_variability_index",
        "stakeholder_multiplicity_index",
        "financial_uncertainty",
        "technical_uncertainty",
        "environmental_uncertainty",
        "planning_maturity_score",
    ]
    optional_cols = [
        "contingency_buffer_ratio",
        "team_experience_score",
        "supplier_dependency_index",
        "regulatory_complexity_score",
    ]
    outlier_cols += [c for c in optional_cols if c in out.columns]

    for col in outlier_cols:
        mask = rng.random(len(out)) < cfg.outlier_rate
        if not mask.any():
            continue

        if col in ["planned_budget", "planned_duration"]:
            # Multiplicative right-tail shocks for scale variables.
            shock = rng.lognormal(mean=0.55, sigma=0.35, size=mask.sum())
            out.loc[mask, col] = out.loc[mask, col].to_numpy() * shock
        else:
            # Push bounded variables near extremes while remaining in [0, 1].
            direction = rng.choice([0, 1], size=mask.sum(), p=[0.35, 0.65])
            base_vals = out.loc[mask, col].to_numpy()
            high_tail = np.clip(base_vals + rng.uniform(0.20, 0.45, size=mask.sum()), 0.0, 1.0)
            low_tail = np.clip(base_vals - rng.uniform(0.15, 0.30, size=mask.sum()), 0.0, 1.0)
            out.loc[mask, col] = np.where(direction == 1, high_tail, low_tail)

    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a compact, theory-driven set of engineered features."""
    out = df.copy()

    # Required log transforms for right-skewed scale variables.
    out["log_planned_budget"] = np.log(out["planned_budget"].clip(lower=1.0))
    out["log_planned_duration"] = np.log(out["planned_duration"].clip(lower=1.0))

    complexity_proxy = (
        out["scope_variability_index"] + out["stakeholder_multiplicity_index"]
    ) / 2.0
    uncertainty_proxy = (
        out["financial_uncertainty"]
        + out["technical_uncertainty"]
        + out["environmental_uncertainty"]
    ) / 3.0
    scale_proxy = (out["log_planned_budget"] + out["log_planned_duration"]) / 2.0

    # Minimal interaction set requested for thesis-defensible feature engineering.
    out["complexity_x_uncertainty"] = complexity_proxy * uncertainty_proxy
    out["scale_x_planning_maturity"] = scale_proxy * out["planning_maturity_score"]
    out["stakeholder_x_governance"] = (
        out["stakeholder_multiplicity_index"] * (1.0 - out["planning_maturity_score"])
    )
    out["scope_x_technical_uncertainty"] = (
        out["scope_variability_index"] * out["technical_uncertainty"]
    )

    # Optional aggregate index retained in addition to original uncertainty variables.
    out["composite_risk_index"] = (
        out["financial_uncertainty"]
        + out["technical_uncertainty"]
        + out["environmental_uncertainty"]
    ) / 3.0

    return out
