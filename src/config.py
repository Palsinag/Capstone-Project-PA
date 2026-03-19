"""Global configuration for reproducible thesis experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Repository-relative base path for portable GitHub clones and local runs.
BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    base_dir: Path = field(default_factory=lambda: BASE_DIR)
    random_state: int = 42
    python_hash_seed: int = 42
    n_projects: int = 2500
    test_size: float = 0.2
    cv_folds: int = 5
    bootstrap_iterations: int = 1000
    missing_rate: float = 0.02
    outlier_rate: float = 0.01
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99
    high_risk_eval_thresholds: List[float] = field(default_factory=lambda: [0.10, 0.15])
    risk_band_edges: List[float] = field(default_factory=lambda: [0.05, 0.15])
    rf_random_search_iter: int = 30
    xgb_random_search_iter: int = 40
    contract_types: List[str] = field(
        default_factory=lambda: ["fixed_price", "cost_reimbursable", "target_cost", "hybrid_other"]
    )
    contract_probs: List[float] = field(default_factory=lambda: [0.45, 0.20, 0.25, 0.10])
    # Joint probabilities over (scale_tier x complexity_tier) to enforce heterogeneous overlap.
    # Rows: small, medium, large. Columns: low, moderate, high.
    # Non-zero entries in all cells avoid deterministic assignments.
    scale_complexity_joint_probs: List[List[float]] = field(
        default_factory=lambda: [
            [0.12, 0.17, 0.07],
            [0.09, 0.19, 0.12],
            [0.04, 0.11, 0.09],
        ]
    )
    include_optional_planning_features: bool = True
    data_root: Path = field(default_factory=lambda: BASE_DIR / "data")
    data_raw_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    data_processed_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")
    output_root: Path = field(default_factory=lambda: BASE_DIR / "outputs")
    output_tables_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs" / "tables")
    output_figures_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs" / "figures")
    output_models_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs" / "models")
    notebooks_dir: Path = field(default_factory=lambda: BASE_DIR / "notebooks")
    sample_size_robustness_enabled: bool = False
    robustness_n_projects: int = 5000
    robustness_rf_random_search_iter: int = 10
    robustness_xgb_random_search_iter: int = 12
    shap_dependence_top_n: int = 2

    def __post_init__(self) -> None:
        base_dir_value = self.base_dir if isinstance(self.base_dir, Path) else Path(self.base_dir)
        if not base_dir_value.is_absolute():
            base_dir_value = (BASE_DIR / base_dir_value).resolve()
        object.__setattr__(self, "base_dir", base_dir_value)

        path_fields = [
            "data_root",
            "data_raw_dir",
            "data_processed_dir",
            "output_root",
            "output_tables_dir",
            "output_figures_dir",
            "output_models_dir",
            "notebooks_dir",
        ]
        for field_name in path_fields:
            value = getattr(self, field_name)
            path_value = value if isinstance(value, Path) else Path(value)
            if not path_value.is_absolute():
                path_value = (base_dir_value / path_value).resolve()
            object.__setattr__(self, field_name, path_value)


TIER_EFFECTS: Dict[str, Dict[str, float]] = {
    "small": {"budget_mu": 14.5, "duration_mu": 4.6, "complexity_shift": -0.10},
    "medium": {"budget_mu": 15.8, "duration_mu": 5.1, "complexity_shift": 0.00},
    "large": {"budget_mu": 17.2, "duration_mu": 5.7, "complexity_shift": 0.12},
}

COMPLEXITY_EFFECTS: Dict[str, Dict[str, float]] = {
    "low": {"beta_shift": -0.08, "uncertainty_shift": -0.10},
    "moderate": {"beta_shift": 0.00, "uncertainty_shift": 0.00},
    "high": {"beta_shift": 0.10, "uncertainty_shift": 0.12},
}
