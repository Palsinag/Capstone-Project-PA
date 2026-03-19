"""Entrypoint for thesis ML workflow."""

from __future__ import annotations

from dataclasses import replace

import joblib
import pandas as pd

from src.config import ExperimentConfig
from src.data_generation import (
    add_engineered_features,
    generate_synthetic_projects,
    inject_mcar_missingness,
    inject_predictor_outliers,
)
from src.diagnostics import dataset_sanity_checks, linear_residual_diagnostics
from src.evaluation import (
    compare_model_rankings,
    evaluate_all_models,
    evaluate_early_warning,
    feature_importance_stability,
    summarize_risk_categories,
)
from src.explainability import (
    compute_shap_outputs,
    compute_tree_feature_importance_rankings,
    summarize_driver_overlap,
)
from src.modeling import train_models
from src.utils import (
    assert_no_post_initiation_features,
    build_summary_statistics,
    build_variable_dictionary,
    ensure_directories,
    save_run_configuration,
    set_global_seed,
)
from src.visualization import (
    plot_early_warning,
    plot_feature_importance_rankings,
    plot_feature_distributions,
    plot_metric_bars,
    plot_selected_model_diagnostics,
    plot_strata_heatmap,
)


def main() -> None:
    cfg = ExperimentConfig()
    set_global_seed(cfg.random_state, cfg.python_hash_seed)

    ensure_directories(
        [
            cfg.data_root,
            cfg.data_raw_dir,
            cfg.data_processed_dir,
            cfg.output_root,
            cfg.output_tables_dir,
            cfg.output_figures_dir,
            cfg.output_models_dir,
            cfg.notebooks_dir,
        ]
    )

    # 1) Data generation and intermediate persistence.
    raw_df = generate_synthetic_projects(cfg)
    raw_df.to_csv(cfg.data_raw_dir / "synthetic_projects_raw.csv", index=False)

    sanity_df = dataset_sanity_checks(raw_df)
    sanity_df.to_csv(cfg.output_tables_dir / "data_sanity_checks.csv", index=False)

    outlier_df = inject_predictor_outliers(raw_df, cfg)
    outlier_df.to_csv(cfg.data_processed_dir / "synthetic_projects_with_outliers.csv", index=False)

    missing_df = inject_mcar_missingness(outlier_df, cfg)
    missing_df.to_csv(cfg.data_processed_dir / "synthetic_projects_with_missingness.csv", index=False)

    model_df = add_engineered_features(missing_df)
    model_df.to_csv(cfg.data_processed_dir / "synthetic_projects_model_ready.csv", index=False)
    model_df.to_csv(cfg.data_processed_dir / "final_synthetic_dataset.csv", index=False)
    variable_dictionary_df = build_variable_dictionary(cfg.include_optional_planning_features)
    summary_statistics_df = build_summary_statistics(model_df)
    variable_dictionary_df.to_csv(cfg.output_tables_dir / "variable_dictionary.csv", index=False)
    summary_statistics_df.to_csv(cfg.output_tables_dir / "summary_statistics.csv", index=False)
    save_run_configuration(cfg, cfg.output_tables_dir / "run_config.json")
    pd.DataFrame(
        [{"random_state": cfg.random_state, "python_hash_seed": cfg.python_hash_seed}]
    ).to_csv(cfg.output_tables_dir / "random_seed_record.csv", index=False)

    # 2) Modeling.
    targets = ["cost_overrun_pct", "schedule_overrun_pct"]
    X = model_df.drop(columns=targets)
    y = model_df[targets]
    assert_no_post_initiation_features(X)

    artifacts, test_predictions, y_test, X_test = train_models(X, y, cfg)
    train_n = len(X) - len(X_test)
    test_n = len(X_test)

    validation_design_df = pd.DataFrame(
        [
            {
                "n_total": len(X),
                "train_size_ratio": 1.0 - cfg.test_size,
                "test_size_ratio": cfg.test_size,
                "n_train": train_n,
                "n_test": test_n,
                "cv_folds_within_training": cfg.cv_folds,
                "cv_used_for_model_selection": True,
                "cv_used_for_hyperparameter_tuning": True,
            }
        ]
    )

    # 3) Evaluation and robustness outputs.
    cv_df = artifacts.cv_scores
    best_hyperparameters_df = artifacts.best_hyperparameters
    metrics_df = evaluate_all_models(y_test, test_predictions, cfg)
    early_warning_df = evaluate_early_warning(y_test, test_predictions, cfg)
    risk_category_detail_df, risk_category_summary_df = summarize_risk_categories(
        y_test, test_predictions, cfg
    )
    fi_stability_df = feature_importance_stability(artifacts.best_estimators, X, y, cfg)
    best_by_cv_rmse_df = (
        cv_df.sort_values(["target", "cv_rmse_mean"], ascending=[True, True])
        .groupby("target", as_index=False)
        .first()
    )
    selected_test_eval_df = (
        metrics_df.merge(
            best_by_cv_rmse_df[["target", "model"]],
            on=["target", "model"],
            how="inner",
        )
        .sort_values("target")
    )
    selected_bootstrap_ci_df = selected_test_eval_df[
        [
            "target",
            "model",
            "rmse_ci_low",
            "rmse_ci_high",
            "mae_ci_low",
            "mae_ci_high",
            "mape_ci_low",
            "mape_ci_high",
            "r2_ci_low",
            "r2_ci_high",
        ]
    ].copy()
    selected_bootstrap_ci_df["bootstrap_resamples_b"] = cfg.bootstrap_iterations
    selected_bootstrap_ci_df["ci_method"] = "empirical_percentile_2.5_97.5"
    selected_risk_category_detail_df = (
        risk_category_detail_df.merge(
            best_by_cv_rmse_df[["target", "model"]],
            on=["target", "model"],
            how="inner",
        )
        .sort_values(["target", "predicted_risk_category"])
    )
    selected_risk_category_summary_df = (
        risk_category_summary_df.merge(
            best_by_cv_rmse_df[["target", "model"]],
            on=["target", "model"],
            how="inner",
        )
        .sort_values(["target", "predicted_risk_category"])
    )
    selected_threshold_sensitivity_df = (
        early_warning_df.merge(
            best_by_cv_rmse_df[["target", "model"]],
            on=["target", "model"],
            how="inner",
        )
        .sort_values(["target", "threshold"])
    )
    selected_prediction_rows = []
    for row in best_by_cv_rmse_df.itertuples(index=False):
        y_true = y_test[row.target].to_numpy()
        y_pred = test_predictions[row.target][row.model]
        selected_prediction_rows.append(
            pd.DataFrame(
                {
                    "target": row.target,
                    "model": row.model,
                    "actual": y_true,
                    "predicted": y_pred,
                }
            )
        )
    selected_predictions_df = pd.concat(selected_prediction_rows, ignore_index=True)
    tree_importance_df = compute_tree_feature_importance_rankings(artifacts.best_estimators)
    shap_importance_df = compute_shap_outputs(
        artifacts.best_estimators,
        X_test,
        cfg.output_figures_dir,
        cfg.shap_dependence_top_n,
    )
    driver_overlap_source_df = shap_importance_df if not shap_importance_df.empty else tree_importance_df
    driver_overlap_df = summarize_driver_overlap(driver_overlap_source_df, top_n=5)

    cv_df.to_csv(cfg.output_tables_dir / "cv_results.csv", index=False)
    best_hyperparameters_df.to_csv(cfg.output_tables_dir / "best_hyperparameters.csv", index=False)
    metrics_df.to_csv(cfg.output_tables_dir / "test_metrics_with_bootstrap_ci.csv", index=False)
    early_warning_df.to_csv(cfg.output_tables_dir / "early_warning_metrics.csv", index=False)
    risk_category_summary_df.to_csv(cfg.output_tables_dir / "risk_category_summary.csv", index=False)
    fi_stability_df.to_csv(cfg.output_tables_dir / "feature_importance_stability.csv", index=False)
    validation_design_df.to_csv(cfg.output_tables_dir / "validation_design.csv", index=False)
    best_by_cv_rmse_df.to_csv(cfg.output_tables_dir / "best_model_selection_by_cv_rmse.csv", index=False)
    selected_test_eval_df.to_csv(cfg.output_tables_dir / "selected_model_test_evaluation.csv", index=False)
    selected_bootstrap_ci_df.to_csv(cfg.output_tables_dir / "selected_model_bootstrap_ci.csv", index=False)
    selected_risk_category_detail_df.to_csv(
        cfg.output_tables_dir / "selected_model_risk_category_detail.csv", index=False
    )
    selected_risk_category_summary_df.to_csv(
        cfg.output_tables_dir / "selected_model_risk_category_summary.csv", index=False
    )
    selected_threshold_sensitivity_df.to_csv(
        cfg.output_tables_dir / "selected_model_threshold_sensitivity.csv", index=False
    )
    tree_importance_df.to_csv(cfg.output_tables_dir / "tree_feature_importance_ranking.csv", index=False)
    if not shap_importance_df.empty:
        shap_importance_df.to_csv(cfg.output_tables_dir / "tree_shap_importance_ranking.csv", index=False)
    if not driver_overlap_df.empty:
        driver_overlap_df.to_csv(cfg.output_tables_dir / "tree_driver_overlap_summary.csv", index=False)

    # 4) Linear diagnostics on test set.
    for target in targets:
        lin_pred = test_predictions[target]["linear_regression"]
        diag_df = linear_residual_diagnostics(y_test[target].to_numpy(), lin_pred)
        diag_df.to_csv(cfg.output_tables_dir / f"linear_residual_diagnostics_{target}.csv", index=False)

    # 5) Save final trained model objects for reproducibility/auditability.
    for target, model_dict in artifacts.best_estimators.items():
        for model_name, estimator in model_dict.items():
            model_path = cfg.output_models_dir / f"{target}__{model_name}.joblib"
            joblib.dump(estimator, model_path)

    # 6) Visual outputs.
    plot_metric_bars(metrics_df, cfg.output_figures_dir / "rmse_comparison.png")
    plot_early_warning(
        selected_threshold_sensitivity_df,
        cfg.output_figures_dir / "early_warning_sensitivity.png",
    )
    plot_strata_heatmap(raw_df, cfg.output_figures_dir / "strata_overlap_heatmap.png")
    plot_feature_distributions(raw_df, cfg.output_figures_dir / "key_distribution_checks.png")
    plot_selected_model_diagnostics(y_test, selected_predictions_df, cfg.output_figures_dir)
    plot_feature_importance_rankings(
        tree_importance_df,
        "importance",
        cfg.output_figures_dir / "tree_feature_importance_top10.png",
    )
    if not shap_importance_df.empty:
        plot_feature_importance_rankings(
            shap_importance_df,
            "mean_abs_shap",
            cfg.output_figures_dir / "tree_shap_importance_top10.png",
        )

    # 7) Optional sample-size robustness check.
    if cfg.sample_size_robustness_enabled:
        robustness_cfg = replace(
            cfg,
            n_projects=cfg.robustness_n_projects,
            rf_random_search_iter=cfg.robustness_rf_random_search_iter,
            xgb_random_search_iter=cfg.robustness_xgb_random_search_iter,
        )
        robustness_raw_df = generate_synthetic_projects(robustness_cfg)
        robustness_outlier_df = inject_predictor_outliers(robustness_raw_df, robustness_cfg)
        robustness_missing_df = inject_mcar_missingness(robustness_outlier_df, robustness_cfg)
        robustness_model_df = add_engineered_features(robustness_missing_df)
        robustness_X = robustness_model_df.drop(columns=targets)
        robustness_y = robustness_model_df[targets]
        assert_no_post_initiation_features(robustness_X)
        robustness_artifacts, _, _, _ = train_models(robustness_X, robustness_y, robustness_cfg)
        robustness_cv_df = robustness_artifacts.cv_scores
        robustness_cv_df.to_csv(
            cfg.output_tables_dir / "sample_size_robustness_cv_results.csv",
            index=False,
        )
        ranking_comparison_df = compare_model_rankings(
            cv_df,
            robustness_cv_df,
            base_label="n2500",
            alt_label=f"n{cfg.robustness_n_projects}",
        )
        ranking_comparison_df.to_csv(
            cfg.output_tables_dir / "sample_size_robustness_ranking_comparison.csv",
            index=False,
        )

    print("Workflow complete.")
    print(f"Outputs saved under: {cfg.output_root}")
    print("Best CV models by target (min CV RMSE):")
    print(cv_df.groupby("target").head(1).to_string(index=False))
    print("Sanity check summary:")
    print(sanity_df.to_string(index=False))


if __name__ == "__main__":
    main()
