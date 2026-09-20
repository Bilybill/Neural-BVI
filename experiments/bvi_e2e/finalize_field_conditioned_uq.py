"""Publish and audit the field-conditioned uncertainty map."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from finalize_joint_early_layered_field_bvi import mapped_arrays
from finalize_shallow_reconciled_field_bvi import configure_matplotlib, render_main_figure


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_joint_early_layered_bvi"
UQ_DIR = ROOT / "field_conditioned_uq"
FIELD_ARRAYS = FIELD_DIR / "field_joint_early_layered_bvi_arrays.npz"
FIELD_SUMMARY = FIELD_DIR / "summary.json"
UQ_ARRAYS = UQ_DIR / "field_conditioned_uq_arrays.npz"
UQ_SUMMARY = UQ_DIR / "summary.json"
PAPER_DIR = HERE.parents[1] / "paper_grsl"
FIGURE_DIR = PAPER_DIR / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values: Any) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def map_stats(array: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(array.mean()),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
        "q995": float(np.quantile(array, 0.995)),
        "max": float(array.max()),
    }


def write_audit(audit: dict[str, Any]) -> None:
    json_path = PAPER_DIR / "FIELD_NEURAL_BVI_STD_AUDIT.json"
    md_path = PAPER_DIR / "FIELD_NEURAL_BVI_STD_AUDIT.md"
    baseline_json = PAPER_DIR / "FIELD_NEURAL_BVI_STD_AUDIT_PCA_BASELINE.json"
    baseline_md = PAPER_DIR / "FIELD_NEURAL_BVI_STD_AUDIT_PCA_BASELINE.md"
    if json_path.exists():
        previous = load_json(json_path)
        if "16-dimensional local full-Laplace" in str(previous.get("definition", "")):
            shutil.copy2(json_path, baseline_json)
            if md_path.exists():
                shutil.copy2(md_path, baseline_md)
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    checks = audit["checks"]
    structural = audit["structural_std_epsilon"]
    calibrated = audit["synthetic_validation_calibrated_std_epsilon"]
    lines = [
        "# Field Neural-BVI Standard-Deviation Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        audit["definition"],
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
        f"| Neural ensemble members | {audit['ensemble']['member_count']} |",
        f"| Synthetic validation cases | {audit['ensemble']['validation_case_count']} |",
        f"| Structural mean sigma_epsilon | {structural['mean']:.6f} |",
        f"| Structural q95 sigma_epsilon | {structural['q95']:.6f} |",
        f"| Calibrated mean sigma_epsilon | {calibrated['mean']:.6f} |",
        f"| Correlation with synthetic-PCA std | {audit['field_specificity']['synthetic_std_pearson']:.6f} |",
        f"| Correlation with inversion edge | {audit['structure_correspondence']['edge_pearson']:.6f} |",
        f"| Edge-high/edge-low std ratio | {audit['structure_correspondence']['edge_ratio']:.6f} |",
        f"| Leave-one-out minimum map correlation | {audit['ensemble']['leave_one_out_map_pearson_min']:.6f} |",
        f"| Analytic/sample map correlation | {audit['sampling']['analytic_sample_pearson']:.6f} |",
        f"| Posterior-predictive B-scan std mean | {audit['posterior_predictive']['bscan_std_mean']:.6f} |",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{name}`: `{value}`" for name, value in checks.items())
    lines.extend(["", audit["interpretation_boundary"]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_csv(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    keys = (
        "mean_epsilon",
        "q95_epsilon",
        "pearson_with_old_pca_std",
        "pearson_with_synthetic_std_mean",
        "pearson_with_inversion_edge",
        "edge_q90_to_low_q50_mean_ratio",
        "top_0_0p5m_mean",
        "target_0p9_1p8m_mean",
        "deep_1p8_2p5m_mean",
    )
    with (FIGURE_DIR / "fig_field_conditioned_uq_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", *keys])
        for method, values in metrics.items():
            writer.writerow([method, *(values[key] for key in keys)])


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    field_summary = load_json(FIELD_SUMMARY)
    uq_summary = load_json(UQ_SUMMARY)
    acquisition = load_json(ROOT / "field_laplace_nn" / "noise_0p080" / "summary.json")
    profile = acquisition["forward_backend"]["acquisition_profile"]
    crop = acquisition["crop"]
    with np.load(FIELD_ARRAYS) as source, np.load(UQ_ARRAYS) as uq:
        arrays = mapped_arrays(source)
        structural_std = uq["combined_structural_std_epsilon"].astype(np.float64)
        calibrated_std = uq["calibrated_total_std_epsilon"].astype(np.float64)
        sampled_std = uq["sampled_total_std_epsilon"].astype(np.float64)
        arrays["posterior_std"] = (structural_std / 8.0)[None, None].astype(np.float32)
        arrays["posterior_samples"] = uq["posterior_samples"].astype(np.float32)
        local_std = uq["localized_std_epsilon"].astype(np.float64)
        ensemble_std = uq["ensemble_std_epsilon"].astype(np.float64)
        old_std = uq["old_pca_std_epsilon"].astype(np.float64)

    display = render_main_figure(
        arrays,
        profile,
        crop,
        std_title="Structural std.",
    )
    for suffix in ("pdf", "svg", "png"):
        shutil.copy2(
            UQ_DIR / f"fig_field_conditioned_uq_candidates.{suffix}",
            FIGURE_DIR / f"fig_field_conditioned_uq_candidates.{suffix}",
        )
    write_metrics_csv(uq_summary)

    metrics = uq_summary["metrics"]
    old_metrics = metrics["current_synthetic_pca"]
    selected_metrics = metrics["combined_structural"]
    sampling = uq_summary["sample_consistency"]
    ensemble_calibration = uq_summary["ensemble_calibration"]
    ensemble_stability = uq_summary["ensemble_leave_one_out_stability"]
    predictive = uq_summary["posterior_predictive"]
    checks = {
        "component_maps_are_finite_and_nonnegative": (
            finite(local_std)
            and finite(ensemble_std)
            and finite(structural_std)
            and float(min(local_std.min(), ensemble_std.min(), structural_std.min())) >= 0.0
        ),
        "law_of_total_variance_is_exact": float(
            np.max(np.abs(np.square(structural_std) - np.square(local_std) - np.square(ensemble_std)))
        )
        <= 2.0e-6,
        "field_map_is_not_synthetic_basis_dominated": selected_metrics["pearson_with_synthetic_std_mean"] < 0.50,
        "synthetic_similarity_drops_by_at_least_half": selected_metrics["pearson_with_synthetic_std_mean"] < 0.5 * old_metrics["pearson_with_synthetic_std_mean"],
        "std_corresponds_to_inversion_edges": selected_metrics["pearson_with_inversion_edge"] >= 0.30,
        "edge_regions_have_higher_std": selected_metrics["edge_q90_to_low_q50_mean_ratio"] >= 1.40,
        "quality_filtered_ensemble_has_at_least_ten_members": int(ensemble_calibration["member_count"]) >= 10,
        "synthetic_validation_calibration_hits_target": abs(float(ensemble_calibration["scaled_95pct_coverage"]) - 0.95) <= 0.005,
        "ensemble_leave_one_out_is_stable": float(ensemble_stability["map_pearson_min"]) >= 0.90,
        "analytic_std_matches_512_sample_reconstruction": (
            float(sampling["analytic_vs_sampled_total_std_pearson"]) >= 0.98
            and float(sampling["analytic_vs_sampled_total_std_relative_l2"]) <= 0.05
        ),
        "posterior_variability_reaches_field_bscan": float(predictive["bscan_std_mean"]) > 0.0,
        "mean_inversion_is_unchanged": np.array_equal(
            arrays["posterior_mean"],
            np.load(FIELD_ARRAYS)["posterior_mean"],
        ),
    }
    audit = {
        "status": "pass_with_boundary" if all(checks.values()) else "fail",
        "definition": (
            "Field-conditioned structural standard deviation around the final Neural-BVI mean. It combines a "
            "64-direction localized Deepwave-Jacobian Laplace component and a field-likelihood-weighted, "
            "quality-filtered 11-network epistemic component by the law of total variance."
        ),
        "checks": checks,
        "structural_std_epsilon": map_stats(structural_std),
        "synthetic_validation_calibrated_std_epsilon": map_stats(calibrated_std),
        "components": {
            "localized_deepwave_jacobian": map_stats(local_std),
            "neural_model_ensemble": map_stats(ensemble_std),
            "previous_synthetic_pca": map_stats(old_std),
        },
        "field_specificity": {
            "synthetic_std_pearson": selected_metrics["pearson_with_synthetic_std_mean"],
            "previous_synthetic_std_pearson": old_metrics["pearson_with_synthetic_std_mean"],
            "previous_map_pearson": selected_metrics["pearson_with_old_pca_std"],
        },
        "structure_correspondence": {
            "edge_pearson": selected_metrics["pearson_with_inversion_edge"],
            "previous_edge_pearson": old_metrics["pearson_with_inversion_edge"],
            "edge_ratio": selected_metrics["edge_q90_to_low_q50_mean_ratio"],
            "previous_edge_ratio": old_metrics["edge_q90_to_low_q50_mean_ratio"],
        },
        "ensemble": {
            "member_count": ensemble_calibration["member_count"],
            "validation_case_count": ensemble_calibration["validation_case_count"],
            "validation_nrmse_max": ensemble_calibration["validation_nrmse_max"],
            "synthetic_validation_scale_95": ensemble_calibration["scale_95"],
            "synthetic_validation_raw_coverage_95": ensemble_calibration["raw_95pct_coverage"],
            "synthetic_validation_scaled_coverage_95": ensemble_calibration["scaled_95pct_coverage"],
            "effective_field_member_count": uq_summary["ensemble_field_weighting"]["effective_member_count"],
            "leave_one_out_map_pearson_min": ensemble_stability["map_pearson_min"],
            "leave_one_out_relative_l2_max": ensemble_stability["relative_l2_max"],
        },
        "sampling": {
            "generated_sample_count": sampling["generated_sample_count"],
            "saved_sample_count": sampling["saved_sample_count"],
            "analytic_sample_pearson": sampling["analytic_vs_sampled_total_std_pearson"],
            "analytic_sample_relative_l2": sampling["analytic_vs_sampled_total_std_relative_l2"],
            "sampled_std": map_stats(sampled_std),
        },
        "posterior_predictive": predictive,
        "display": display,
        "interpretation_boundary": (
            "The displayed map is a structural ranking signal, not a field-calibrated credible interval. The "
            "magnitude-calibrated map uses synthetic validation coverage and is retained only as a supplementary "
            "sensitivity. The same field record informed source matching, the shallow prior, and the final weighting."
        ),
    }
    write_audit(audit)
    provenance = {
        "status": "complete",
        "generator": "experiments/bvi_e2e/finalize_field_conditioned_uq.py",
        "experiment_generator": "experiments/bvi_e2e/run_field_conditioned_uq.py",
        "source_arrays": str(UQ_ARRAYS),
        "source_summary": str(UQ_SUMMARY),
        "final_field_mean_source": str(FIELD_ARRAYS),
        "method": uq_summary["method"],
        "selected_local_scale": uq_summary["selected_local_scale"],
        "metrics": metrics,
        "std_audit": audit,
        "field_mean_status": field_summary["status"],
        "claim_boundary": uq_summary["claim_boundary"],
    }
    (FIGURE_DIR / "fig_field_conditioned_uq_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "std_audit": audit["status"],
                "structural_std": audit["structural_std_epsilon"],
                "synthetic_std_pearson": audit["field_specificity"]["synthetic_std_pearson"],
                "edge_pearson": audit["structure_correspondence"]["edge_pearson"],
                "edge_ratio": audit["structure_correspondence"]["edge_ratio"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
