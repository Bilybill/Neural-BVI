"""Finalize the joint-calibrated, layered-prior measured-data Neural-BVI result."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from finalize_hyperbola_conditioned_field_bvi import (
    build_shallow_audit,
    build_std_audit,
    write_shallow_audit,
    write_std_audit,
)
from finalize_shallow_reconciled_field_bvi import (
    MODEL_VMAX,
    MODEL_VMIN,
    configure_matplotlib,
    metric_payload,
    panel_labels,
    render_main_figure,
    save_all_formats,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_laplace_nn" / "noise_0p080"
RESULT_DIR = ROOT / "field_joint_early_layered_bvi"
PAPER_DIR = HERE.parents[1] / "paper_grsl"
FIGURE_DIR = PAPER_DIR / "figures"
ARRAYS_PATH = RESULT_DIR / "field_joint_early_layered_bvi_arrays.npz"
SUMMARY_PATH = RESULT_DIR / "summary.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mapped_arrays(source: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "observation": source["observation"],
        "original_mean": source["current_posterior_mean"],
        "map_center": source["layered_prior"],
        "posterior_samples": source["posterior_samples"],
        "posterior_mean": source["posterior_mean"],
        "posterior_std": source["posterior_std"],
        "prediction": source["prediction"],
        "previous_prediction": source["current_prediction"],
        "fwi_model": source["fwi_model"],
        "fwi_prediction": source["fwi_prediction"],
    }


def render_calibration_figure(
    source: np.lib.npyio.NpzFile,
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> None:
    raw = source["raw_observation"].squeeze()
    matched = source["raw_matched_prediction"].squeeze()
    corrected = source["raw_corrected_observation"].squeeze()
    coupling = source["antenna_coupling"].squeeze()
    surface = source["surface_reflection"].squeeze()
    residual = np.abs(matched - corrected)
    posterior = 2.0 + 8.0 * source["posterior_mean"].squeeze()
    layered_prior = 2.0 + 8.0 * source["layered_prior"].squeeze()
    current = 2.0 + 8.0 * source["current_posterior_mean"].squeeze()
    fwi = 2.0 + 8.0 * source["fwi_model"].squeeze()
    layer_values = source["layer_values_epsilon"]
    boundaries = source["layer_boundaries_m"]
    depth_limit = float(profile["model_domain"]["target_depth_m"])
    depth = np.linspace(0.0, depth_limit, posterior.shape[0])
    layer_index = np.searchsorted(boundaries[1:-1], depth, side="right")
    inverted_profile = layer_values[layer_index]
    dt_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    stop = min(raw.shape[0], int(round(15.0 / dt_ns)))
    bscan_extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        15.0,
        0.0,
    )
    model_extent = (0.0, float(profile["model_domain"]["width_m"]), 1.0, 0.0)
    amp_limit = max(
        float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        [raw[:stop].ravel(), matched[:stop].ravel(), corrected[:stop].ravel()]
                    )
                ),
                0.995,
            )
        ),
        1.0e-6,
    )
    nuisance_limit = max(
        float(np.quantile(np.abs(np.concatenate([coupling[:stop].ravel(), surface[:stop].ravel()])), 0.995)),
        1.0e-6,
    )
    residual_limit = max(float(np.quantile(residual[:stop], 0.995)), 1.0e-6)

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.9), constrained_layout=True)
    for axis, image, title in zip(
        axes[0, :2],
        (raw[:stop], matched[:stop]),
        ("Raw early B-scan", "Source/time-zero matched"),
    ):
        axis.imshow(image, cmap="gray", vmin=-amp_limit, vmax=amp_limit, extent=bscan_extent, aspect="auto")
        axis.set_title(title)
    for axis, image, title in zip(
        axes[0, 2:],
        (coupling[:stop], surface[:stop]),
        ("Antenna coupling", "Surface reflection"),
    ):
        axis.imshow(image, cmap="coolwarm", vmin=-nuisance_limit, vmax=nuisance_limit, extent=bscan_extent, aspect="auto")
        axis.set_title(title)
    axes[1, 0].imshow(
        corrected[:stop],
        cmap="gray",
        vmin=-amp_limit,
        vmax=amp_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    axes[1, 0].set_title("Nuisance-corrected data")
    axes[1, 1].imshow(
        residual[:stop],
        cmap="inferno",
        vmin=0.0,
        vmax=residual_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    axes[1, 1].set_title("Corrected residual")
    shallow = depth <= 1.0
    axes[1, 2].plot(current.mean(axis=1)[shallow], depth[shallow], color="0.55", lw=1.0, label="Previous")
    axes[1, 2].plot(inverted_profile[shallow], depth[shallow], color="#009E73", lw=1.0, label="1D inversion")
    axes[1, 2].plot(layered_prior.mean(axis=1)[shallow], depth[shallow], color="#0072B2", lw=1.1, label="Layered prior")
    axes[1, 2].plot(posterior.mean(axis=1)[shallow], depth[shallow], color="#D55E00", lw=1.1, label="Posterior")
    axes[1, 2].plot(fwi.mean(axis=1)[shallow], depth[shallow], color="0.15", lw=0.8, ls="--", label="FWI")
    axes[1, 2].invert_yaxis()
    axes[1, 2].set_xlim(MODEL_VMIN, MODEL_VMAX)
    axes[1, 2].set_ylim(1.0, 0.0)
    axes[1, 2].set_title("Shallow profiles")
    axes[1, 2].set_xlabel(r"$\epsilon_r$")
    axes[1, 2].set_ylabel("Depth (m)")
    axes[1, 2].legend(loc="lower right", fontsize=4.8, handlelength=1.3)
    axes[1, 3].imshow(
        posterior[shallow],
        cmap="viridis",
        vmin=MODEL_VMIN,
        vmax=MODEL_VMAX,
        extent=model_extent,
        aspect="auto",
    )
    axes[1, 3].set_title("Neural-BVI: 0-1 m")
    axes[1, 3].set_xlabel("Distance (m)")
    axes[1, 3].set_ylabel("Depth (m)")
    for axis in (*axes[0, :], axes[1, 0], axes[1, 1]):
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Time (ns)")
    panel_labels(axes)
    save_all_formats(fig, FIGURE_DIR / "fig_field_joint_early_calibration_diagnostic")
    plt.close(fig)


def write_metrics_csv(
    posterior: dict[str, float],
    fwi: dict[str, float],
    previous: dict[str, float],
    prior: dict[str, float],
) -> None:
    path = FIGURE_DIR / "fig_field_joint_early_layered_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "neural_bvi", "conventional_fwi", "previous_neural_bvi", "layered_prior"])
        for metric in (
            "rmse",
            "pearson",
            "trace_ncc",
            "envelope_rmse",
            "rmse_0_12ns",
            "rmse_12_20ns",
            "rmse_0_20ns",
            "rmse_20_30ns",
            "envelope_overshoot_0_12ns",
            "envelope_overshoot_0_20ns",
        ):
            writer.writerow([metric, posterior[metric], fwi[metric], previous[metric], prior[metric]])


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    field_summary = load_json(FIELD_DIR / "summary.json")
    result_summary = load_json(SUMMARY_PATH)
    profile = field_summary["forward_backend"]["acquisition_profile"]
    dt_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    with np.load(ARRAYS_PATH) as source:
        arrays = mapped_arrays(source)
        render_calibration_figure(source, profile, field_summary["crop"])

    observation = arrays["observation"].squeeze().astype(np.float64)
    posterior_metrics = metric_payload(arrays["prediction"].squeeze().astype(np.float64), observation, dt_ns)
    fwi_metrics = metric_payload(arrays["fwi_prediction"].squeeze().astype(np.float64), observation, dt_ns)
    previous_metrics = metric_payload(arrays["previous_prediction"].squeeze().astype(np.float64), observation, dt_ns)
    with np.load(ARRAYS_PATH) as source:
        prior_metrics = metric_payload(source["layered_prior_prediction"].squeeze().astype(np.float64), observation, dt_ns)
    render_main_figure(arrays, profile, field_summary["crop"])
    write_metrics_csv(posterior_metrics, fwi_metrics, previous_metrics, prior_metrics)

    std_audit = build_std_audit(
        arrays,
        profile,
        result_summary["bvi_result"],
        result_summary["posterior_predictive"],
        expected_sample_count=int(result_summary["config"]["posterior_samples"]),
    )
    write_std_audit(std_audit)
    shallow_audit = build_shallow_audit(
        arrays,
        profile,
        posterior_metrics,
        fwi_metrics,
        previous_metrics,
        std_audit,
    )
    write_shallow_audit(shallow_audit)

    starts = result_summary["starts"]
    layer_matrix = np.asarray([record["layer_values"] for record in starts], dtype=np.float64)
    time_zero = np.asarray([record["time_zero_ns"] for record in starts], dtype=np.float64)
    posterior_epsilon = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    previous_epsilon = 2.0 + 8.0 * arrays["original_mean"].squeeze()
    depth = np.linspace(0.0, float(profile["model_domain"]["target_depth_m"]), posterior_epsilon.shape[0])
    deep = depth >= 0.65
    provenance = {
        "status": "complete",
        "method": "joint time-zero/source/coupling/surface calibration, 1D layered shallow inversion, and local full-Laplace Neural-BVI",
        "nuisance": result_summary["nuisance"],
        "layer_boundaries_m": result_summary["config"]["layer_boundaries_m"],
        "selected_layer_values_epsilon": result_summary["selected_layer_values_epsilon"],
        "layer_start_stability": {
            "layer_value_mean": layer_matrix.mean(axis=0).tolist(),
            "layer_value_std": layer_matrix.std(axis=0).tolist(),
            "time_zero_mean_ns": float(time_zero.mean()),
            "time_zero_std_ns": float(time_zero.std()),
        },
        "selected_layered_prior": result_summary["selected_layered_prior"],
        "posterior_metrics": posterior_metrics,
        "conventional_fwi_metrics": fwi_metrics,
        "previous_neural_bvi_metrics": previous_metrics,
        "layered_prior_metrics": prior_metrics,
        "deep_rms_change_epsilon_z_ge_0p65m": float(
            np.sqrt(np.mean(np.square(posterior_epsilon[deep] - previous_epsilon[deep])))
        ),
        "std_audit": std_audit,
        "shallow_audit": shallow_audit,
        "claim_boundary": result_summary["claim_boundary"],
    }
    (FIGURE_DIR / "fig_field_joint_early_layered_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "posterior_metrics": posterior_metrics,
                "shallow_audit": shallow_audit["status"],
                "std_audit": std_audit["status"],
                "time_zero_mean_ns": provenance["layer_start_stability"]["time_zero_mean_ns"],
                "layer_value_std_max": max(provenance["layer_start_stability"]["layer_value_std"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
