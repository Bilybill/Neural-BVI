"""Finalize and audit the hyperbola-conditioned measured-data Neural-BVI result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from make_field_hyperbola_morphology_figure import apparent_morphology, morphology_payload
from publication_training import load_surrogate_checkpoint, set_seed
from run_deepwave_field_fwi import adaptive_source_match, analytic_envelope, data_metrics
from run_deepwave_map_bvi_synthetic import run_map_bvi_case


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_laplace_nn" / "noise_0p080"
REFINE_DIR = ROOT / "field_neural_bvi_trust_refine" / "full_c64_reg500_shallow_os5"
FWI_DIR = ROOT / "field_fwi_adaptive_source" / "eps5p5_c16_reg500_full"
LEGACY_DIR = ROOT / "field_hyperbola_conditioned_bvi"
OUT_DIR = ROOT / "field_hyperbola_conditioned_bvi_shallow"
PAPER_FIGURES = HERE.parents[1] / "paper_grsl" / "figures"
HYPERBOLA_PROVENANCE = PAPER_FIGURES / "fig_field_hyperbola_morphology_provenance.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--refine-dir", type=Path, default=REFINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class AdaptiveMatchedForward(torch.nn.Module):
    def __init__(self, forward: torch.nn.Module, observation: torch.Tensor) -> None:
        super().__init__()
        self.forward_backend = forward
        self.register_buffer("observation", observation)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        prediction = self.forward_backend(model)
        matched, _ = adaptive_source_match(
            prediction,
            self.observation,
            waterlevel=0.03,
            smoothing_bins=2.0,
        )
        return matched


def add_fixed_hyperbola_atoms(
    model: torch.Tensor,
    hyperbola: dict[str, Any],
    width_m: float,
    depth_m: float,
) -> torch.Tensor:
    height, width = model.shape[-2:]
    x = torch.linspace(0.0, width_m, width, device=model.device)
    z = torch.linspace(0.0, depth_m, height, device=model.device)
    z_grid, x_grid = torch.meshgrid(z, x, indexing="ij")
    conditioned = model.clone()
    for event_id in ("left", "right"):
        event = hyperbola["events"][event_id]
        extraction = event["extraction"]
        shape = event["neural_bvi_apparent_morphology"]
        center_x = float(extraction["local_apex_x_m"])
        center_z = float(extraction["conditional_apex_depth_m"])
        sigma_x = float(shape["half_prominence_width_m"]) / 2.355
        sigma_z = float(shape["half_prominence_thickness_m"]) / 2.355
        amplitude = float(shape["peak_contrast_epsilon"]) / 8.0
        atom = torch.exp(
            -0.5 * ((x_grid - center_x) / sigma_x).square()
            -0.5 * ((z_grid - center_z) / sigma_z).square()
        )[None, None]
        conditioned = torch.clamp(conditioned + amplitude * atom, 0.0, 1.0)
    return conditioned


def time_metrics(
    prediction: np.ndarray,
    observation: np.ndarray,
    sample_interval_ns: float,
) -> dict[str, float]:
    residual = prediction - observation
    metrics = {
        "rmse_0_16ns": float(np.sqrt(np.mean(np.square(residual[:41])))),
        "rmse_16_70ns": float(np.sqrt(np.mean(np.square(residual[41:])))),
    }
    prediction_tensor = torch.from_numpy(prediction)[None, None].float()
    observation_tensor = torch.from_numpy(observation)[None, None].float()
    overshoot = torch.relu(
        analytic_envelope(prediction_tensor) - analytic_envelope(observation_tensor)
    ).squeeze().numpy()
    for label, start_ns, end_ns in (
        ("0_12ns", 0.0, 12.0),
        ("12_20ns", 12.0, 20.0),
        ("20_30ns", 20.0, 30.0),
        ("0_20ns", 0.0, 20.0),
        ("0_30ns", 0.0, 30.0),
    ):
        start = max(0, int(round(start_ns / sample_interval_ns)))
        stop = min(residual.shape[0], int(round(end_ns / sample_interval_ns)))
        metrics[f"rmse_{label}"] = float(np.sqrt(np.mean(np.square(residual[start:stop]))))
        metrics[f"envelope_overshoot_{label}"] = float(
            np.sqrt(np.mean(np.square(overshoot[start:stop])))
        )
    return metrics


def morphology_metrics(
    model_epsilon: np.ndarray,
    hyperbola: dict[str, Any],
    width_m: float,
    depth_m: float,
) -> dict[str, dict[str, Any]]:
    x = np.linspace(0.0, width_m, model_epsilon.shape[1])
    z = np.linspace(0.0, depth_m, model_epsilon.shape[0])
    records = {}
    for event_id in ("left", "right"):
        extraction = hyperbola["events"][event_id]["extraction"]
        event = {
            "local_apex_x_m": extraction["local_apex_x_m"],
            "fitted_depth_m": extraction["conditional_apex_depth_m"],
        }
        shape = apparent_morphology(
            model_epsilon,
            x,
            z,
            (float(event["local_apex_x_m"]), float(event["fitted_depth_m"])),
        )
        payload = morphology_payload(shape, event, 2.0)
        payload["half_prominence_area_m2"] = float(
            payload["half_prominence_width_m"] * payload["half_prominence_thickness_m"]
        )
        records[event_id] = payload
    return records


def comparison_gate(
    optimized: dict[str, float],
    fwi: dict[str, float],
    previous: dict[str, float],
    optimized_shape: dict[str, dict[str, Any]],
    fwi_shape: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "rmse_lower": optimized["rmse"] < fwi["rmse"],
        "pearson_higher": optimized["pearson"] > fwi["pearson"],
        "trace_ncc_higher": optimized["trace_ncc"] > fwi["trace_ncc"],
        "envelope_rmse_lower": optimized["envelope_rmse"] < fwi["envelope_rmse"],
        "early_rmse_lower": optimized["rmse_0_16ns"] < fwi["rmse_0_16ns"],
        "late_rmse_lower": optimized["rmse_16_70ns"] < fwi["rmse_16_70ns"],
        "rmse_0_12ns_lower_than_fwi": optimized["rmse_0_12ns"] < fwi["rmse_0_12ns"],
        "rmse_12_20ns_lower_than_fwi": optimized["rmse_12_20ns"] < fwi["rmse_12_20ns"],
        "rmse_20_30ns_lower_than_fwi": optimized["rmse_20_30ns"] < fwi["rmse_20_30ns"],
        "overshoot_0_12ns_lower_than_previous": (
            optimized["envelope_overshoot_0_12ns"]
            < previous["envelope_overshoot_0_12ns"]
        ),
        "overshoot_0_20ns_lower_than_previous": (
            optimized["envelope_overshoot_0_20ns"]
            < previous["envelope_overshoot_0_20ns"]
        ),
    }
    for event_id in ("left", "right"):
        checks[f"{event_id}_centroid_offset_lower"] = (
            optimized_shape[event_id]["centroid_distance_from_hyperbola_m"]
            < fwi_shape[event_id]["centroid_distance_from_hyperbola_m"]
        )
        checks[f"{event_id}_footprint_area_lower"] = (
            optimized_shape[event_id]["half_prominence_area_m2"]
            < fwi_shape[event_id]["half_prominence_area_m2"]
        )
        checks[f"{event_id}_peak_contrast_higher"] = (
            optimized_shape[event_id]["peak_contrast_epsilon"]
            > fwi_shape[event_id]["peak_contrast_epsilon"]
        )
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "passed_count": int(sum(checks.values())),
        "check_count": len(checks),
        "checks": checks,
    }


def save_figure(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> None:
    observation = arrays["observation"].squeeze()
    optimized = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    optimized_std = 8.0 * arrays["posterior_std"].squeeze()
    fwi = 2.0 + 8.0 * arrays["fwi_model"].squeeze()
    optimized_prediction = arrays["prediction"].squeeze()
    previous_prediction = arrays["previous_prediction"].squeeze()
    fwi_prediction = arrays["fwi_prediction"].squeeze()
    sample_interval_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    time_ns = np.arange(observation.shape[0]) * sample_interval_ns

    def diagnostic_series(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        residual_rmse = np.sqrt(np.mean(np.square(prediction - observation), axis=1))
        prediction_tensor = torch.from_numpy(prediction)[None, None].float()
        observation_tensor = torch.from_numpy(observation)[None, None].float()
        overshoot = torch.relu(
            analytic_envelope(prediction_tensor) - analytic_envelope(observation_tensor)
        ).squeeze().numpy()
        return residual_rmse, np.sqrt(np.mean(np.square(overshoot), axis=1))

    previous_rmse, previous_overshoot = diagnostic_series(previous_prediction)
    optimized_rmse, optimized_overshoot = diagnostic_series(optimized_prediction)
    fwi_rmse, fwi_overshoot = diagnostic_series(fwi_prediction)
    source_data_path = PAPER_FIGURES / "fig_field_optimized_neural_bvi_shallow_source_data.csv"
    with source_data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_ns",
                "previous_rmse",
                "optimized_rmse",
                "traditional_fwi_rmse",
                "previous_envelope_overshoot",
                "optimized_envelope_overshoot",
                "traditional_fwi_envelope_overshoot",
            ]
        )
        writer.writerows(
            zip(
                time_ns,
                previous_rmse,
                optimized_rmse,
                fwi_rmse,
                previous_overshoot,
                optimized_overshoot,
                fwi_overshoot,
            )
        )

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.9), constrained_layout=True)
    model_extent = (0.0, float(profile["model_domain"]["width_m"]), float(profile["model_domain"]["target_depth_m"]), 0.0)
    bscan_extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        float(crop["target_time_window_ns"]),
        0.0,
    )
    bscan_limit = max(
        float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        [
                            observation.ravel(),
                            previous_prediction.ravel(),
                            optimized_prediction.ravel(),
                        ]
                    )
                ),
                0.995,
            )
        ),
        1.0e-6,
    )
    obs_limit = max(float(np.quantile(np.abs(observation), 0.995)), 1.0e-6)
    axes[0, 0].imshow(observation, cmap="gray", vmin=-obs_limit, vmax=obs_limit, extent=bscan_extent, aspect="auto")
    axes[0, 0].set_title("Measured B-scan")
    optimized_image = axes[0, 1].imshow(
        optimized,
        cmap="viridis",
        vmin=2.0,
        vmax=8.0,
        extent=model_extent,
        aspect="auto",
    )
    axes[0, 1].set_title("Final Neural-BVI mean")
    std_limit = max(float(np.quantile(optimized_std, 0.995)), 1.0e-6)
    std_image = axes[0, 2].imshow(
        optimized_std,
        cmap="magma",
        vmin=0.0,
        vmax=std_limit,
        extent=model_extent,
        aspect="auto",
    )
    axes[0, 2].set_title("Posterior std.")
    axes[0, 3].imshow(fwi, cmap="viridis", vmin=2.0, vmax=8.0, extent=model_extent, aspect="auto")
    axes[0, 3].set_title("Conventional FWI")
    shallow_stop = min(observation.shape[0], int(round(30.0 / sample_interval_ns)))
    shallow_extent = (bscan_extent[0], bscan_extent[1], 30.0, 0.0)
    for axis, image, title in zip(
        axes[1, :3],
        (observation, previous_prediction, optimized_prediction),
        ("Measured: 0-30 ns", "Previous reforward", "Final reforward"),
    ):
        axis.imshow(
            image[:shallow_stop],
            cmap="gray",
            vmin=-bscan_limit,
            vmax=bscan_limit,
            extent=shallow_extent,
            aspect="auto",
        )
        axis.axhline(20.0, color="#0072B2", lw=0.7, ls="--")
        axis.set_title(title)
    axes[1, 3].axvspan(0.0, 20.0, color="#EEEEEE", zorder=0)
    axes[1, 3].plot(time_ns, previous_overshoot, color="#777777", lw=1.0, label="Previous")
    axes[1, 3].plot(time_ns, optimized_overshoot, color="#0072B2", lw=1.2, label="Final Neural-BVI")
    axes[1, 3].plot(time_ns, fwi_overshoot, color="#D55E00", lw=0.9, label="FWI")
    axes[1, 3].set_xlim(0.0, 30.0)
    axes[1, 3].set_ylim(bottom=0.0)
    axes[1, 3].set_title("Positive envelope excess")
    axes[1, 3].set_ylabel("RMS amplitude")
    axes[1, 3].legend(loc="upper right", fontsize=5.2, handlelength=1.5)
    for axis in axes[0, 1:]:
        axis.set_ylabel("Depth (m)")
    for axis in (axes[0, 0], axes[1, 0], axes[1, 1], axes[1, 2]):
        axis.set_ylabel("Time (ns)")
    for axis in axes[1, :3]:
        axis.set_xlabel("Distance (m)")
    axes[1, 3].set_xlabel("Time (ns)")
    fig.colorbar(
        optimized_image,
        ax=[axes[0, 1], axes[0, 3]],
        orientation="vertical",
        fraction=0.026,
        pad=0.02,
        label=r"$\epsilon_r$",
    )
    fig.colorbar(
        std_image,
        ax=axes[0, 2],
        orientation="vertical",
        fraction=0.048,
        pad=0.02,
        label=r"$\sigma_{\epsilon_r}$",
    )
    for panel, axis in zip("abcdefgh", axes.ravel()):
        axis.text(-0.14, 1.03, panel, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
    stem = PAPER_FIGURES / "fig_field_optimized_neural_bvi_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_final_figure(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> dict[str, float]:
    observation = arrays["observation"].squeeze()
    neural_bvi = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    neural_bvi_std = 8.0 * arrays["posterior_std"].squeeze()
    fwi = 2.0 + 8.0 * arrays["fwi_model"].squeeze()
    neural_bvi_prediction = arrays["prediction"].squeeze()
    fwi_prediction = arrays["fwi_prediction"].squeeze()
    neural_bvi_residual = np.abs(neural_bvi_prediction - observation)
    fwi_residual = np.abs(fwi_prediction - observation)

    combined_models = np.concatenate([neural_bvi.ravel(), fwi.ravel()])
    model_vmin = math.floor(float(np.quantile(combined_models, 0.005)) * 10.0) / 10.0
    model_vmax = math.ceil(float(np.quantile(combined_models, 0.999)) * 10.0) / 10.0
    model_vmin = max(2.0, model_vmin)
    model_vmax = max(model_vmin + 0.5, min(8.0, model_vmax))
    bscan_limit = max(
        float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        [observation.ravel(), neural_bvi_prediction.ravel(), fwi_prediction.ravel()]
                    )
                ),
                0.995,
            )
        ),
        1.0e-6,
    )
    residual_limit = max(
        float(np.quantile(np.concatenate([neural_bvi_residual.ravel(), fwi_residual.ravel()]), 0.995)),
        1.0e-6,
    )
    std_limit = max(float(np.quantile(neural_bvi_std, 0.995)), 1.0e-6)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.8), constrained_layout=True)
    model_extent = (
        0.0,
        float(profile["model_domain"]["width_m"]),
        float(profile["model_domain"]["target_depth_m"]),
        0.0,
    )
    bscan_extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        float(crop["target_time_window_ns"]),
        0.0,
    )
    axes[0, 0].imshow(
        observation,
        cmap="gray",
        vmin=-bscan_limit,
        vmax=bscan_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    axes[0, 0].set_title("Measured B-scan")
    neural_image = axes[0, 1].imshow(
        neural_bvi,
        cmap="viridis",
        vmin=model_vmin,
        vmax=model_vmax,
        extent=model_extent,
        aspect="auto",
    )
    axes[0, 1].set_title("Neural-BVI mean")
    std_image = axes[0, 2].imshow(
        neural_bvi_std,
        cmap="magma",
        vmin=0.0,
        vmax=std_limit,
        extent=model_extent,
        aspect="auto",
    )
    axes[0, 2].set_title("Local posterior std.")
    axes[0, 3].imshow(
        fwi,
        cmap="viridis",
        vmin=model_vmin,
        vmax=model_vmax,
        extent=model_extent,
        aspect="auto",
    )
    axes[0, 3].set_title("Conventional FWI")

    for axis, image, title in zip(
        axes[1, :2],
        (neural_bvi_prediction, fwi_prediction),
        ("Neural-BVI reforward", "FWI reforward"),
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=-bscan_limit,
            vmax=bscan_limit,
            extent=bscan_extent,
            aspect="auto",
        )
        axis.set_title(title)
    residual_image = axes[1, 2].imshow(
        neural_bvi_residual,
        cmap="inferno",
        vmin=0.0,
        vmax=residual_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    axes[1, 2].set_title("Neural-BVI residual")
    axes[1, 3].imshow(
        fwi_residual,
        cmap="inferno",
        vmin=0.0,
        vmax=residual_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    axes[1, 3].set_title("FWI residual")

    for axis in axes[0, 1:]:
        axis.set_ylabel("Depth (m)")
    for axis in (axes[0, 0], *axes[1, :]):
        axis.set_ylabel("Time (ns)")
    for axis in axes[1, :]:
        axis.set_xlabel("Distance (m)")
    fig.colorbar(
        neural_image,
        ax=[axes[0, 1], axes[0, 3]],
        orientation="vertical",
        fraction=0.026,
        pad=0.02,
        label=r"$\epsilon_r$",
    )
    fig.colorbar(
        std_image,
        ax=axes[0, 2],
        orientation="vertical",
        fraction=0.048,
        pad=0.02,
        label=r"$\sigma_{\epsilon_r}$",
    )
    fig.colorbar(
        residual_image,
        ax=[axes[1, 2], axes[1, 3]],
        orientation="vertical",
        fraction=0.026,
        pad=0.02,
        label="Absolute residual",
    )
    for panel, axis in zip("abcdefgh", axes.ravel()):
        axis.text(-0.14, 1.03, panel, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
    stem = PAPER_FIGURES / "fig_field_neural_bvi_final_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "model_vmin": model_vmin,
        "model_vmax": model_vmax,
        "bscan_abs_limit": bscan_limit,
        "residual_max": residual_limit,
        "std_max": std_limit,
    }


def posterior_predictive_audit(
    samples: torch.Tensor,
    matched_forward: torch.nn.Module,
    observation: torch.Tensor,
    sample_count: int = 16,
) -> dict[str, float]:
    indices = np.linspace(0, len(samples) - 1, min(sample_count, len(samples)), dtype=int)
    predictions = []
    rmses = []
    with torch.no_grad():
        for index in indices:
            prediction = matched_forward(samples[index : index + 1].to(observation.device))
            predictions.append(prediction.cpu())
            rmses.append(float(torch.sqrt(torch.mean((prediction - observation) ** 2))))
    prediction_stack = torch.cat(predictions, dim=0)
    prediction_std = prediction_stack.std(dim=0, unbiased=False)
    return {
        "sample_count": len(indices),
        "matched_rmse_mean": float(np.mean(rmses)),
        "matched_rmse_std": float(np.std(rmses)),
        "matched_rmse_min": float(np.min(rmses)),
        "matched_rmse_max": float(np.max(rmses)),
        "bscan_std_mean": float(prediction_std.mean()),
        "bscan_std_q95": float(torch.quantile(prediction_std, 0.95)),
        "bscan_std_max": float(prediction_std.max()),
    }


def build_std_audit(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    bvi_result: dict[str, Any] | None,
    predictive: dict[str, float] | None,
    expected_sample_count: int = 128,
) -> dict[str, Any]:
    samples = torch.from_numpy(arrays["posterior_samples"]).float()
    saved_std = torch.from_numpy(arrays["posterior_std"]).float()
    recomputed_std = samples.std(dim=0, keepdim=True, unbiased=False)
    epsilon_std = (8.0 * recomputed_std).squeeze().numpy()
    centered = (samples - samples.mean(dim=0, keepdim=True)).flatten(1)
    singular_values = torch.linalg.svdvals(centered)
    variance_fraction = singular_values.square() / singular_values.square().sum().clamp_min(1.0e-12)
    cumulative = torch.cumsum(variance_fraction, dim=0).cpu().numpy()
    effective_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)
    z = np.linspace(0.0, float(profile["model_domain"]["target_depth_m"]), epsilon_std.shape[0])
    regions = {
        "top_0_0p9m": z < 0.9,
        "target_band_0p9_1p8m": (z >= 0.9) & (z < 1.8),
        "deep_1p8_2p5m": z >= 1.8,
    }
    regional = {
        name: {
            "mean": float(epsilon_std[mask].mean()),
            "q95": float(np.quantile(epsilon_std[mask], 0.95)),
            "max": float(epsilon_std[mask].max()),
        }
        for name, mask in regions.items()
    }
    precision_condition = (
        float(bvi_result["posterior_precision_condition"])
        if bvi_result and bvi_result.get("posterior_precision_condition") is not None
        else None
    )

    def spatial_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
        first_flat = first.double().flatten()
        second_flat = second.double().flatten()
        first_flat = first_flat - first_flat.mean()
        second_flat = second_flat - second_flat.mean()
        denominator = torch.sqrt(first_flat.square().sum() * second_flat.square().sum()).clamp_min(1.0e-12)
        return float((first_flat * second_flat).sum() / denominator)

    def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
        denominator = reference.double().square().sum().clamp_min(1.0e-12)
        return float(torch.sqrt((candidate.double() - reference.double()).square().sum() / denominator))

    half_count = int(samples.shape[0]) // 2
    first_half_std = samples[:half_count].std(dim=0, keepdim=True, unbiased=False)
    second_half_std = samples[half_count : 2 * half_count].std(dim=0, keepdim=True, unbiased=False)
    generator = torch.Generator().manual_seed(20_260_715)
    half_sample_relative_l2: list[float] = []
    half_sample_correlations: list[float] = []
    half_sample_mean_epsilon_std: list[float] = []
    for _ in range(16):
        indices = torch.randperm(int(samples.shape[0]), generator=generator)[:half_count]
        half_std = samples[indices].std(dim=0, keepdim=True, unbiased=False)
        half_sample_relative_l2.append(relative_l2(half_std, recomputed_std))
        half_sample_correlations.append(spatial_correlation(half_std, recomputed_std))
        half_sample_mean_epsilon_std.append(float(8.0 * half_std.mean()))
    monte_carlo_stability = {
        "split_half_sample_count": half_count,
        "split_half_mean_epsilon_std_first": float(8.0 * first_half_std.mean()),
        "split_half_mean_epsilon_std_second": float(8.0 * second_half_std.mean()),
        "split_half_map_pearson": spatial_correlation(first_half_std, second_half_std),
        "split_half_relative_l2_difference": relative_l2(first_half_std, second_half_std),
        "half_sample_resampling_count": len(half_sample_relative_l2),
        "half_sample_mean_epsilon_std_sd": float(np.std(half_sample_mean_epsilon_std, ddof=1)),
        "half_sample_relative_l2_to_full_median": float(np.median(half_sample_relative_l2)),
        "half_sample_relative_l2_to_full_q95": float(np.quantile(half_sample_relative_l2, 0.95)),
        "half_sample_map_pearson_to_full_min": float(np.min(half_sample_correlations)),
        "half_sample_map_pearson_to_full_median": float(np.median(half_sample_correlations)),
        "gaussian_std_relative_mcse_approx": float(1.0 / math.sqrt(2.0 * (int(samples.shape[0]) - 1))),
    }
    checks = {
        "sample_count_matches_expected": int(samples.shape[0]) == int(expected_sample_count),
        "saved_std_exactly_reproducible": float((saved_std - recomputed_std).abs().max()) <= 1.0e-8,
        "finite_samples_and_std": bool(torch.isfinite(samples).all() and torch.isfinite(recomputed_std).all()),
        "clipping_below_one_percent": float(((samples <= 0.0) | (samples >= 1.0)).float().mean()) < 0.01,
        "latent_precision_well_conditioned": precision_condition is not None and precision_condition < 1.0e6,
        "effective_rank_within_latent_dimension": 1 <= effective_rank_99 <= 16,
        "posterior_variability_reaches_forward_data": predictive is not None and predictive["bscan_std_mean"] > 0.0,
        "split_half_std_map_correlated": monte_carlo_stability["split_half_map_pearson"] >= 0.90,
        "half_sample_std_map_stable": (
            monte_carlo_stability["half_sample_map_pearson_to_full_min"] >= 0.98
            and monte_carlo_stability["half_sample_relative_l2_to_full_q95"] <= 0.08
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "definition": (
            f"Pixelwise population standard deviation across {int(samples.shape[0])} independent samples from the "
            "16-dimensional local full-Laplace residual posterior, converted by sigma_epsilon=8*sigma_normalized."
        ),
        "checks": checks,
        "sample_count": int(samples.shape[0]),
        "epsilon_std": {
            "mean": float(epsilon_std.mean()),
            "min": float(epsilon_std.min()),
            "q50": float(np.quantile(epsilon_std, 0.50)),
            "q95": float(np.quantile(epsilon_std, 0.95)),
            "q995": float(np.quantile(epsilon_std, 0.995)),
            "max": float(epsilon_std.max()),
        },
        "sample_bounds": {
            "lower_clip_fraction": float((samples <= 0.0).float().mean()),
            "upper_clip_fraction": float((samples >= 1.0).float().mean()),
        },
        "latent_structure": {
            "nominal_dimension": 16,
            "effective_rank_99pct": effective_rank_99,
            "variance_beyond_rank16": float(variance_fraction[16:].sum()),
            "posterior_latent_std_mean": bvi_result.get("posterior_latent_std_mean") if bvi_result else None,
            "posterior_latent_std_min": bvi_result.get("posterior_latent_std_min") if bvi_result else None,
            "posterior_latent_std_max": bvi_result.get("posterior_latent_std_max") if bvi_result else None,
            "posterior_precision_condition": precision_condition,
            "posterior_precision_data_trace": bvi_result.get("posterior_precision_data_trace") if bvi_result else None,
        },
        "regional_epsilon_std": regional,
        "monte_carlo_stability": monte_carlo_stability,
        "posterior_predictive": predictive,
        "interpretation_boundary": (
            "The field record has no permittivity truth, so calibration and std-error ranking cannot be measured. "
            "This map is local residual-subspace dispersion under the scalar Deepwave likelihood and error-PCA basis; "
            "it excludes neural-weight, model-form, acquisition, and full electromagnetic uncertainty."
        ),
    }


def write_std_audit(audit: dict[str, Any]) -> None:
    json_path = PAPER_FIGURES.parent / "FIELD_NEURAL_BVI_STD_AUDIT.json"
    md_path = PAPER_FIGURES.parent / "FIELD_NEURAL_BVI_STD_AUDIT.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    epsilon = audit["epsilon_std"]
    latent = audit["latent_structure"]
    stability = audit["monte_carlo_stability"]
    predictive = audit["posterior_predictive"] or {}
    lines = [
        "# Field Neural-BVI Standard-Deviation Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        audit["definition"],
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
        f"| Samples | {audit['sample_count']} |",
        f"| Mean sigma_epsilon | {epsilon['mean']:.6f} |",
        f"| 95th percentile sigma_epsilon | {epsilon['q95']:.6f} |",
        f"| 99.5th percentile sigma_epsilon | {epsilon['q995']:.6f} |",
        f"| Effective rank at 99% variance | {latent['effective_rank_99pct']} |",
        f"| Precision condition number | {latent['posterior_precision_condition']:.6f} |",
        f"| Split-half std-map Pearson | {stability['split_half_map_pearson']:.6f} |",
        f"| Half-sample relative L2 to full, 95th percentile | {stability['half_sample_relative_l2_to_full_q95']:.6f} |",
        f"| Approximate std Monte Carlo relative error | {stability['gaussian_std_relative_mcse_approx']:.6f} |",
        f"| Posterior-predictive B-scan std mean | {predictive.get('bscan_std_mean', float('nan')):.6f} |",
        "",
        audit["interpretation_boundary"],
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_shallow_audit(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    neural_metrics: dict[str, float],
    fwi_metrics: dict[str, float],
    previous_metrics: dict[str, float],
    std_audit: dict[str, Any],
) -> dict[str, Any]:
    neural = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    map_center = 2.0 + 8.0 * arrays["map_center"].squeeze()
    original = 2.0 + 8.0 * arrays["original_mean"].squeeze()
    fwi = 2.0 + 8.0 * arrays["fwi_model"].squeeze()
    depth = np.linspace(0.0, float(profile["model_domain"]["target_depth_m"]), neural.shape[0])

    def band_stats(model: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        values = model[mask]
        row_median = np.median(values, axis=1, keepdims=True)
        lateral_deviation = np.abs(values - row_median)
        return {
            "mean_epsilon": float(values.mean()),
            "min_epsilon": float(values.min()),
            "max_epsilon": float(values.max()),
            "mean_lateral_std_epsilon": float(np.std(values, axis=1).mean()),
            "row_centered_abs_deviation_q95": float(np.quantile(lateral_deviation, 0.95)),
            "row_centered_abs_deviation_max": float(lateral_deviation.max()),
        }

    top_mask = depth < 0.5
    transition_mask = (depth >= 0.5) & (depth < 0.9)
    shallow_mask = depth < 0.9
    neural_top = band_stats(neural, top_mask)
    fwi_top = band_stats(fwi, top_mask)
    neural_transition = band_stats(neural, transition_mask)
    shallow_update_rms = float(np.sqrt(np.mean(np.square(neural[shallow_mask] - map_center[shallow_mask]))))
    shallow_original_rms = float(np.sqrt(np.mean(np.square(neural[shallow_mask] - original[shallow_mask]))))

    windows = {}
    for label in ("0_12ns", "12_20ns", "0_20ns", "20_30ns"):
        windows[label] = {
            "neural_bvi_rmse": float(neural_metrics[f"rmse_{label}"]),
            "fwi_rmse": float(fwi_metrics[f"rmse_{label}"]),
            "previous_neural_bvi_rmse": float(previous_metrics[f"rmse_{label}"]),
            "neural_bvi_positive_envelope_excess": float(neural_metrics[f"envelope_overshoot_{label}"]),
            "fwi_positive_envelope_excess": float(fwi_metrics[f"envelope_overshoot_{label}"]),
            "previous_neural_bvi_positive_envelope_excess": float(
                previous_metrics[f"envelope_overshoot_{label}"]
            ),
        }

    checks = {
        "top_0_0p5m_has_no_bound_clipping": neural_top["min_epsilon"] > 2.0 and neural_top["max_epsilon"] < 10.0,
        "top_0_0p5m_lateral_q95_below_0p3": neural_top["row_centered_abs_deviation_q95"] < 0.3,
        "top_mean_within_0p75_of_fwi": abs(neural_top["mean_epsilon"] - fwi_top["mean_epsilon"]) < 0.75,
        "local_posterior_update_is_small_shallower_than_0p9m": shallow_update_rms < 0.05,
        "rmse_0_12ns_below_fwi": windows["0_12ns"]["neural_bvi_rmse"] < windows["0_12ns"]["fwi_rmse"],
        "rmse_12_20ns_below_fwi": windows["12_20ns"]["neural_bvi_rmse"] < windows["12_20ns"]["fwi_rmse"],
        "rmse_0_20ns_below_fwi": windows["0_20ns"]["neural_bvi_rmse"] < windows["0_20ns"]["fwi_rmse"],
        "rmse_20_30ns_below_fwi": windows["20_30ns"]["neural_bvi_rmse"] < windows["20_30ns"]["fwi_rmse"],
        "positive_excess_0_12ns_reduced_from_previous": (
            windows["0_12ns"]["neural_bvi_positive_envelope_excess"]
            < windows["0_12ns"]["previous_neural_bvi_positive_envelope_excess"]
        ),
        "positive_excess_0_20ns_reduced_from_previous": (
            windows["0_20ns"]["neural_bvi_positive_envelope_excess"]
            < windows["0_20ns"]["previous_neural_bvi_positive_envelope_excess"]
        ),
    }
    boundary_flags = {
        "transition_0p5_0p9m_has_strong_lateral_structure": neural_transition["row_centered_abs_deviation_q95"] > 0.5,
        "positive_excess_0_12ns_above_fwi": (
            windows["0_12ns"]["neural_bvi_positive_envelope_excess"]
            > windows["0_12ns"]["fwi_positive_envelope_excess"]
        ),
        "positive_excess_0_20ns_above_fwi": (
            windows["0_20ns"]["neural_bvi_positive_envelope_excess"]
            > windows["0_20ns"]["fwi_positive_envelope_excess"]
        ),
    }
    top_mean_gap = abs(neural_top["mean_epsilon"] - fwi_top["mean_epsilon"])
    return {
        "status": "pass_with_boundary" if all(checks.values()) else "fail",
        "assessment": (
            "Shallow response is data-consistent under the declared scalar-wave and adaptive-source contract, "
            "but absolute shallow permittivity is not ground-truth validated."
        ),
        "checks": checks,
        "boundary_flags": boundary_flags,
        "model_bands": {
            "neural_bvi_top_0_0p5m": neural_top,
            "fwi_top_0_0p5m": fwi_top,
            "neural_bvi_transition_0p5_0p9m": neural_transition,
        },
        "model_comparisons": {
            "posterior_vs_map_center_rms_epsilon_shallower_than_0p9m": shallow_update_rms,
            "posterior_vs_original_neural_bvi_rms_epsilon_shallower_than_0p9m": shallow_original_rms,
            "top_mean_abs_difference_neural_bvi_vs_fwi": top_mean_gap,
            "mean_sigma_epsilon_above_0p9m": float(std_audit["regional_epsilon_std"]["top_0_0p9m"]["mean"]),
        },
        "time_windows": windows,
        "interpretation_boundary": (
            "No field permittivity truth is available. The FWI-consistent upper-layer mean, bounded lateral "
            "variation, and lower shallow-window RMSE support response consistency, not unique recovery. Strong "
            "0.5-0.9 m lateral structure and FWI's lower one-sided positive-envelope excess retain a shallow "
            "non-uniqueness boundary under source correction."
        ),
    }


def write_shallow_audit(audit: dict[str, Any]) -> None:
    json_path = PAPER_FIGURES.parent / "FIELD_NEURAL_BVI_SHALLOW_AUDIT.json"
    md_path = PAPER_FIGURES.parent / "FIELD_NEURAL_BVI_SHALLOW_AUDIT.md"
    json_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    top = audit["model_bands"]["neural_bvi_top_0_0p5m"]
    transition = audit["model_bands"]["neural_bvi_transition_0p5_0p9m"]
    comparisons = audit["model_comparisons"]
    lines = [
        "# Field Neural-BVI Shallow-Inversion Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        audit["assessment"],
        "",
        "| Diagnostic | Value |",
        "| --- | ---: |",
        f"| Top 0-0.5 m mean epsilon | {top['mean_epsilon']:.4f} |",
        f"| Top 0-0.5 m epsilon range | {top['min_epsilon']:.4f}-{top['max_epsilon']:.4f} |",
        f"| Top 0-0.5 m lateral-deviation q95 | {top['row_centered_abs_deviation_q95']:.4f} |",
        f"| Transition 0.5-0.9 m lateral-deviation q95 | {transition['row_centered_abs_deviation_q95']:.4f} |",
        f"| Posterior vs. MAP-center RMS epsilon shallower than 0.9 m | {comparisons['posterior_vs_map_center_rms_epsilon_shallower_than_0p9m']:.4f} |",
        f"| Neural-BVI/FWI top-mean absolute difference | {comparisons['top_mean_abs_difference_neural_bvi_vs_fwi']:.4f} |",
        "",
        "| Time window | Neural-BVI RMSE | FWI RMSE | Previous RMSE | Neural-BVI excess | FWI excess | Previous excess |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, window in audit["time_windows"].items():
        lines.append(
            f"| {label.replace('_', '-')} | {window['neural_bvi_rmse']:.4f} | {window['fwi_rmse']:.4f} | "
            f"{window['previous_neural_bvi_rmse']:.4f} | {window['neural_bvi_positive_envelope_excess']:.4f} | "
            f"{window['fwi_positive_envelope_excess']:.4f} | {window['previous_neural_bvi_positive_envelope_excess']:.4f} |"
        )
    lines.extend(["", audit["interpretation_boundary"]])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    refine_dir = args.refine_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "hyperbola_conditioned_bvi_arrays.npz"
    field_arrays = np.load(FIELD_DIR / "la010010_full_laplace_arrays.npz")
    refine_arrays = np.load(refine_dir / "la010010_field_fwi_arrays.npz")
    previous_arrays = np.load(LEGACY_DIR / "hyperbola_conditioned_bvi_arrays.npz")
    fwi_arrays = np.load(FWI_DIR / "la010010_field_fwi_arrays.npz")
    field_summary = load_json(FIELD_DIR / "summary.json")
    refine_summary = load_json(refine_dir / "summary.json")
    fwi_summary = load_json(FWI_DIR / "summary.json")
    hyperbola = load_json(HYPERBOLA_PROVENANCE)
    profile = field_summary["forward_backend"]["acquisition_profile"]
    width_m = float(profile["model_domain"]["width_m"])
    depth_m = float(profile["model_domain"]["target_depth_m"])

    if not args.render_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("Hyperbola-conditioned field BVI requires CUDA")
        observation = torch.from_numpy(field_arrays["observation"]).float()
        map_center = add_fixed_hyperbola_atoms(
            torch.from_numpy(refine_arrays["fwi"]).float().to(device),
            hyperbola,
            width_m,
            depth_m,
        ).cpu()
        forward, _ = load_surrogate_checkpoint(ROOT / "surrogate" / "best.pt", device)
        matched_forward = AdaptiveMatchedForward(forward, observation.to(device))
        basis_payload = torch.load(
            ROOT / "basis" / "error_pca_mean_latent16_train96_snrs0-5-10_seed7.pt",
            map_location="cpu",
            weights_only=False,
        )
        set_seed(20_271_423)
        result, samples, _ = run_map_bvi_case(
            matched_forward,
            observation,
            None,
            None,
            map_center,
            latent_dim=16,
            residual_scale=0.003,
            model_prior_std=0.005,
            latent_prior_weight=0.5,
            steps=8,
            lr=0.03,
            basis_type="error_pca_mean",
            posterior_latent_std=0.03,
            posterior_sample_count=128,
            event_threshold=0.5,
            device=device,
            noise_std_override=0.08,
            basis_override=basis_payload["basis"].to(device),
            latent_prior_mean=torch.zeros(16, device=device),
            latent_prior_std=(basis_payload["latent_raw_std"] / 0.003).to(device),
            latent_init="zero",
            model_prior_center="nn",
            posterior_sampler="laplace_full",
            include_map_center_sample=False,
        )
        posterior_mean = samples.mean(dim=0, keepdim=True)
        posterior_std = samples.std(dim=0, keepdim=True, unbiased=False)
        with torch.no_grad():
            prediction = matched_forward(posterior_mean.to(device)).cpu()
        predictive_audit = posterior_predictive_audit(
            samples,
            matched_forward,
            observation.to(device),
        )
        np.savez_compressed(
            arrays_path,
            observation=observation.numpy(),
            original_mean=field_arrays["posterior_mean"],
            map_center=map_center.numpy(),
            posterior_samples=samples.numpy(),
            posterior_mean=posterior_mean.numpy(),
            posterior_std=posterior_std.numpy(),
            prediction=prediction.numpy(),
            refine_prediction=refine_arrays["prediction_fwi"],
            previous_prediction=previous_arrays["prediction"],
            fwi_model=fwi_arrays["fwi"],
            fwi_prediction=fwi_arrays["prediction_fwi"],
        )
        bvi_result = result.__dict__
    else:
        existing_summary = load_json(out_dir / "summary.json") if (out_dir / "summary.json").exists() else {}
        bvi_result = existing_summary.get("bvi_result")
        predictive_audit = existing_summary.get("std_audit", {}).get("posterior_predictive")

    arrays_npz = np.load(arrays_path)
    observation_np = arrays_npz["observation"].squeeze().astype(np.float64)
    optimized_prediction = arrays_npz["prediction"].squeeze().astype(np.float64)
    previous_prediction = arrays_npz["previous_prediction"].squeeze().astype(np.float64)
    fwi_prediction = arrays_npz["fwi_prediction"].squeeze().astype(np.float64)
    optimized_tensor = torch.from_numpy(optimized_prediction)[None, None].float()
    observation_tensor = torch.from_numpy(observation_np)[None, None].float()
    optimized_metrics = data_metrics(optimized_tensor, observation_tensor)
    sample_interval_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    optimized_metrics.update(time_metrics(optimized_prediction, observation_np, sample_interval_ns))
    previous_tensor = torch.from_numpy(previous_prediction)[None, None].float()
    previous_metrics = data_metrics(previous_tensor, observation_tensor)
    previous_metrics.update(time_metrics(previous_prediction, observation_np, sample_interval_ns))
    fwi_metrics = dict(fwi_summary["comparison"]["fwi"])
    fwi_metrics.update(time_metrics(fwi_prediction, observation_np, sample_interval_ns))
    optimized_shape = morphology_metrics(
        2.0 + 8.0 * arrays_npz["posterior_mean"].squeeze(),
        hyperbola,
        width_m,
        depth_m,
    )
    fwi_shape = morphology_metrics(
        2.0 + 8.0 * arrays_npz["fwi_model"].squeeze(),
        hyperbola,
        width_m,
        depth_m,
    )
    gate = comparison_gate(
        optimized_metrics,
        fwi_metrics,
        previous_metrics,
        optimized_shape,
        fwi_shape,
    )

    figure_arrays = {key: arrays_npz[key] for key in arrays_npz.files}
    save_figure(figure_arrays, profile, field_summary["crop"])
    final_display = save_final_figure(figure_arrays, profile, field_summary["crop"])
    std_audit = build_std_audit(figure_arrays, profile, bvi_result, predictive_audit)
    write_std_audit(std_audit)
    shallow_audit = build_shallow_audit(
        figure_arrays,
        profile,
        optimized_metrics,
        fwi_metrics,
        previous_metrics,
        std_audit,
    )
    write_shallow_audit(shallow_audit)
    metrics_path = PAPER_FIGURES / "fig_field_optimized_neural_bvi_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "optimized_neural_bvi", "traditional_fwi", "previous_neural_bvi", "optimized_passes"])
        for metric in (
            "rmse",
            "pearson",
            "trace_ncc",
            "envelope_rmse",
            "rmse_0_16ns",
            "rmse_16_70ns",
            "rmse_0_12ns",
            "rmse_12_20ns",
            "rmse_20_30ns",
        ):
            higher = metric in {"pearson", "trace_ncc"}
            passed = optimized_metrics[metric] > fwi_metrics[metric] if higher else optimized_metrics[metric] < fwi_metrics[metric]
            writer.writerow([metric, optimized_metrics[metric], fwi_metrics[metric], previous_metrics[metric], int(passed)])
        for metric in (
            "envelope_overshoot_0_12ns",
            "envelope_overshoot_12_20ns",
            "envelope_overshoot_20_30ns",
            "envelope_overshoot_0_20ns",
            "envelope_overshoot_0_30ns",
        ):
            passed = optimized_metrics[metric] < previous_metrics[metric]
            writer.writerow([metric, optimized_metrics[metric], fwi_metrics[metric], previous_metrics[metric], int(passed)])
        for event_id in ("left", "right"):
            for metric in ("centroid_distance_from_hyperbola_m", "half_prominence_area_m2", "peak_contrast_epsilon"):
                higher = metric == "peak_contrast_epsilon"
                passed = optimized_shape[event_id][metric] > fwi_shape[event_id][metric] if higher else optimized_shape[event_id][metric] < fwi_shape[event_id][metric]
                writer.writerow([f"{event_id}_{metric}", optimized_shape[event_id][metric], fwi_shape[event_id][metric], "", int(passed)])

    summary = {
        "status": "complete",
        "method": "hyperbola-conditioned trust-region Neural-BVI with adaptive-source Deepwave likelihood",
        "config": {
            "refinement_control_size": 64,
            "refinement_regularization_scale": 500.0,
            "refinement_prior_center": "neural_bvi",
            "refinement_initialization": "neural_bvi",
            "direct_wave_weight": refine_summary["config"]["direct_wave_weight"],
            "shallow_overshoot_scale": refine_summary["config"]["shallow_overshoot_scale"],
            "shallow_overshoot_end_ns": refine_summary["config"]["shallow_overshoot_end_ns"],
            "shallow_lateral_smoothing_depth_m": refine_summary["config"]["shallow_lateral_smoothing_depth_m"],
            "shallow_lateral_smoothing_strength": refine_summary["config"]["shallow_lateral_smoothing_strength"],
            "target_atoms": "two fixed anisotropic Gaussians from extracted apex positions and original Neural-BVI half-prominence morphology",
            "local_bvi_latent_dim": 16,
            "local_bvi_steps": 8,
            "posterior_samples": 128,
        },
        "optimized_metrics": optimized_metrics,
        "previous_optimized_metrics": previous_metrics,
        "traditional_fwi_metrics": fwi_metrics,
        "optimized_morphology": optimized_shape,
        "traditional_fwi_morphology": fwi_shape,
        "comparison_gate": gate,
        "bvi_result": bvi_result,
        "std_audit": std_audit,
        "shallow_audit": shallow_audit,
        "arrays": str(arrays_path),
        "figure": str(PAPER_FIGURES / "fig_field_optimized_neural_bvi_comparison.pdf"),
        "main_figure": str(PAPER_FIGURES / "fig_field_neural_bvi_final_comparison.pdf"),
        "main_figure_display_ranges": final_display,
        "metrics_csv": str(metrics_path),
        "source_data_csv": str(PAPER_FIGURES / "fig_field_optimized_neural_bvi_shallow_source_data.csv"),
        "claim_boundary": (
            "This is post-hoc conditioning on the same measured record and uses a 64x64 refinement space versus the 16x16 conventional-FWI baseline. "
            "It demonstrates attainable joint data-fit and morphology performance but is not independent field generalization or field accuracy."
        ),
    }
    summary_path = PAPER_FIGURES / "fig_field_optimized_neural_bvi_provenance.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "gate": gate, "optimized_metrics": optimized_metrics}, indent=2))


if __name__ == "__main__":
    main()
