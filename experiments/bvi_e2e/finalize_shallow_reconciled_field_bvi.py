"""Render and audit the shallow-reconciled measured-data Neural-BVI result."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from finalize_hyperbola_conditioned_field_bvi import (
    build_shallow_audit,
    build_std_audit,
    time_metrics,
    write_shallow_audit,
    write_std_audit,
)
from run_deepwave_field_fwi import data_metrics


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_laplace_nn" / "noise_0p080"
NEW_DIR = ROOT / "field_shallow_reconciled_bvi"
PREVIOUS_DIR = ROOT / "field_hyperbola_conditioned_bvi_shallow"
SEARCH_DIR = ROOT / "field_shallow_reconciliation_search"
PAPER_DIR = HERE.parents[1] / "paper_grsl"
FIGURE_DIR = PAPER_DIR / "figures"

NEW_ARRAYS = NEW_DIR / "shallow_reconciled_bvi_arrays.npz"
NEW_SUMMARY = NEW_DIR / "summary.json"
PREVIOUS_ARRAYS = PREVIOUS_DIR / "hyperbola_conditioned_bvi_arrays.npz"
SELECTED_CONTROL = SEARCH_DIR / "coarse_end_0.65_balanced5.npz"

MODEL_VMIN = 2.2
MODEL_VMAX = 6.2


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_payload(
    prediction: np.ndarray,
    observation: np.ndarray,
    sample_interval_ns: float,
) -> dict[str, float]:
    prediction_tensor = torch.from_numpy(prediction)[None, None].float()
    observation_tensor = torch.from_numpy(observation)[None, None].float()
    metrics = {
        key: float(value)
        for key, value in data_metrics(prediction_tensor, observation_tensor).items()
    }
    metrics.update(time_metrics(prediction, observation, sample_interval_ns))
    return metrics


def mapped_arrays(new: np.lib.npyio.NpzFile, previous: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        "observation": new["observation"],
        "original_mean": new["previous_posterior_mean"],
        "map_center": new["selected_center"],
        "posterior_samples": new["posterior_samples"],
        "posterior_mean": new["posterior_mean"],
        "posterior_std": new["posterior_std"],
        "prediction": new["prediction"],
        "previous_prediction": previous["prediction"],
        "fwi_model": new["fwi_model"],
        "fwi_prediction": new["fwi_prediction"],
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(axis: plt.Axes, label: str, *, x: float = -0.14) -> None:
    axis.text(
        x,
        1.03,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def panel_labels(axes: np.ndarray) -> None:
    for label, axis in zip("abcdefgh", axes.ravel()):
        add_panel_label(axis, label)


def set_panel_title(axis: plt.Axes, label: str, title: str) -> None:
    axis.set_title(rf"$\bf{{({label})}}$  {title}", loc="left", pad=3.0)


def save_all_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)


def render_main_figure(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    crop: dict[str, Any],
    *,
    std_title: str = "Uncertainty (std.)",
) -> dict[str, Any]:
    observation = arrays["observation"].squeeze()
    neural = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    neural_std = 8.0 * arrays["posterior_std"].squeeze()
    fwi = 2.0 + 8.0 * arrays["fwi_model"].squeeze()
    neural_prediction = arrays["prediction"].squeeze()
    fwi_prediction = arrays["fwi_prediction"].squeeze()
    neural_residual = np.abs(neural_prediction - observation)
    fwi_residual = np.abs(fwi_prediction - observation)

    bscan_limit = max(
        float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        [observation.ravel(), neural_prediction.ravel(), fwi_prediction.ravel()]
                    )
                ),
                0.995,
            )
        ),
        1.0e-6,
    )
    residual_limit = max(
        float(np.quantile(np.concatenate([neural_residual.ravel(), fwi_residual.ravel()]), 0.995)),
        1.0e-6,
    )
    std_limit = max(float(np.quantile(neural_std, 0.995)), 1.0e-6)
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

    sample_interval_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    neural_metrics = metric_payload(neural_prediction, observation, sample_interval_ns)
    fwi_metrics = metric_payload(fwi_prediction, observation, sample_interval_ns)

    overall_specs = (
        ("Data RMSE", "rmse", "lower"),
        ("Envelope RMSE", "envelope_rmse", "lower"),
        ("Pearson", "pearson", "higher"),
        ("Trace NCC", "trace_ncc", "higher"),
    )
    overall_improvement: dict[str, float] = {}
    for label, key, direction in overall_specs:
        reference = max(abs(float(fwi_metrics[key])), 1.0e-12)
        delta = (
            float(fwi_metrics[key]) - float(neural_metrics[key])
            if direction == "lower"
            else float(neural_metrics[key]) - float(fwi_metrics[key])
        )
        overall_improvement[label] = 100.0 * delta / reference

    window_specs = (
        ("0-12", "rmse_0_12ns"),
        ("12-20", "rmse_12_20ns"),
        ("20-30", "rmse_20_30ns"),
        ("0-20", "rmse_0_20ns"),
    )
    window_improvement: dict[str, float] = {}
    for label, key in window_specs:
        reference = max(abs(float(fwi_metrics[key])), 1.0e-12)
        window_improvement[label] = 100.0 * (
            float(fwi_metrics[key]) - float(neural_metrics[key])
        ) / reference

    fig = plt.figure(figsize=(7.16, 4.10))
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(1.0, 1.0, 0.68),
        left=0.070,
        right=0.960,
        bottom=0.105,
        top=0.975,
        hspace=0.52,
    )
    image_axes = []
    for row in range(2):
        row_grid = outer_grid[row].subgridspec(1, 4, wspace=0.34)
        image_axes.append([fig.add_subplot(row_grid[0, column]) for column in range(4)])
    axes = np.asarray(image_axes)
    metric_grid = outer_grid[2].subgridspec(1, 2, wspace=0.34)
    overall_axis = fig.add_subplot(metric_grid[0, 0])
    window_axis = fig.add_subplot(metric_grid[0, 1])
    axes[0, 0].imshow(
        observation,
        cmap="gray",
        vmin=-bscan_limit,
        vmax=bscan_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    set_panel_title(axes[0, 0], "a", "Measured B-scan")
    model_image = axes[0, 1].imshow(
        neural,
        cmap="viridis",
        vmin=MODEL_VMIN,
        vmax=MODEL_VMAX,
        extent=model_extent,
        aspect="auto",
    )
    set_panel_title(axes[0, 1], "b", "Neural-BVI mean")
    std_image = axes[0, 2].imshow(
        neural_std,
        cmap="magma",
        vmin=0.0,
        vmax=std_limit,
        extent=model_extent,
        aspect="auto",
    )
    set_panel_title(axes[0, 2], "c", std_title)
    axes[0, 3].imshow(
        fwi,
        cmap="viridis",
        vmin=MODEL_VMIN,
        vmax=MODEL_VMAX,
        extent=model_extent,
        aspect="auto",
    )
    set_panel_title(axes[0, 3], "d", "FWI")

    for axis, image, title in zip(
        axes[1, :2],
        (neural_prediction, fwi_prediction),
        (("e", "Neural-BVI reforward"), ("f", "FWI reforward")),
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=-bscan_limit,
            vmax=bscan_limit,
            extent=bscan_extent,
            aspect="auto",
        )
        set_panel_title(axis, title[0], title[1])
    residual_image = axes[1, 2].imshow(
        neural_residual,
        cmap="inferno",
        vmin=0.0,
        vmax=residual_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    set_panel_title(axes[1, 2], "g", "Neural-BVI residual")
    axes[1, 3].imshow(
        fwi_residual,
        cmap="inferno",
        vmin=0.0,
        vmax=residual_limit,
        extent=bscan_extent,
        aspect="auto",
    )
    set_panel_title(axes[1, 3], "h", "FWI residual")

    axes[0, 0].set_ylabel("Time (ns)")
    axes[0, 1].set_ylabel("Depth (m)")
    axes[1, 0].set_ylabel("Time (ns)")
    for axis in (axes[0, 2], axes[0, 3], *axes[1, 1:]):
        axis.tick_params(labelleft=False)

    def inset_colorbar(image: Any, axis: plt.Axes, label: str) -> None:
        color_axis = axis.inset_axes([1.045, 0.12, 0.040, 0.76])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label(label, labelpad=2.0)
        colorbar.ax.tick_params(labelsize=7.0, length=2.5, width=0.6, pad=1.5)
        colorbar.outline.set_linewidth(0.6)

    inset_colorbar(std_image, axes[0, 2], r"$\sigma_{\epsilon_r}$")
    inset_colorbar(model_image, axes[0, 3], r"$\epsilon_r$")
    inset_colorbar(residual_image, axes[1, 3], "Absolute residual")

    blue = "#0072B2"
    pale_blue = "#9ECAE1"
    gray = "#595959"
    pale_gray = "#BDBDBD"
    overall_values = np.asarray(list(overall_improvement.values()), dtype=np.float64)
    overall_positions = np.arange(overall_values.size)[::-1]
    overall_axis.hlines(overall_positions, 0.0, overall_values, color=pale_blue, linewidth=2.2)
    overall_axis.scatter(
        overall_values,
        overall_positions,
        s=24,
        color=blue,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    overall_axis.set_yticks(overall_positions, list(overall_improvement.keys()))
    overall_axis.set_xlim(0.0, max(4.5, float(overall_values.max()) * 1.18))
    overall_axis.set_xlabel("Favorable change vs. FWI (%)")
    set_panel_title(overall_axis, "i", "Overall metric improvement")
    overall_axis.grid(axis="x", color="0.88", linewidth=0.5)
    overall_axis.set_axisbelow(True)
    for position, value in zip(overall_positions, overall_values):
        overall_axis.text(
            value + 0.08,
            position,
            f"{value:.1f}%",
            va="center",
            fontsize=7.5,
            color=blue,
        )

    window_positions = np.arange(len(window_specs))[::-1]
    neural_window_values = np.asarray(
        [float(neural_metrics[key]) for _, key in window_specs], dtype=np.float64
    )
    fwi_window_values = np.asarray(
        [float(fwi_metrics[key]) for _, key in window_specs], dtype=np.float64
    )
    window_reductions = np.asarray(list(window_improvement.values()), dtype=np.float64)
    window_axis.hlines(
        window_positions,
        neural_window_values,
        fwi_window_values,
        color=pale_gray,
        linewidth=1.5,
        zorder=1,
    )
    window_axis.scatter(
        neural_window_values,
        window_positions,
        s=24,
        color=blue,
        marker="o",
        label="Neural-BVI",
        zorder=3,
    )
    window_axis.scatter(
        fwi_window_values,
        window_positions,
        s=22,
        facecolor="white",
        edgecolor=gray,
        linewidth=1.0,
        marker="s",
        label="FWI",
        zorder=3,
    )
    window_axis.set_yticks(window_positions, [f"{label} ns" for label, _ in window_specs])
    window_axis.set_xlim(0.08, 0.390)
    window_axis.set_xlabel("RMSE")
    set_panel_title(window_axis, "j", "Time-window waveform fit")
    window_axis.grid(axis="x", color="0.88", linewidth=0.5)
    window_axis.set_axisbelow(True)
    for position, value in zip(window_positions, window_reductions):
        window_axis.text(
            0.384,
            position,
            f"{value:.1f}% lower",
            ha="right",
            va="center",
            fontsize=7.5,
            color=blue,
        )
    for axis in (overall_axis, window_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    save_all_formats(fig, FIGURE_DIR / "fig_field_neural_bvi_final_comparison")
    plt.close(fig)
    return {
        "model_vmin": MODEL_VMIN,
        "model_vmax": MODEL_VMAX,
        "bscan_abs_limit": bscan_limit,
        "residual_max": residual_limit,
        "std_max": std_limit,
        "overall_improvement_percent": overall_improvement,
        "time_window_rmse_reduction_percent": window_improvement,
    }


def render_diagnostic_figure(
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> None:
    observation = arrays["observation"].squeeze()
    previous = 2.0 + 8.0 * arrays["original_mean"].squeeze()
    neural = 2.0 + 8.0 * arrays["posterior_mean"].squeeze()
    fwi = 2.0 + 8.0 * arrays["fwi_model"].squeeze()
    previous_prediction = arrays["previous_prediction"].squeeze()
    neural_prediction = arrays["prediction"].squeeze()
    fwi_prediction = arrays["fwi_prediction"].squeeze()
    sample_interval_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    shallow_stop = min(observation.shape[0], int(round(30.0 / sample_interval_ns)))
    bscan_limit = max(
        float(
            np.quantile(
                np.abs(
                    np.concatenate(
                        [
                            observation[:shallow_stop].ravel(),
                            previous_prediction[:shallow_stop].ravel(),
                            neural_prediction[:shallow_stop].ravel(),
                            fwi_prediction[:shallow_stop].ravel(),
                        ]
                    )
                ),
                0.995,
            )
        ),
        1.0e-6,
    )
    width_m = float(profile["model_domain"]["width_m"])
    depth_m = float(profile["model_domain"]["target_depth_m"])
    model_extent = (0.0, width_m, depth_m, 0.0)
    shallow_extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        30.0,
        0.0,
    )
    depth = np.linspace(0.0, depth_m, neural.shape[0])
    profile_mask = depth <= 0.9

    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.85), constrained_layout=True)
    for axis, image, title in zip(
        axes[0, :3],
        (previous, neural, fwi),
        ("Previous Neural-BVI", "Neural-BVI mean", "Conventional FWI"),
    ):
        axis.imshow(
            image,
            cmap="viridis",
            vmin=MODEL_VMIN,
            vmax=MODEL_VMAX,
            extent=model_extent,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_ylabel("Depth (m)")
        axis.set_xlabel("Distance (m)")
    axes[0, 3].plot(previous.mean(axis=1)[profile_mask], depth[profile_mask], color="0.45", lw=1.0, label="Previous")
    axes[0, 3].plot(neural.mean(axis=1)[profile_mask], depth[profile_mask], color="#0072B2", lw=1.2, label="Neural-BVI")
    axes[0, 3].plot(fwi.mean(axis=1)[profile_mask], depth[profile_mask], color="#D55E00", lw=1.0, label="FWI")
    axes[0, 3].invert_yaxis()
    axes[0, 3].set_xlim(MODEL_VMIN, MODEL_VMAX)
    axes[0, 3].set_ylim(0.9, 0.0)
    axes[0, 3].set_title("Lateral-mean profile")
    axes[0, 3].set_xlabel(r"$\epsilon_r$")
    axes[0, 3].set_ylabel("Depth (m)")
    axes[0, 3].legend(loc="lower right", fontsize=5.2, handlelength=1.4)

    for axis, image, title in zip(
        axes[1, :],
        (
            observation[:shallow_stop],
            previous_prediction[:shallow_stop],
            neural_prediction[:shallow_stop],
            fwi_prediction[:shallow_stop],
        ),
        ("Measured B-scan", "Previous reforward", "Neural-BVI reforward", "FWI reforward"),
    ):
        axis.imshow(
            image,
            cmap="gray",
            vmin=-bscan_limit,
            vmax=bscan_limit,
            extent=shallow_extent,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Time (ns)")
    panel_labels(axes)
    save_all_formats(fig, FIGURE_DIR / "fig_field_shallow_reconciliation_diagnostic")
    plt.close(fig)


def write_metrics_csv(
    neural: dict[str, float],
    fwi: dict[str, float],
    previous: dict[str, float],
) -> None:
    path = FIGURE_DIR / "fig_field_shallow_reconciliation_metrics.csv"
    metrics = (
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
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "neural_bvi", "conventional_fwi", "previous_neural_bvi"])
        for metric in metrics:
            writer.writerow([metric, neural[metric], fwi[metric], previous[metric]])


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    field_summary = load_json(FIELD_DIR / "summary.json")
    new_summary = load_json(NEW_SUMMARY)
    profile = field_summary["forward_backend"]["acquisition_profile"]
    crop = field_summary["crop"]
    with np.load(NEW_ARRAYS) as new, np.load(PREVIOUS_ARRAYS) as previous:
        arrays = mapped_arrays(new, previous)

    observation = arrays["observation"].squeeze().astype(np.float64)
    sample_interval_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    neural_metrics = metric_payload(arrays["prediction"].squeeze().astype(np.float64), observation, sample_interval_ns)
    fwi_metrics = metric_payload(arrays["fwi_prediction"].squeeze().astype(np.float64), observation, sample_interval_ns)
    previous_metrics = metric_payload(arrays["previous_prediction"].squeeze().astype(np.float64), observation, sample_interval_ns)

    display = render_main_figure(arrays, profile, crop)
    render_diagnostic_figure(arrays, profile, crop)
    write_metrics_csv(neural_metrics, fwi_metrics, previous_metrics)

    std_audit = build_std_audit(
        arrays,
        profile,
        new_summary["bvi_result"],
        new_summary["posterior_predictive"],
    )
    write_std_audit(std_audit)
    shallow_audit = build_shallow_audit(
        arrays,
        profile,
        neural_metrics,
        fwi_metrics,
        previous_metrics,
        std_audit,
    )
    write_shallow_audit(shallow_audit)

    selected = np.load(SELECTED_CONTROL)
    provenance = {
        "status": "complete",
        "method": "shallow-prior-constrained Neural-BVI with differentiable Deepwave data fitting",
        "selection": {
            "base_epsilon_shift": 2.75,
            "full_shift_depth_m": 0.5,
            "cosine_taper_end_depth_m": 0.65,
            "control_shape": list(selected["control"].shape[-2:]),
            "general_optimization_steps": 30,
            "balanced_12_20ns_steps": 5,
            "local_bvi_latent_dim": 16,
            "local_bvi_map_steps": 8,
            "posterior_samples": 128,
        },
        "neural_bvi_metrics": neural_metrics,
        "conventional_fwi_metrics": fwi_metrics,
        "previous_neural_bvi_metrics": previous_metrics,
        "display_ranges": display,
        "std_audit": std_audit,
        "shallow_audit": shallow_audit,
        "claim_boundary": (
            "No measured permittivity truth is available. This is an FWI-consistent and Deepwave-data-consistent "
            "upper-layer solution under an explicit shallow prior, not a uniquely validated field model."
        ),
    }
    (FIGURE_DIR / "fig_field_shallow_reconciliation_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    selected.close()
    print(
        json.dumps(
            {
                "status": "complete",
                "neural_bvi": neural_metrics,
                "fwi": fwi_metrics,
                "shallow_audit": shallow_audit["status"],
                "std_audit": std_audit["status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
