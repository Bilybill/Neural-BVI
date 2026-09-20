"""Extract two measured reflection hyperbolas and compare apparent morphology."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.signal import hilbert


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_laplace_nn" / "noise_0p080"
FWI_DIR = ROOT / "field_fwi_adaptive_source" / "eps5p5_c16_reg500_full"
SCAN_DIR = ROOT / "field_velocity_scan"
PAPER_FIGURES = HERE.parents[1] / "paper_grsl" / "figures"
C0 = 299_792_458.0


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hyperbola_time_ns(parameters: np.ndarray, distance_m: np.ndarray) -> np.ndarray:
    apex_distance_m, apex_time_ns, epsilon = parameters
    moveout_ns2 = 4.0 * (distance_m - apex_distance_m) ** 2 * epsilon / C0**2 * 1.0e18
    return np.sqrt(apex_time_ns**2 + moveout_ns2)


def normalized_envelope(observation: np.ndarray) -> np.ndarray:
    envelope = np.abs(hilbert(observation, axis=0))
    trace_rms = np.sqrt(np.mean(np.square(observation), axis=0, keepdims=True))
    normalized = envelope / (trace_rms + 1.0e-6)
    normalized /= gaussian_filter(normalized, sigma=(12.0, 0.0)) + 0.08
    return gaussian_filter(normalized, sigma=(1.0, 0.7))


def interpolate_curve(image: np.ndarray, sample_index: np.ndarray) -> np.ndarray:
    lower = np.floor(sample_index).astype(np.int64)
    fraction = sample_index - lower
    valid = (lower >= 0) & (lower < image.shape[0] - 1)
    values = np.zeros(image.shape[1], dtype=np.float64)
    columns = np.flatnonzero(valid)
    values[columns] = (
        (1.0 - fraction[columns]) * image[lower[columns], columns]
        + fraction[columns] * image[lower[columns] + 1, columns]
    )
    return values


def scan_local_hyperbola(
    envelope: np.ndarray,
    distance_m: np.ndarray,
    dt_ns: float,
    *,
    apex_distance_range_m: tuple[float, float],
    apex_time_range_ns: tuple[float, float],
) -> np.ndarray:
    x_candidates = distance_m[
        (distance_m >= apex_distance_range_m[0]) & (distance_m <= apex_distance_range_m[1])
    ][::2]
    t_candidates = np.arange(apex_time_range_ns[0], apex_time_range_ns[1] + 0.01, 0.5)
    epsilon_candidates = np.arange(2.5, 9.01, 0.1)
    best_score = -np.inf
    best = None
    for apex_distance in x_candidates:
        aperture_weight = np.exp(-0.5 * ((distance_m - apex_distance) / 0.5) ** 2)
        aperture_weight[np.abs(distance_m - apex_distance) > 0.8] = 0.0
        aperture_weight /= aperture_weight.sum() + 1.0e-12
        for apex_time in t_candidates:
            for epsilon in epsilon_candidates:
                candidate = np.asarray([apex_distance, apex_time, epsilon])
                values = interpolate_curve(envelope, hyperbola_time_ns(candidate, distance_m) / dt_ns)
                score = float(np.sum(values * aperture_weight))
                if score > best_score:
                    best_score = score
                    best = candidate
    if best is None:
        raise RuntimeError("Local hyperbola scan did not produce a finite candidate")
    return best


def extract_hyperbola(
    envelope: np.ndarray,
    distance_m: np.ndarray,
    dt_ns: float,
    initial: np.ndarray,
    *,
    aperture_half_width_m: float,
) -> dict[str, np.ndarray | float]:
    initial_time = hyperbola_time_ns(initial, distance_m)
    trace_indices = np.flatnonzero(np.abs(distance_m - initial[0]) <= aperture_half_width_m)
    half_window = max(2, int(round(3.0 / dt_ns)))
    picked_time: list[float] = []
    picked_score: list[float] = []
    picked_distance: list[float] = []
    for trace_index in trace_indices:
        center = int(round(initial_time[trace_index] / dt_ns))
        lower = max(0, center - half_window)
        upper = min(envelope.shape[0], center + half_window + 1)
        peak = lower + int(np.argmax(envelope[lower:upper, trace_index]))
        picked_distance.append(float(distance_m[trace_index]))
        picked_time.append(float(peak * dt_ns))
        picked_score.append(float(envelope[peak, trace_index]))

    x = np.asarray(picked_distance)
    t = np.asarray(picked_time)
    score = np.asarray(picked_score)
    weight = np.clip(score / np.median(score), 0.4, 2.5)
    centered_x = x - initial[0]
    design = np.column_stack([np.square(centered_x), centered_x, np.ones_like(centered_x)])
    robust_weight = np.ones_like(weight)
    inlier = np.ones_like(t, dtype=bool)
    fitted = initial.copy()
    for _ in range(8):
        combined = np.sqrt(weight * robust_weight)
        weighted_design = design * combined[:, None]
        weighted_target = np.square(t) * combined
        normal_matrix = np.asarray(
            [
                [float(np.sum(weighted_design[:, row] * weighted_design[:, col])) for col in range(3)]
                for row in range(3)
            ],
            dtype=float,
        )
        normal_target = np.asarray(
            [float(np.sum(weighted_design[:, row] * weighted_target)) for row in range(3)],
            dtype=float,
        )
        augmented = [list(normal_matrix[row]) + [float(normal_target[row])] for row in range(3)]
        for pivot in range(3):
            swap = max(range(pivot, 3), key=lambda row: abs(augmented[row][pivot]))
            if abs(augmented[swap][pivot]) < 1.0e-12:
                raise RuntimeError("Extracted ridge quadratic fit is singular")
            augmented[pivot], augmented[swap] = augmented[swap], augmented[pivot]
            scale = augmented[pivot][pivot]
            augmented[pivot] = [value / scale for value in augmented[pivot]]
            for row in range(3):
                if row == pivot:
                    continue
                factor = augmented[row][pivot]
                augmented[row] = [
                    augmented[row][col] - factor * augmented[pivot][col] for col in range(4)
                ]
        coefficients = np.asarray([augmented[row][3] for row in range(3)])
        curvature, linear, constant = coefficients
        if curvature <= 0.0:
            raise RuntimeError("Extracted ridge does not define a positive-curvature hyperbola")
        offset = -linear / (2.0 * curvature)
        apex_time_squared = constant - curvature * offset**2
        if apex_time_squared <= 0.0:
            raise RuntimeError("Extracted ridge does not define a physical apex time")
        fitted = np.asarray(
            [
                initial[0] + offset,
                np.sqrt(apex_time_squared),
                curvature * C0**2 / (4.0 * 1.0e18),
            ]
        )
        residual = hyperbola_time_ns(fitted, x) - t
        center = np.median(residual)
        mad = 1.4826 * np.median(np.abs(residual - center))
        scale = max(mad, 0.25)
        standardized = np.abs(residual - center) / (1.345 * scale)
        robust_weight = np.minimum(1.0, 1.0 / np.maximum(standardized, 1.0e-12))
        inlier = np.abs(residual - center) <= max(1.2, 2.5 * mad)

    fitted_time = hyperbola_time_ns(fitted, x)
    fit_residual = t - fitted_time
    return {
        "distance_m": x,
        "picked_time_ns": t,
        "score": score,
        "inlier": inlier,
        "fitted_parameters": fitted,
        "fitted_time_ns": fitted_time,
        "residual_ns": fit_residual,
        "fit_rmse_ns": float(np.sqrt(np.mean(np.square(fit_residual[inlier])))),
    }


def apparent_morphology(
    model: np.ndarray,
    x_m: np.ndarray,
    depth_m: np.ndarray,
    expected_center: tuple[float, float],
) -> dict[str, object]:
    x_grid, z_grid = np.meshgrid(x_m, depth_m)
    x0, z0 = expected_center
    search = (np.abs(x_grid - x0) <= 0.45) & (np.abs(z_grid - z0) <= 0.45)
    background = gaussian_filter(model, sigma=(18.0, 18.0))
    contrast = gaussian_filter(model - background, sigma=1.2)
    peak_index = np.unravel_index(int(np.argmax(np.where(search, contrast, -np.inf))), model.shape)
    peak_value = float(contrast[peak_index])
    threshold = 0.5 * peak_value
    components, _ = label((contrast >= threshold) & search)
    component_id = int(components[peak_index])
    component = components == component_id
    if component_id == 0 or int(component.sum()) < 3:
        raise RuntimeError("Could not isolate a connected target morphology")

    weights = np.maximum(contrast[component], 0.0)
    x_values = x_grid[component]
    z_values = z_grid[component]
    centroid_x = float(np.average(x_values, weights=weights))
    centroid_z = float(np.average(z_values, weights=weights))
    centered_x = x_values - centroid_x
    centered_z = z_values - centroid_z
    weight_sum = float(weights.sum())
    covariance_xx = float(np.sum(weights * centered_x * centered_x) / weight_sum)
    covariance_xz = float(np.sum(weights * centered_x * centered_z) / weight_sum)
    covariance_zz = float(np.sum(weights * centered_z * centered_z) / weight_sum)
    orientation = float(
        np.degrees(0.5 * np.arctan2(2.0 * covariance_xz, covariance_xx - covariance_zz))
    )
    if orientation > 90.0:
        orientation -= 180.0
    if orientation <= -90.0:
        orientation += 180.0
    return {
        "contrast": contrast,
        "component": component,
        "peak_contrast": peak_value,
        "threshold": threshold,
        "centroid": (centroid_x, centroid_z),
        "peak": (float(x_m[peak_index[1]]), float(depth_m[peak_index[0]])),
        "width_m": float(x_values.max() - x_values.min() + np.mean(np.diff(x_m))),
        "thickness_m": float(z_values.max() - z_values.min() + np.mean(np.diff(depth_m))),
        "orientation_deg": orientation,
        "pixel_count": int(component.sum()),
    }


def morphology_payload(shape: dict[str, object], event: dict[str, object], start_m: float) -> dict[str, object]:
    centroid = shape["centroid"]
    peak = shape["peak"]
    lateral_offset = float(centroid[0] - float(event["local_apex_x_m"]))
    depth_offset = float(centroid[1] - float(event["fitted_depth_m"]))
    return {
        "centroid_local_x_m": float(centroid[0]),
        "centroid_global_x_m": float(centroid[0] + start_m),
        "centroid_depth_m": float(centroid[1]),
        "peak_local_x_m": float(peak[0]),
        "peak_global_x_m": float(peak[0] + start_m),
        "peak_depth_m": float(peak[1]),
        "centroid_lateral_offset_from_hyperbola_m": lateral_offset,
        "centroid_depth_offset_from_hyperbola_m": depth_offset,
        "centroid_distance_from_hyperbola_m": float(np.hypot(lateral_offset, depth_offset)),
        "half_prominence_width_m": float(shape["width_m"]),
        "half_prominence_thickness_m": float(shape["thickness_m"]),
        "orientation_deg": float(shape["orientation_deg"]),
        "peak_contrast_epsilon": float(shape["peak_contrast"]),
        "pixel_count": int(shape["pixel_count"]),
    }


def main() -> None:
    field_arrays = np.load(FIELD_DIR / "la010010_full_laplace_arrays.npz")
    fwi_arrays = np.load(FWI_DIR / "la010010_field_fwi_arrays.npz")
    field_summary = load_json(FIELD_DIR / "summary.json")
    scan_summary = load_json(SCAN_DIR / "summary.json")
    profile = field_summary["forward_backend"]["acquisition_profile"]

    observation = field_arrays["observation"].squeeze().astype(np.float64)
    neural_bvi = 2.0 + 8.0 * field_arrays["posterior_mean"].squeeze().astype(np.float64)
    fwi = 2.0 + 8.0 * fwi_arrays["fwi"].squeeze().astype(np.float64)
    fwi_update = fwi - 5.5
    dt_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    dx_m = float(profile["observation"]["trace_spacing_m"])
    start_m = float(field_summary["crop"]["start_distance_m"])
    distance_m = start_m + np.arange(observation.shape[1]) * dx_m
    envelope = normalized_envelope(observation)

    initial_left = scan_local_hyperbola(
        envelope,
        distance_m,
        dt_ns,
        apex_distance_range_m=(3.0, 3.5),
        apex_time_range_ns=(20.0, 30.0),
    )
    initial_right = np.asarray(
        [
            scan_summary["best_apex_distance_m"],
            scan_summary["best_apex_time_ns"],
            scan_summary["best_background_epsilon"],
        ],
        dtype=np.float64,
    )
    events: list[dict[str, object]] = [
        {"id": "left", "short": "L", "label": "Left event", "color": "#00A6A6", "initial": initial_left, "aperture": 0.75},
        {"id": "right", "short": "R", "label": "Right event", "color": "#D55E00", "initial": initial_right, "aperture": 1.0},
    ]

    width_m = float(profile["model_domain"]["width_m"])
    depth_limit_m = float(profile["model_domain"]["target_depth_m"])
    model_x_m = np.linspace(0.0, width_m, neural_bvi.shape[1])
    model_depth_m = np.linspace(0.0, depth_limit_m, neural_bvi.shape[0])
    for event in events:
        extracted = extract_hyperbola(
            envelope,
            distance_m,
            dt_ns,
            np.asarray(event["initial"]),
            aperture_half_width_m=float(event["aperture"]),
        )
        fitted = np.asarray(extracted["fitted_parameters"])
        depth_m = float(0.5 * C0 / np.sqrt(fitted[2]) * fitted[1] * 1.0e-9)
        local_x_m = float(fitted[0] - start_m)
        event.update(
            {
                "extracted": extracted,
                "fitted": fitted,
                "fitted_depth_m": depth_m,
                "local_apex_x_m": local_x_m,
                "neural_shape": apparent_morphology(neural_bvi, model_x_m, model_depth_m, (local_x_m, depth_m)),
                "fwi_shape": apparent_morphology(fwi, model_x_m, model_depth_m, (local_x_m, depth_m)),
            }
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
            "axes.spines.right": False,
            "axes.spines.top": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.16, 4.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.94, 1.06])
    ax_bscan = fig.add_subplot(grid[0, 0])
    ax_residual = fig.add_subplot(grid[0, 1])
    ax_neural = fig.add_subplot(grid[1, 0])
    ax_fwi = fig.add_subplot(grid[1, 1])

    obs_limit = max(float(np.quantile(np.abs(observation), 0.995)), 1.0e-6)
    ax_bscan.imshow(
        observation,
        cmap="gray",
        vmin=-obs_limit,
        vmax=obs_limit,
        extent=(distance_m[0], distance_m[-1], observation.shape[0] * dt_ns, 0.0),
        aspect="auto",
        interpolation="nearest",
    )
    ax_residual.axhline(0.0, color="0.4", linewidth=0.8)
    residual_summary = []
    for event in events:
        extracted = event["extracted"]
        fitted = np.asarray(event["fitted"])
        inlier = np.asarray(extracted["inlier"], dtype=bool)
        picked_x = np.asarray(extracted["distance_m"])
        picked_t = np.asarray(extracted["picked_time_ns"])
        residual = np.asarray(extracted["residual_ns"])
        color = str(event["color"])
        curve_x = np.linspace(picked_x.min(), picked_x.max(), 400)
        ax_bscan.plot(curve_x, hyperbola_time_ns(fitted, curve_x), color=color, linewidth=1.3, label=f"{event['label']} fit")
        ax_bscan.scatter(picked_x[inlier], picked_t[inlier], s=5, color=color, edgecolors="none", alpha=0.65)
        ax_bscan.plot(fitted[0], fitted[1], marker="+", color=color, markersize=7, markeredgewidth=1.2)
        ax_bscan.text(fitted[0] + 0.04, fitted[1] - 1.2, str(event["short"]), color=color, fontsize=6, fontweight="bold")
        ax_residual.scatter(picked_x[inlier], residual[inlier], s=8, color=color, alpha=0.82, label=str(event["label"]))
        if np.any(~inlier):
            ax_residual.scatter(picked_x[~inlier], residual[~inlier], s=10, marker="x", color=color, alpha=0.5)
        residual_summary.append(
            f"{event['short']}: $x_0={fitted[0]:.2f}$ m, $t_0={fitted[1]:.1f}$ ns, "
            f"$\\epsilon_b={fitted[2]:.2f}$, $z_0={float(event['fitted_depth_m']):.2f}$ m, "
            f"RMSE={float(extracted['fit_rmse_ns']):.2f} ns"
        )
    ax_bscan.set_title("Extracted reflection hyperbolas")
    ax_bscan.set_xlabel("Distance (m)")
    ax_bscan.set_ylabel("Time (ns)")
    ax_bscan.legend(loc="lower right", fontsize=5.0, frameon=True, framealpha=0.85)
    ax_residual.set_title("Ridge-fit residuals")
    ax_residual.set_xlabel("Distance (m)")
    ax_residual.set_ylabel("Picked $-$ fitted time (ns)")
    ax_residual.text(0.02, 0.97, "\n".join(residual_summary), transform=ax_residual.transAxes, va="top", fontsize=4.9)
    ax_residual.legend(loc="lower left", fontsize=5.0, frameon=False)

    model_extent = (0.0, width_m, depth_limit_m, 0.0)
    neural_image = ax_neural.imshow(neural_bvi, cmap="viridis", vmin=2.0, vmax=10.0, extent=model_extent, aspect="auto")
    update_limit = max(float(np.quantile(np.abs(fwi_update), 0.995)), 1.0e-3)
    fwi_image = ax_fwi.imshow(
        fwi_update,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-update_limit, vcenter=0.0, vmax=update_limit),
        extent=model_extent,
        aspect="auto",
    )
    ax_neural.set_title("Neural-BVI posterior mean")
    ax_fwi.set_title("FWI update")
    for axis in (ax_neural, ax_fwi):
        axis.set_xlabel("Local distance (m)")
        axis.set_ylabel("Depth (m)")
    cb_neural = fig.colorbar(neural_image, ax=ax_neural, orientation="horizontal", pad=0.16, fraction=0.06)
    cb_neural.set_label(r"$\epsilon_r$", labelpad=1)
    cb_fwi = fig.colorbar(fwi_image, ax=ax_fwi, orientation="horizontal", pad=0.16, fraction=0.06)
    cb_fwi.set_label(r"$\Delta\epsilon_r$", labelpad=1)

    for panel, axis in zip("abcd", (ax_bscan, ax_residual, ax_neural, ax_fwi)):
        axis.text(-0.13, 1.03, panel, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")

    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    stem = PAPER_FIGURES / "fig_field_hyperbola_morphology"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    picks_path = stem.with_name(f"{stem.name}_source_data.csv")
    with picks_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event", "distance_m", "picked_time_ns", "fitted_time_ns", "residual_ns", "ridge_score", "inlier"])
        for event in events:
            extracted = event["extracted"]
            inlier = np.asarray(extracted["inlier"], dtype=bool)
            for row in zip(
                np.asarray(extracted["distance_m"]),
                np.asarray(extracted["picked_time_ns"]),
                np.asarray(extracted["fitted_time_ns"]),
                np.asarray(extracted["residual_ns"]),
                np.asarray(extracted["score"]),
                inlier.astype(int),
            ):
                writer.writerow([event["id"], *row])

    event_payloads = {}
    for event in events:
        extracted = event["extracted"]
        fitted = np.asarray(event["fitted"])
        initial = np.asarray(event["initial"])
        inlier = np.asarray(extracted["inlier"], dtype=bool)
        event_payloads[str(event["id"])] = {
            "label": event["label"],
            "extraction": {
                "initial_apex_distance_m": float(initial[0]),
                "initial_apex_time_ns": float(initial[1]),
                "initial_background_epsilon": float(initial[2]),
                "fitted_apex_distance_m": float(fitted[0]),
                "fitted_apex_time_ns": float(fitted[1]),
                "fitted_background_epsilon": float(fitted[2]),
                "conditional_apex_depth_m": float(event["fitted_depth_m"]),
                "local_apex_x_m": float(event["local_apex_x_m"]),
                "picked_trace_count": int(len(np.asarray(extracted["distance_m"]))),
                "inlier_trace_count": int(inlier.sum()),
                "ridge_fit_rmse_ns": float(extracted["fit_rmse_ns"]),
            },
            "neural_bvi_apparent_morphology": morphology_payload(event["neural_shape"], event, start_m),
            "fwi_apparent_morphology": morphology_payload(event["fwi_shape"], event, start_m),
        }
    summary = {
        "figure": stem.name,
        "method": "two-event normalized Hilbert-envelope ridge picking followed by robust IRLS quadratic-hyperbola fitting",
        "left_local_scan_window": {"distance_m": [3.0, 3.5], "time_ns": [20.0, 30.0]},
        "events": event_payloads,
        "source_data": str(picks_path),
        "source_arrays": {
            "field": str(FIELD_DIR / "la010010_full_laplace_arrays.npz"),
            "fwi": str(FWI_DIR / "la010010_field_fwi_arrays.npz"),
            "right_initial_scan": str(SCAN_DIR / "summary.json"),
        },
        "display_processing": "Global clipping only. Inversion panels contain no target markers, contours, centroid links, or morphology text overlays; morphology is retained only in the numerical provenance.",
        "claim_boundary": (
            "Hyperbola-derived depths assume a homogeneous scalar-wave background and the recorded time zero. "
            "Reported widths and thicknesses are resolution- and prior-limited image-domain apparent morphology, not pipe diameter truth."
        ),
    }
    stem.with_name(f"{stem.name}_provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
