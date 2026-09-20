"""Multiscale conventional Deepwave FWI for the matched LA010010 field window."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from publication_training import load_surrogate_checkpoint, set_seed


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "publication_la010010" / "full"


@dataclass(frozen=True)
class FWIStage:
    cutoff_mhz: float
    iterations: int
    learning_rate: float
    smoothing_sigma: float
    mse_weight: float
    ncc_weight: float
    envelope_weight: float
    prior_weight: float
    tv_weight: float
    laplacian_weight: float


def default_stages(quick: bool) -> list[FWIStage]:
    if quick:
        return [
            FWIStage(220.0, 3, 0.060, 6.0, 0.40, 0.40, 0.25, 0.30, 0.08, 0.04),
            FWIStage(380.0, 3, 0.045, 3.0, 0.70, 0.30, 0.10, 0.20, 0.05, 0.025),
            FWIStage(620.0, 4, 0.025, 1.5, 1.00, 0.15, 0.00, 0.15, 0.03, 0.015),
        ]
    return [
        FWIStage(180.0, 12, 0.080, 6.0, 0.40, 0.40, 0.30, 0.40, 0.10, 0.050),
        FWIStage(280.0, 15, 0.060, 4.0, 0.60, 0.30, 0.20, 0.30, 0.07, 0.035),
        FWIStage(420.0, 18, 0.040, 2.5, 0.80, 0.20, 0.10, 0.22, 0.05, 0.025),
        FWIStage(620.0, 20, 0.025, 1.5, 1.00, 0.15, 0.00, 0.18, 0.04, 0.020),
        FWIStage(620.0, 15, 0.015, 1.0, 1.00, 0.10, 0.00, 0.15, 0.03, 0.015),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--field-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--init",
        choices=["constant", "smooth_1d", "neural", "neural_bvi"],
        default="constant",
    )
    parser.add_argument("--initial-epsilon", type=float, default=4.0)
    parser.add_argument("--epsilon-min", type=float, default=2.2)
    parser.add_argument("--epsilon-max", type=float, default=8.0)
    parser.add_argument("--control-size", type=int, default=64)
    parser.add_argument("--regularization-scale", type=float, default=1.0)
    parser.add_argument(
        "--prior-center",
        choices=["background", "neural", "neural_bvi"],
        default="background",
        help="Model-space trust-region center; background preserves the conventional-FWI contract.",
    )
    parser.add_argument(
        "--objective",
        choices=["fixed_wavelet", "source_independent", "adaptive_source"],
        default="fixed_wavelet",
    )
    parser.add_argument("--reference-count", type=int, default=3)
    parser.add_argument("--data-term-scale", type=float, default=1.0)
    parser.add_argument("--source-waterlevel", type=float, default=0.03)
    parser.add_argument("--source-smoothing-bins", type=float, default=2.0)
    parser.add_argument("--direct-wave-weight", type=float, default=0.25)
    parser.add_argument("--direct-wave-end-ns", type=float, default=16.0)
    parser.add_argument(
        "--shallow-overshoot-scale",
        type=float,
        default=0.0,
        help="One-sided Hilbert-envelope penalty for modeled energy above the measured shallow envelope.",
    )
    parser.add_argument("--shallow-overshoot-end-ns", type=float, default=12.0)
    parser.add_argument(
        "--shallow-lateral-smoothing-depth-m",
        type=float,
        default=0.0,
        help="Bottom of the tapered shallow zone whose unsupported lateral anomalies are shrunk.",
    )
    parser.add_argument("--shallow-lateral-smoothing-strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def gaussian_blur(model: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return model
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, dtype=model.dtype, device=model.device)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel = kernel / kernel.sum()
    padded = F.pad(model, (radius, radius, 0, 0), mode="reflect")
    blurred = F.conv2d(padded, kernel.view(1, 1, 1, -1))
    padded = F.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel.view(1, 1, -1, 1))


def frequency_lowpass(data: torch.Tensor, dt: float, cutoff_hz: float) -> torch.Tensor:
    n_time = data.shape[-2]
    frequencies = torch.fft.rfftfreq(n_time, d=float(dt), device=data.device)
    start = 0.82 * float(cutoff_hz)
    stop = float(cutoff_hz)
    phase = ((frequencies - start) / max(stop - start, 1.0)).clamp(0.0, 1.0)
    taper = torch.where(
        frequencies <= start,
        torch.ones_like(frequencies),
        torch.where(frequencies >= stop, torch.zeros_like(frequencies), 0.5 * (1.0 + torch.cos(torch.pi * phase))),
    )
    spectrum = torch.fft.rfft(data, dim=-2)
    return torch.fft.irfft(spectrum * taper.view(1, 1, -1, 1), n=n_time, dim=-2)


def analytic_envelope(data: torch.Tensor) -> torch.Tensor:
    n_time = data.shape[-2]
    multiplier = data.new_zeros(n_time)
    multiplier[0] = 1.0
    if n_time % 2 == 0:
        multiplier[1 : n_time // 2] = 2.0
        multiplier[n_time // 2] = 1.0
    else:
        multiplier[1 : (n_time + 1) // 2] = 2.0
    analytic = torch.fft.ifft(torch.fft.fft(data, dim=-2) * multiplier.view(1, 1, -1, 1), dim=-2)
    return analytic.abs()


def trace_ncc_loss(prediction: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-2, keepdim=True)
    observation = observation - observation.mean(dim=-2, keepdim=True)
    numerator = (prediction * observation).sum(dim=-2)
    denominator = torch.sqrt(prediction.square().sum(dim=-2) * observation.square().sum(dim=-2)).clamp_min(1.0e-8)
    return 1.0 - (numerator / denominator).mean()


def select_reference_traces(observation: torch.Tensor, count: int) -> list[int]:
    """Select high-energy references from aperture-wide, non-overlapping bins."""
    width = int(observation.shape[-1])
    count = max(1, min(int(count), width))
    edge = max(1, int(round(0.05 * width)))
    start = edge
    stop = max(start + count, width - edge)
    boundaries = torch.linspace(start, stop, count + 1).round().to(torch.int64)
    energy = observation.square().mean(dim=-2).mean(dim=(0, 1))
    indices: list[int] = []
    for bin_index in range(count):
        left = int(boundaries[bin_index])
        right = max(left + 1, int(boundaries[bin_index + 1]))
        right = min(right, width)
        indices.append(left + int(torch.argmax(energy[left:right])))
    return indices


def source_independent_convolution_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    reference_indices: list[int],
) -> torch.Tensor:
    """Cross-convolution objective that cancels a trace-invariant source wavelet."""
    n_time = int(prediction.shape[-2])
    n_fft = 1 << int(math.ceil(math.log2(max(2, 2 * n_time - 1))))
    prediction_spectrum = torch.fft.rfft(prediction, n=n_fft, dim=-2, norm="ortho")
    observation_spectrum = torch.fft.rfft(observation, n=n_fft, dim=-2, norm="ortho")
    losses = []
    for index in reference_indices:
        modeled_observed_reference = torch.fft.irfft(
            prediction_spectrum * observation_spectrum[..., index : index + 1],
            n=n_fft,
            dim=-2,
            norm="ortho",
        )[..., : 2 * n_time - 1, :]
        observed_modeled_reference = torch.fft.irfft(
            observation_spectrum * prediction_spectrum[..., index : index + 1],
            n=n_fft,
            dim=-2,
            norm="ortho",
        )[..., : 2 * n_time - 1, :]
        scale = torch.sqrt(
            0.5
            * (
                modeled_observed_reference.detach().square().mean()
                + observed_modeled_reference.detach().square().mean()
            )
        ).clamp_min(1.0e-6)
        residual = (modeled_observed_reference - observed_modeled_reference) / scale
        losses.append(F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.1))
    return torch.stack(losses).mean()


def smooth_frequency_response(response: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return response
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coords = torch.arange(-radius, radius + 1, dtype=response.real.dtype, device=response.device)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    stacked = torch.stack([response.real, response.imag], dim=0).unsqueeze(0)
    stacked = F.pad(stacked, (radius, radius), mode="replicate")
    smoothed = F.conv1d(stacked, kernel.expand(2, 1, -1), groups=2).squeeze(0)
    return torch.complex(smoothed[0], smoothed[1])


def adaptive_source_match(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    *,
    waterlevel: float,
    smoothing_bins: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate one global source filter by variable projection."""
    n_time = int(prediction.shape[-2])
    prediction_spectrum = torch.fft.rfft(prediction, dim=-2)
    observation_spectrum = torch.fft.rfft(observation, dim=-2)
    with torch.no_grad():
        numerator = (
            prediction_spectrum.detach().conj() * observation_spectrum.detach()
        ).sum(dim=(0, 1, 3))
        denominator = prediction_spectrum.detach().abs().square().sum(dim=(0, 1, 3))
        stabilization = float(waterlevel) * denominator.max().clamp_min(1.0e-8)
        response = numerator / (denominator + stabilization)
        response = smooth_frequency_response(response, float(smoothing_bins))
    corrected_spectrum = prediction_spectrum * response.view(1, 1, -1, 1)
    corrected = torch.fft.irfft(corrected_spectrum, n=n_time, dim=-2)
    return corrected, response


def field_time_weight(
    n_time: int,
    dt: float,
    *,
    direct_wave_weight: float,
    direct_wave_end_ns: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    time_ns = torch.arange(n_time, device=device, dtype=dtype) * float(dt) * 1.0e9
    ramp_start = max(0.0, float(direct_wave_end_ns) - 6.0)
    ramp = ((time_ns - ramp_start) / max(float(direct_wave_end_ns) - ramp_start, 1.0)).clamp(0.0, 1.0)
    early = float(direct_wave_weight) + (1.0 - float(direct_wave_weight)) * 0.5 * (
        1.0 - torch.cos(torch.pi * ramp)
    )
    late_start = max(float(time_ns[-1]) - 8.0, float(direct_wave_end_ns) + 1.0)
    late = ((float(time_ns[-1]) - time_ns) / max(float(time_ns[-1]) - late_start, 1.0)).clamp(0.0, 1.0)
    late = torch.where(time_ns <= late_start, torch.ones_like(late), 0.5 * (1.0 - torch.cos(torch.pi * late)))
    return (early * late).view(1, 1, -1, 1)


def regularization(model: torch.Tensor, background: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_x = model[..., :, 1:] - model[..., :, :-1]
    grad_z = model[..., 1:, :] - model[..., :-1, :]
    tv = grad_x.abs().mean() + grad_z.abs().mean()
    laplacian = (
        -4.0 * model[..., 1:-1, 1:-1]
        + model[..., :-2, 1:-1]
        + model[..., 2:, 1:-1]
        + model[..., 1:-1, :-2]
        + model[..., 1:-1, 2:]
    )
    return F.mse_loss(model, background), tv, laplacian.square().mean()


def boundary_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    z = torch.linspace(0.0, 1.0, height, device=device)
    x = torch.linspace(0.0, 1.0, width, device=device)
    top = ((z - 0.04) / 0.06).clamp(0.0, 1.0)
    bottom = ((0.98 - z) / 0.08).clamp(0.0, 1.0)
    left = ((x - 0.015) / 0.04).clamp(0.0, 1.0)
    right = ((0.985 - x) / 0.04).clamp(0.0, 1.0)
    return (top * bottom)[:, None] * (left * right)[None, :]


def suppress_shallow_lateral_anomalies(
    model: torch.Tensor,
    *,
    target_depth_m: float,
    smoothing_depth_m: float,
    strength: float,
) -> torch.Tensor:
    if smoothing_depth_m <= 0.0 or strength <= 0.0:
        return model
    strength = min(max(float(strength), 0.0), 1.0)
    z = torch.linspace(0.0, float(target_depth_m), model.shape[-2], device=model.device, dtype=model.dtype)
    taper_start = 0.75 * float(smoothing_depth_m)
    phase = ((z - taper_start) / max(float(smoothing_depth_m) - taper_start, 1.0e-6)).clamp(0.0, 1.0)
    taper = torch.where(z <= taper_start, torch.ones_like(z), 0.5 * (1.0 + torch.cos(torch.pi * phase)))
    taper = torch.where(z >= float(smoothing_depth_m), torch.zeros_like(z), taper)
    blend = (strength * taper).view(1, 1, -1, 1)
    lateral_background = model.median(dim=-1, keepdim=True).values.expand_as(model)
    return blend * lateral_background + (1.0 - blend) * model


def model_metrics(model: torch.Tensor, background: torch.Tensor) -> dict[str, float]:
    epsilon = 2.0 + 8.0 * model
    grad_x = epsilon[..., :, 1:] - epsilon[..., :, :-1]
    grad_z = epsilon[..., 1:, :] - epsilon[..., :-1, :]
    return {
        "epsilon_min": float(epsilon.min()),
        "epsilon_max": float(epsilon.max()),
        "epsilon_mean": float(epsilon.mean()),
        "epsilon_std": float(epsilon.std()),
        "epsilon_update_rms": float(torch.sqrt(torch.mean((model - background).square())) * 8.0),
        "epsilon_tv": float(grad_x.abs().mean() + grad_z.abs().mean()),
        "high_epsilon_area": float((epsilon > 6.0).float().mean()),
    }


def data_metrics(prediction: torch.Tensor, observation: torch.Tensor) -> dict[str, float]:
    x = prediction.flatten()
    y = observation.flatten()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    pearson = (x_centered * y_centered).sum() / torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    ).clamp_min(1.0e-8)
    return {
        "rmse": float(torch.sqrt(F.mse_loss(prediction, observation))),
        "pearson": float(pearson),
        "trace_ncc": float(1.0 - trace_ncc_loss(prediction, observation)),
        "envelope_rmse": float(torch.sqrt(F.mse_loss(analytic_envelope(prediction), analytic_envelope(observation)))),
    }


def initial_model(
    mode: str,
    initial_epsilon: float,
    neural: torch.Tensor,
    neural_bvi: torch.Tensor,
    control_size: int,
) -> torch.Tensor:
    if mode == "constant":
        full = torch.full_like(neural, (float(initial_epsilon) - 2.0) / 8.0)
    elif mode == "smooth_1d":
        profile = neural.median(dim=-1, keepdim=True).values.expand_as(neural)
        full = gaussian_blur(profile, 10.0)
    elif mode == "neural_bvi":
        full = gaussian_blur(neural_bvi, 4.0)
    else:
        full = gaussian_blur(neural, 4.0)
    return F.interpolate(full, size=(control_size, control_size), mode="area")


def save_figure(
    path: Path,
    *,
    observation: np.ndarray,
    neural: np.ndarray,
    neural_bvi: np.ndarray,
    fwi: np.ndarray,
    bvi_prediction: np.ndarray,
    fwi_prediction: np.ndarray,
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.75), constrained_layout=True)
    model_extent = (0.0, float(profile["model_domain"]["width_m"]), float(profile["model_domain"]["target_depth_m"]), 0.0)
    bscan_extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        float(crop["target_time_window_ns"]),
        0.0,
    )
    obs_limit = max(float(np.quantile(np.abs(observation), 0.995)), 1.0e-6)
    model_images = []
    axes[0, 0].imshow(observation, cmap="gray", vmin=-obs_limit, vmax=obs_limit, extent=bscan_extent, aspect="auto")
    axes[0, 0].set_title("Measured B-scan")
    for axis, array, title in zip(
        axes[0, 1:],
        [neural, neural_bvi, fwi],
        ["Neural estimate", "Neural-BVI mean", "Conventional FWI"],
    ):
        model_images.append(axis.imshow(2.0 + 8.0 * array, cmap="viridis", vmin=2.0, vmax=10.0, extent=model_extent, aspect="auto"))
        axis.set_title(title)
    pred_limit = max(float(np.quantile(np.abs(np.concatenate([bvi_prediction.ravel(), fwi_prediction.ravel()])), 0.995)), 1.0e-6)
    residual_bvi = np.abs(bvi_prediction - observation)
    residual_fwi = np.abs(fwi_prediction - observation)
    residual_limit = max(float(np.quantile(np.concatenate([residual_bvi.ravel(), residual_fwi.ravel()]), 0.995)), 1.0e-6)
    axes[1, 0].imshow(bvi_prediction, cmap="gray", vmin=-pred_limit, vmax=pred_limit, extent=bscan_extent, aspect="auto")
    axes[1, 0].set_title("Neural-BVI reforward")
    axes[1, 1].imshow(fwi_prediction, cmap="gray", vmin=-pred_limit, vmax=pred_limit, extent=bscan_extent, aspect="auto")
    axes[1, 1].set_title("FWI reforward")
    residual_images = [
        axes[1, 2].imshow(residual_bvi, cmap="inferno", vmin=0.0, vmax=residual_limit, extent=bscan_extent, aspect="auto"),
        axes[1, 3].imshow(residual_fwi, cmap="inferno", vmin=0.0, vmax=residual_limit, extent=bscan_extent, aspect="auto"),
    ]
    axes[1, 2].set_title("Neural-BVI residual")
    axes[1, 3].set_title("FWI residual")
    for axis in axes[:, 0]:
        axis.set_ylabel("Time (ns)")
    for axis in axes[0, 1:]:
        axis.set_ylabel("Depth (m)")
    for axis in axes[1, :]:
        axis.set_xlabel("Distance (m)")
    fig.colorbar(model_images[-1], ax=list(axes[0, 1:]), orientation="horizontal", fraction=0.045, pad=0.13, label=r"$\epsilon_r$")
    fig.colorbar(residual_images[-1], ax=list(axes[1, 2:]), orientation="horizontal", fraction=0.045, pad=0.13, label="Absolute residual")
    for label, axis in zip("abcdefgh", axes.ravel()):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontsize=8, fontweight="bold", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    field_dir = (
        args.field_dir.resolve()
        if args.field_dir is not None
        else root / "field_laplace_nn" / "noise_0p080"
    )
    suffix = "quick" if args.quick else "full"
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else root / f"field_fwi_{args.init}_{suffix}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The full-resolution field FWI requires CUDA")

    protocol = json.loads((root / "protocol.snapshot.json").read_text(encoding="utf-8"))
    field_summary = json.loads((field_dir / "summary.json").read_text(encoding="utf-8"))
    arrays = np.load(field_dir / "la010010_full_laplace_arrays.npz")
    observation = torch.from_numpy(arrays["observation"]).float().to(device)
    neural = torch.from_numpy(arrays["nn_prediction"]).float().to(device)
    neural_bvi = torch.from_numpy(arrays["posterior_mean"]).float().to(device)
    forward, forward_summary = load_surrogate_checkpoint(root / "surrogate" / "best.pt", device)
    profile = forward.profile
    dt = float(profile["observation"]["sample_interval_s"])
    neural_bvi_trust_center = suppress_shallow_lateral_anomalies(
        neural_bvi,
        target_depth_m=float(profile["model_domain"]["target_depth_m"]),
        smoothing_depth_m=float(args.shallow_lateral_smoothing_depth_m),
        strength=float(args.shallow_lateral_smoothing_strength),
    ).detach()
    time_weight = field_time_weight(
        int(observation.shape[-2]),
        dt,
        direct_wave_weight=float(args.direct_wave_weight),
        direct_wave_end_ns=float(args.direct_wave_end_ns),
        device=device,
        dtype=observation.dtype,
    )
    shallow_end_index = max(
        1,
        min(
            int(observation.shape[-2]),
            int(round(float(args.shallow_overshoot_end_ns) * 1.0e-9 / dt)),
        ),
    )

    background = torch.full_like(neural, (float(args.initial_epsilon) - 2.0) / 8.0)
    prior_center = {
        "background": background,
        "neural": neural,
        "neural_bvi": neural_bvi_trust_center,
    }[str(args.prior_center)].detach()
    control_initial = initial_model(
        args.init,
        args.initial_epsilon,
        neural,
        neural_bvi_trust_center,
        int(args.control_size),
    )
    epsilon_initial = 2.0 + 8.0 * control_initial
    fraction = ((epsilon_initial - float(args.epsilon_min)) / (float(args.epsilon_max) - float(args.epsilon_min))).clamp(1.0e-4, 1.0 - 1.0e-4)
    parameter = torch.nn.Parameter(torch.logit(fraction))
    boundary_center = background if args.prior_center == "background" else prior_center
    mask = boundary_mask(neural.shape[-2], neural.shape[-1], device)[None, None]
    depth_preconditioner = torch.linspace(0.45, 1.8, int(args.control_size), device=device).view(1, 1, -1, 1)

    def decode(sigma: float) -> torch.Tensor:
        epsilon_control = float(args.epsilon_min) + (float(args.epsilon_max) - float(args.epsilon_min)) * torch.sigmoid(parameter)
        normalized = (epsilon_control - 2.0) / 8.0
        normalized = F.interpolate(normalized, size=neural.shape[-2:], mode="bicubic", align_corners=False)
        normalized = gaussian_blur(normalized, sigma)
        return torch.clamp(boundary_center + mask * (normalized - boundary_center), 0.0, 1.0)

    def match_source(prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if args.objective != "adaptive_source":
            return prediction, None
        return adaptive_source_match(
            prediction,
            observation,
            waterlevel=float(args.source_waterlevel),
            smoothing_bins=float(args.source_smoothing_bins),
        )

    with torch.no_grad():
        baseline_predictions_raw = {
            "neural": forward(neural),
            "neural_bvi": forward(neural_bvi),
            "initial": forward(decode(default_stages(args.quick)[0].smoothing_sigma)),
        }
        baseline_predictions = {
            name: match_source(prediction)[0]
            for name, prediction in baseline_predictions_raw.items()
        }
    baseline_metrics = {
        name: data_metrics(prediction, observation) for name, prediction in baseline_predictions.items()
    }
    baseline_metrics_raw = {
        name: data_metrics(prediction, observation) for name, prediction in baseline_predictions_raw.items()
    }
    full_reference_indices = select_reference_traces(observation, int(args.reference_count))
    baseline_source_independent = {
        name: float(
            source_independent_convolution_loss(prediction, observation, full_reference_indices)
        )
        for name, prediction in baseline_predictions.items()
    }

    stages = default_stages(bool(args.quick))
    history: list[dict[str, Any]] = []
    started = time.time()
    global_step = 0
    for stage_index, stage in enumerate(stages, start=1):
        optimizer = torch.optim.Adam([parameter], lr=float(stage.learning_rate), betas=(0.9, 0.98))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(stage.iterations)),
            eta_min=float(stage.learning_rate) * 0.2,
        )
        observation_filtered = frequency_lowpass(observation, dt, stage.cutoff_mhz * 1.0e6)
        reference_indices = select_reference_traces(observation_filtered, int(args.reference_count))
        best_loss = float("inf")
        best_parameter = parameter.detach().clone()
        for iteration in range(1, int(stage.iterations) + 1):
            global_step += 1
            model = decode(stage.smoothing_sigma)
            prediction_raw = forward(model)
            prediction_filtered_raw = frequency_lowpass(prediction_raw, dt, stage.cutoff_mhz * 1.0e6)
            if args.objective == "adaptive_source":
                prediction_filtered, source_response = adaptive_source_match(
                    prediction_filtered_raw,
                    observation_filtered,
                    waterlevel=float(args.source_waterlevel),
                    smoothing_bins=float(args.source_smoothing_bins),
                )
                prediction_for_loss = prediction_filtered * time_weight
                observation_for_loss = observation_filtered * time_weight
            else:
                prediction_filtered = prediction_filtered_raw
                source_response = None
                prediction_for_loss = prediction_filtered
                observation_for_loss = observation_filtered
            observation_power = observation_for_loss.square().mean().clamp_min(1.0e-6)
            mse = F.smooth_l1_loss(prediction_for_loss, observation_for_loss, beta=0.08) / torch.sqrt(observation_power)
            ncc = trace_ncc_loss(prediction_for_loss, observation_for_loss)
            envelope = F.smooth_l1_loss(
                analytic_envelope(prediction_for_loss),
                analytic_envelope(observation_for_loss),
                beta=0.08,
            ) / torch.sqrt(observation_power)
            prediction_envelope = analytic_envelope(prediction_filtered)
            observation_envelope = analytic_envelope(observation_filtered)
            shallow_overshoot = F.smooth_l1_loss(
                torch.relu(
                    prediction_envelope[..., :shallow_end_index, :]
                    - observation_envelope[..., :shallow_end_index, :]
                ),
                torch.zeros_like(prediction_envelope[..., :shallow_end_index, :]),
                beta=0.08,
            ) / torch.sqrt(
                observation_envelope[..., :shallow_end_index, :].square().mean().clamp_min(1.0e-6)
            )
            source_independent = source_independent_convolution_loss(
                prediction_filtered_raw,
                observation_filtered,
                reference_indices,
            )
            prior, tv, laplacian = regularization(model, prior_center)
            if args.objective == "source_independent":
                data_loss = source_independent
            else:
                data_loss = (
                    stage.mse_weight * mse
                    + stage.ncc_weight * ncc
                    + stage.envelope_weight * envelope
                )
            loss = (
                float(args.data_term_scale) * data_loss
                + float(args.shallow_overshoot_scale) * shallow_overshoot
                + float(args.regularization_scale)
                * (
                    stage.prior_weight * prior
                    + stage.tv_weight * tv
                    + stage.laplacian_weight * laplacian
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Non-finite FWI gradient at stage {stage_index}, iteration {iteration}")
            parameter.grad.mul_(depth_preconditioner)
            torch.nn.utils.clip_grad_norm_([parameter], max_norm=10.0)
            optimizer.step()
            scheduler.step()
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_parameter = parameter.detach().clone()
            record = {
                "global_step": global_step,
                "stage": stage_index,
                "iteration": iteration,
                "cutoff_mhz": stage.cutoff_mhz,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": value,
                "mse_loss": float(mse.detach()),
                "ncc_loss": float(ncc.detach()),
                "envelope_loss": float(envelope.detach()),
                "shallow_envelope_overshoot_loss": float(shallow_overshoot.detach()),
                "source_independent_loss": float(source_independent.detach()),
                "reference_indices": reference_indices,
                "prior_loss": float(prior.detach()),
                "tv_loss": float(tv.detach()),
                "laplacian_loss": float(laplacian.detach()),
                "raw_rmse": float(torch.sqrt(F.mse_loss(prediction_raw, observation)).detach()),
                "raw_pearson": data_metrics(prediction_raw.detach(), observation)["pearson"],
                "matched_rmse": float(torch.sqrt(F.mse_loss(prediction_filtered, observation_filtered)).detach()),
                "matched_pearson": data_metrics(prediction_filtered.detach(), observation_filtered)["pearson"],
                "source_response_rms": (
                    float(torch.sqrt(source_response.abs().square().mean()))
                    if source_response is not None
                    else None
                ),
            }
            history.append(record)
            print(
                f"FWI stage {stage_index}/{len(stages)} iter {iteration:03d}/{stage.iterations}: "
                f"loss={value:.5f} fit_rmse={record['matched_rmse']:.5f} "
                f"corr={record['matched_pearson']:.3f} raw_rmse={record['raw_rmse']:.5f}",
                flush=True,
            )
        parameter.data.copy_(best_parameter)
        torch.save(
            {
                "parameter": parameter.detach().cpu(),
                "stage": asdict(stage),
                "history": history,
            },
            out_dir / f"stage_{stage_index:02d}.pt",
        )

    final_model = decode(stages[-1].smoothing_sigma).detach()
    with torch.no_grad():
        final_prediction_raw = forward(final_model)
        final_prediction, final_source_response = match_source(final_prediction_raw)
    comparison = {
        "initial": baseline_metrics["initial"],
        "neural": baseline_metrics["neural"],
        "neural_bvi": baseline_metrics["neural_bvi"],
        "fwi": data_metrics(final_prediction, observation),
    }
    comparison_raw_fixed_source = {
        **baseline_metrics_raw,
        "fwi": data_metrics(final_prediction_raw, observation),
    }
    comparison_source_independent = {
        **baseline_source_independent,
        "fwi": float(
            source_independent_convolution_loss(
                final_prediction,
                observation,
                full_reference_indices,
            )
        ),
    }
    arrays_path = out_dir / "la010010_field_fwi_arrays.npz"
    np.savez_compressed(
        arrays_path,
        observation=observation.cpu().numpy(),
        neural=neural.cpu().numpy(),
        neural_bvi=neural_bvi.cpu().numpy(),
        prior_center=prior_center.cpu().numpy(),
        fwi=final_model.cpu().numpy(),
        prediction_initial=baseline_predictions["initial"].cpu().numpy(),
        prediction_neural=baseline_predictions["neural"].cpu().numpy(),
        prediction_neural_bvi=baseline_predictions["neural_bvi"].cpu().numpy(),
        prediction_fwi=final_prediction.cpu().numpy(),
        prediction_initial_raw=baseline_predictions_raw["initial"].cpu().numpy(),
        prediction_neural_raw=baseline_predictions_raw["neural"].cpu().numpy(),
        prediction_neural_bvi_raw=baseline_predictions_raw["neural_bvi"].cpu().numpy(),
        prediction_fwi_raw=final_prediction_raw.cpu().numpy(),
        source_response_real=(
            final_source_response.real.cpu().numpy()
            if final_source_response is not None
            else np.empty(0, dtype=np.float32)
        ),
        source_response_imag=(
            final_source_response.imag.cpu().numpy()
            if final_source_response is not None
            else np.empty(0, dtype=np.float32)
        ),
    )
    figure_path = out_dir / "fig_field_la010010_fwi_comparison"
    save_figure(
        figure_path,
        observation=observation.cpu().numpy().squeeze(),
        neural=neural.cpu().numpy().squeeze(),
        neural_bvi=neural_bvi.cpu().numpy().squeeze(),
        fwi=final_model.cpu().numpy().squeeze(),
        bvi_prediction=baseline_predictions["neural_bvi"].cpu().numpy().squeeze(),
        fwi_prediction=final_prediction.cpu().numpy().squeeze(),
        profile=profile,
        crop=field_summary["crop"],
    )
    summary = {
        "status": "complete",
        "method": f"conventional_multiscale_deepwave_fwi_{args.objective}",
        "device": str(device),
        "protocol_hash": protocol["protocol_hash"],
        "field_source_summary": str(field_dir / "summary.json"),
        "forward_backend": forward_summary,
        "config": {
            "initialization": args.init,
            "initial_epsilon": float(args.initial_epsilon),
            "epsilon_bounds": [float(args.epsilon_min), float(args.epsilon_max)],
            "control_size": int(args.control_size),
            "regularization_scale": float(args.regularization_scale),
            "prior_center": str(args.prior_center),
            "objective": args.objective,
            "reference_count": int(args.reference_count),
            "full_band_reference_indices": full_reference_indices,
            "data_term_scale": float(args.data_term_scale),
            "source_waterlevel": float(args.source_waterlevel),
            "source_smoothing_bins": float(args.source_smoothing_bins),
            "direct_wave_weight": float(args.direct_wave_weight),
            "direct_wave_end_ns": float(args.direct_wave_end_ns),
            "shallow_overshoot_scale": float(args.shallow_overshoot_scale),
            "shallow_overshoot_end_ns": float(args.shallow_overshoot_end_ns),
            "shallow_lateral_smoothing_depth_m": float(args.shallow_lateral_smoothing_depth_m),
            "shallow_lateral_smoothing_strength": float(args.shallow_lateral_smoothing_strength),
            "model_size": list(neural.shape[-2:]),
            "stages": [asdict(stage) for stage in stages],
            "boundary_mask": (
                "fixed to the constant background"
                if args.prior_center == "background"
                else f"fixed to the {args.prior_center} trust-region center"
            ),
            "gradient_preconditioner": "linear depth weighting 0.45 to 1.8",
            "optimizer": "Adam with per-stage cosine decay and best-stage rollback",
            "seed": int(args.seed),
        },
        "comparison": comparison,
        "comparison_raw_fixed_source": comparison_raw_fixed_source,
        "comparison_source_independent_loss": comparison_source_independent,
        "model_diagnostics": model_metrics(final_model, background),
        "history": history,
        "seconds": time.time() - started,
        "arrays": str(arrays_path),
        "figure_pdf": str(figure_path.with_suffix(".pdf")),
        "figure_svg": str(figure_path.with_suffix(".svg")),
        "figure_png": str(figure_path.with_suffix(".png")),
        "claim_boundary": (
            "Single measured record with no permittivity truth. FWI is assessed by exact-Deepwave "
            "data consistency, stability, bounds, and structural plausibility; field accuracy is not claimed."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["status", "method", "comparison", "model_diagnostics", "seconds"]}, indent=2))


if __name__ == "__main__":
    main()
