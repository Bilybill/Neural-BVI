"""Joint early-time calibration, 1D shallow inversion, and field Neural-BVI."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

from deepwave_physics_surrogate import DeepwavePhysicsSurrogate, preprocess_bscan
from finalize_hyperbola_conditioned_field_bvi import (
    AdaptiveMatchedForward,
    posterior_predictive_audit,
    time_metrics,
)
from publication_training import load_surrogate_checkpoint, set_seed
from run_deepwave_field_fwi import analytic_envelope, data_metrics
from run_deepwave_field_laplace import read_rd3_pair
from run_deepwave_map_bvi_synthetic import run_map_bvi_case


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_laplace_nn" / "noise_0p080"
CURRENT_DIR = ROOT / "field_shallow_reconciled_bvi"
FWI_DIR = ROOT / "field_fwi_adaptive_source" / "eps5p5_c16_reg500_full"
OUT_DIR = ROOT / "field_joint_early_layered_bvi"
CHECKPOINT = ROOT / "surrogate" / "best.pt"
BASIS = ROOT / "basis" / "error_pca_mean_latent16_train96_snrs0-5-10_seed7.pt"

LAYER_BOUNDARIES_M = (0.0, 0.12, 0.25, 0.40, 0.55, 0.72, 0.90, 2.50)
EPSILON_MIN = 2.5
EPSILON_MAX = 6.8


@dataclass
class Calibration:
    time_zero_ns: float
    source_response: np.ndarray
    coupling: np.ndarray
    surface: np.ndarray
    matched_prediction: np.ndarray
    corrected_observation: np.ndarray
    residual_rms: float
    coupling_energy_fraction: float
    surface_energy_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-bvi", action="store_true")
    parser.add_argument("--reuse-layer-inversion", action="store_true")
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_716)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cosine_window(time_ns: np.ndarray, start_ns: float, end_ns: float, edge_ns: float) -> np.ndarray:
    window = np.zeros_like(time_ns, dtype=np.float64)
    interior = (time_ns >= start_ns) & (time_ns <= end_ns)
    window[interior] = 1.0
    if edge_ns > 0.0:
        rise = (time_ns >= start_ns) & (time_ns < start_ns + edge_ns)
        fall = (time_ns > end_ns - edge_ns) & (time_ns <= end_ns)
        window[rise] = 0.5 * (1.0 - np.cos(np.pi * (time_ns[rise] - start_ns) / edge_ns))
        window[fall] = 0.5 * (1.0 - np.cos(np.pi * (end_ns - time_ns[fall]) / edge_ns))
    return window


def preprocess_raw_window(
    field_summary: dict[str, Any],
    profile_raw: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    raw, _ = read_rd3_pair(Path(field_summary["source"]))
    reference = profile_raw["field_reference"]
    crop = raw[
        int(reference["sample_start"]) : int(reference["sample_end"]),
        int(reference["trace_start"]) : int(reference["trace_end"]),
    ]
    return crop.astype(np.float32), preprocess_bscan(crop, profile_raw)


def weighted_linear_phase_delay(
    response: np.ndarray,
    dt_s: float,
    low_hz: float = 45.0e6,
    high_hz: float = 620.0e6,
) -> float:
    frequency = np.fft.rfftfreq((len(response) - 1) * 2, dt_s)
    amplitude = np.abs(response)
    mask = (
        (frequency >= low_hz)
        & (frequency <= high_hz)
        & (amplitude >= 0.12 * max(float(amplitude.max()), 1.0e-12))
    )
    if int(mask.sum()) < 6:
        return 0.0
    phase = np.unwrap(np.angle(response))
    weight = np.square(amplitude[mask])
    design = np.column_stack([frequency[mask], np.ones(int(mask.sum()))])
    weighted = np.sqrt(weight)
    coefficients = np.linalg.lstsq(
        design * weighted[:, None],
        phase[mask] * weighted,
        rcond=None,
    )[0]
    delay_s = -float(coefficients[0]) / (2.0 * np.pi)
    return float(np.clip(delay_s, -3.0e-9, 8.0e-9))


def estimate_source_response(
    prediction: np.ndarray,
    observation: np.ndarray,
    dt_s: float,
    time_ns: np.ndarray,
) -> tuple[np.ndarray, float]:
    calibration_window = cosine_window(time_ns, 0.0, 15.0, 1.2)[:, None]
    pred_spectrum = np.fft.rfft(prediction * calibration_window, axis=0)
    obs_spectrum = np.fft.rfft(observation * calibration_window, axis=0)
    numerator = np.sum(np.conj(pred_spectrum) * obs_spectrum, axis=1)
    denominator = np.sum(np.abs(pred_spectrum) ** 2, axis=1)
    response = numerator / (denominator + 0.04 * max(float(denominator.max()), 1.0e-12))
    response = scipy.ndimage.gaussian_filter1d(response.real, 2.5) + 1j * scipy.ndimage.gaussian_filter1d(response.imag, 2.5)
    delay_s = weighted_linear_phase_delay(response, dt_s)
    frequency = np.fft.rfftfreq(prediction.shape[0], dt_s)
    zero_delay = response * np.exp(1j * 2.0 * np.pi * frequency * delay_s)
    zero_delay = scipy.ndimage.gaussian_filter1d(zero_delay.real, 2.0) + 1j * scipy.ndimage.gaussian_filter1d(zero_delay.imag, 2.0)
    full_response = zero_delay * np.exp(-1j * 2.0 * np.pi * frequency * delay_s)
    return full_response.astype(np.complex64), delay_s * 1.0e9


def apply_response_numpy(prediction: np.ndarray, response: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(prediction, axis=0)
    return np.fft.irfft(spectrum * response[:, None], n=prediction.shape[0], axis=0).astype(np.float32)


def apply_response_torch(prediction: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft(prediction, dim=-2)
    return torch.fft.irfft(
        spectrum * response.view(1, 1, -1, 1),
        n=prediction.shape[-2],
        dim=-2,
    )


def smooth_rank_one_component(
    residual: np.ndarray,
    time_ns: np.ndarray,
    start_ns: float,
    end_ns: float,
    spatial_sigma: float,
) -> np.ndarray:
    window = cosine_window(time_ns, start_ns, end_ns, min(1.0, 0.25 * (end_ns - start_ns)))
    weighted = residual * window[:, None]
    u, singular, vh = np.linalg.svd(weighted, full_matrices=False)
    spatial = scipy.ndimage.gaussian_filter1d(vh[0], spatial_sigma, mode="nearest")
    norm = float(np.dot(spatial, spatial))
    if norm <= 1.0e-12:
        return np.zeros_like(residual)
    temporal = weighted @ spatial / norm
    component = temporal[:, None] * spatial[None, :]
    component *= window[:, None]
    rank_one_energy = float(singular[0] ** 2 / max(np.square(singular).sum(), 1.0e-12))
    strength = min(1.0, 0.85 / max(rank_one_energy, 1.0e-6))
    return (strength * component).astype(np.float32)


def estimate_calibration(
    prediction: np.ndarray,
    observation: np.ndarray,
    dt_s: float,
    time_ns: np.ndarray,
) -> Calibration:
    response, delay_ns = estimate_source_response(prediction, observation, dt_s, time_ns)
    matched = apply_response_numpy(prediction, response)
    residual = observation - matched
    coupling = smooth_rank_one_component(residual, time_ns, 0.0, 6.0, spatial_sigma=7.0)
    surface = smooth_rank_one_component(residual - coupling, time_ns, 5.5, 12.0, spatial_sigma=10.0)
    corrected = observation - coupling - surface
    early = time_ns <= 15.0
    residual_energy = float(np.mean(np.square(residual[early])))
    denominator = max(residual_energy, 1.0e-12)
    return Calibration(
        time_zero_ns=delay_ns,
        source_response=response,
        coupling=coupling,
        surface=surface,
        matched_prediction=matched,
        corrected_observation=corrected,
        residual_rms=float(np.sqrt(np.mean(np.square((matched - corrected)[early])))),
        coupling_energy_fraction=float(np.mean(np.square(coupling[early])) / denominator),
        surface_energy_fraction=float(np.mean(np.square(surface[early])) / denominator),
    )


class LayeredShallowModel(torch.nn.Module):
    def __init__(self, initial_values: list[float], height: int, width: int, depth_m: float) -> None:
        super().__init__()
        if len(initial_values) != len(LAYER_BOUNDARIES_M) - 1:
            raise ValueError("Initial layer values do not match the fixed boundaries")
        scaled = (np.asarray(initial_values) - EPSILON_MIN) / (EPSILON_MAX - EPSILON_MIN)
        scaled = np.clip(scaled, 1.0e-4, 1.0 - 1.0e-4)
        self.logits = torch.nn.Parameter(torch.from_numpy(np.log(scaled / (1.0 - scaled))).float())
        depth = torch.linspace(0.0, depth_m, height)
        boundaries = torch.tensor(LAYER_BOUNDARIES_M[1:-1])
        self.register_buffer("layer_index", torch.bucketize(depth, boundaries))
        self.width = int(width)

    def layer_values(self) -> torch.Tensor:
        return EPSILON_MIN + (EPSILON_MAX - EPSILON_MIN) * torch.sigmoid(self.logits)

    def forward(self) -> torch.Tensor:
        profile = self.layer_values()[self.layer_index]
        normalized = ((profile - 2.0) / 8.0).clamp(0.0, 1.0)
        return normalized[None, None, :, None].expand(1, 1, -1, self.width)


def stack_trace(data: torch.Tensor) -> torch.Tensor:
    margin = max(4, int(round(0.08 * data.shape[-1])))
    return data[..., margin:-margin].mean(dim=-1)


def layered_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    time_weight: torch.Tensor,
    layer_values: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = stack_trace(prediction)
    obs = stack_trace(observation)
    scale = torch.sqrt(torch.sum(time_weight * obs.square()) / time_weight.sum()).clamp_min(1.0e-4)
    residual = (pred - obs) / scale
    mse = torch.sum(time_weight * residual.square()) / time_weight.sum()
    pred_centered = pred - torch.sum(time_weight * pred) / time_weight.sum()
    obs_centered = obs - torch.sum(time_weight * obs) / time_weight.sum()
    numerator = torch.sum(time_weight * pred_centered * obs_centered)
    denominator = torch.sqrt(
        torch.sum(time_weight * pred_centered.square())
        * torch.sum(time_weight * obs_centered.square())
    ).clamp_min(1.0e-8)
    ncc = 1.0 - numerator / denominator
    pred_envelope = analytic_envelope(prediction).mean(dim=-1)
    obs_envelope = analytic_envelope(observation).mean(dim=-1)
    envelope = torch.sum(time_weight * ((pred_envelope - obs_envelope) / scale).square()) / time_weight.sum()
    smooth = torch.mean((layer_values[1:] - layer_values[:-1]).square())
    curvature = torch.mean((layer_values[2:] - 2.0 * layer_values[1:-1] + layer_values[:-2]).square())
    prior = torch.mean(((layer_values - 5.2) / 0.9).square())
    loss = mse + 0.18 * ncc + 0.10 * envelope + 0.025 * smooth + 0.015 * curvature + 0.020 * prior
    return loss, {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "ncc_loss": float(ncc.detach()),
        "envelope": float(envelope.detach()),
        "smooth": float(smooth.detach()),
        "curvature": float(curvature.detach()),
        "prior": float(prior.detach()),
    }


def optimize_start(
    *,
    name: str,
    initial_values: list[float],
    forward_raw: torch.nn.Module,
    observation_raw: np.ndarray,
    time_ns: np.ndarray,
    dt_s: float,
    profile: dict[str, Any],
    outer_iterations: int,
    steps_per_outer: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Calibration]:
    height, width = map(int, profile["model_shape"])
    model = LayeredShallowModel(initial_values, height, width, float(profile["model_domain"]["target_depth_m"])).to(device)
    observation_tensor = torch.from_numpy(observation_raw)[None, None].float().to(device)
    time_weight_np = cosine_window(time_ns, 0.4, 15.0, 1.0)
    time_weight = torch.from_numpy(time_weight_np)[None, None].float().to(device)
    history: list[dict[str, Any]] = []
    calibration: Calibration | None = None
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    for outer in range(outer_iterations):
        with torch.no_grad():
            raw_prediction = forward_raw(model()).squeeze().cpu().numpy()
        calibration = estimate_calibration(raw_prediction, observation_raw, dt_s, time_ns)
        corrected = torch.from_numpy(calibration.corrected_observation)[None, None].float().to(device)
        response = torch.from_numpy(calibration.source_response).to(device)
        for step in range(steps_per_outer):
            optimizer.zero_grad(set_to_none=True)
            normalized_model = model()
            prediction = apply_response_torch(forward_raw(normalized_model), response)
            loss, diagnostics = layered_loss(
                prediction,
                corrected,
                time_weight,
                model.layer_values(),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            history.append(
                {
                    "outer": outer,
                    "step": step,
                    **diagnostics,
                    "layer_values": [float(value) for value in model.layer_values().detach().cpu()],
                    "time_zero_ns": calibration.time_zero_ns,
                }
            )

    with torch.no_grad():
        normalized_model = model()
        raw_prediction = forward_raw(normalized_model).squeeze().cpu().numpy()
    calibration = estimate_calibration(raw_prediction, observation_raw, dt_s, time_ns)
    final_prediction = calibration.matched_prediction
    corrected = calibration.corrected_observation
    early = time_ns <= 15.0
    pred_stack = final_prediction[early, 10:-10].mean(axis=1)
    obs_stack = corrected[early, 10:-10].mean(axis=1)
    centered_pred = pred_stack - pred_stack.mean()
    centered_obs = obs_stack - obs_stack.mean()
    correlation = float(
        np.dot(centered_pred, centered_obs)
        / max(np.linalg.norm(centered_pred) * np.linalg.norm(centered_obs), 1.0e-12)
    )
    record = {
        "name": name,
        "initial_values": initial_values,
        "layer_values": [float(value) for value in model.layer_values().detach().cpu()],
        "time_zero_ns": calibration.time_zero_ns,
        "corrected_0_15ns_rmse": float(np.sqrt(np.mean(np.square(pred_stack - obs_stack)))),
        "corrected_0_15ns_pearson": correlation,
        "coupling_energy_fraction": calibration.coupling_energy_fraction,
        "surface_energy_fraction": calibration.surface_energy_fraction,
        "history": history,
    }
    return record, normalized_model.detach().cpu().numpy(), final_prediction, calibration


def insert_layered_prior(
    current_model: np.ndarray,
    layer_values: np.ndarray,
    depth_m: float,
    *,
    strength: float,
    taper_end_m: float,
) -> np.ndarray:
    height = current_model.shape[-2]
    depth = np.linspace(0.0, depth_m, height)
    index = np.searchsorted(np.asarray(LAYER_BOUNDARIES_M[1:-1]), depth, side="right")
    profile_epsilon = layer_values[index]
    current_epsilon = 2.0 + 8.0 * current_model.squeeze()
    taper_start_m = 0.50
    weight = np.zeros_like(depth)
    weight[depth <= taper_start_m] = float(strength)
    transition = (depth > taper_start_m) & (depth < taper_end_m)
    weight[transition] = float(strength) * 0.5 * (
        1.0 + np.cos(np.pi * (depth[transition] - taper_start_m) / (taper_end_m - taper_start_m))
    )
    prior_epsilon = current_epsilon * (1.0 - weight[:, None]) + profile_epsilon[:, None] * weight[:, None]
    return np.clip((prior_epsilon - 2.0) / 8.0, 0.0, 1.0)[None, None].astype(np.float32)


def fwi_gate(candidate: dict[str, float], fwi: dict[str, float]) -> bool:
    return (
        candidate["rmse"] < fwi["rmse"]
        and candidate["pearson"] > fwi["pearson"]
        and candidate["trace_ncc"] > fwi["trace_ncc"]
        and candidate["envelope_rmse"] < fwi["envelope_rmse"]
        and candidate["rmse_0_12ns"] < fwi["rmse_0_12ns"]
        and candidate["rmse_12_20ns"] < fwi["rmse_12_20ns"]
        and candidate["rmse_0_20ns"] < fwi["rmse_0_20ns"]
        and candidate["rmse_20_30ns"] < fwi["rmse_20_30ns"]
    )


def metric_payload(prediction: np.ndarray, observation: np.ndarray, dt_ns: float) -> dict[str, float]:
    pred = torch.from_numpy(prediction)[None, None].float()
    obs = torch.from_numpy(observation)[None, None].float()
    metrics = {key: float(value) for key, value in data_metrics(pred, obs).items()}
    metrics.update(time_metrics(prediction, observation, dt_ns))
    return metrics


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_diagnostic_figure(
    out_dir: Path,
    arrays: dict[str, np.ndarray],
    profile: dict[str, Any],
    crop: dict[str, Any],
) -> None:
    configure_matplotlib()
    raw = arrays["raw_observation"].squeeze()
    matched = arrays["raw_matched_prediction"].squeeze()
    corrected = arrays["raw_corrected_observation"].squeeze()
    coupling = arrays["antenna_coupling"].squeeze()
    surface = arrays["surface_reflection"].squeeze()
    residual = np.abs(matched - corrected)
    prior = 2.0 + 8.0 * arrays["layered_prior"].squeeze()
    current = 2.0 + 8.0 * arrays["current_posterior_mean"].squeeze()
    posterior = 2.0 + 8.0 * arrays.get("posterior_mean", arrays["layered_prior"]).squeeze()
    dt_ns = float(profile["observation"]["sample_interval_s"]) * 1.0e9
    stop = min(raw.shape[0], int(round(15.0 / dt_ns)))
    extent = (
        float(crop["start_distance_m"]),
        float(crop["start_distance_m"]) + float(crop["target_aperture_m"]),
        15.0,
        0.0,
    )
    model_extent = (0.0, float(profile["model_domain"]["width_m"]), float(profile["model_domain"]["target_depth_m"]), 0.0)
    amp_limit = max(float(np.quantile(np.abs(np.concatenate([raw[:stop].ravel(), matched[:stop].ravel(), corrected[:stop].ravel()])), 0.995)), 1.0e-6)
    nuisance_limit = max(float(np.quantile(np.abs(np.concatenate([coupling[:stop].ravel(), surface[:stop].ravel()])), 0.995)), 1.0e-6)
    residual_limit = max(float(np.quantile(residual[:stop], 0.995)), 1.0e-6)
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 6.3), constrained_layout=True)
    for axis, image, title in zip(
        axes[0],
        (raw[:stop], matched[:stop], corrected[:stop]),
        ("Raw early B-scan", "Source/time-zero matched", "Nuisance-corrected data"),
    ):
        axis.imshow(image, cmap="gray", vmin=-amp_limit, vmax=amp_limit, extent=extent, aspect="auto")
        axis.set_title(title)
    for axis, image, title in zip(
        axes[1],
        (coupling[:stop], surface[:stop], residual[:stop]),
        ("Antenna coupling", "Surface reflection", "Corrected absolute residual"),
    ):
        if title == "Corrected absolute residual":
            axis.imshow(image, cmap="inferno", vmin=0.0, vmax=residual_limit, extent=extent, aspect="auto")
        else:
            axis.imshow(image, cmap="coolwarm", vmin=-nuisance_limit, vmax=nuisance_limit, extent=extent, aspect="auto")
        axis.set_title(title)
    depth = np.linspace(0.0, float(profile["model_domain"]["target_depth_m"]), prior.shape[0])
    shallow = depth <= 1.0
    axes[2, 0].plot(current.mean(axis=1)[shallow], depth[shallow], color="0.5", label="Current")
    axes[2, 0].plot(prior.mean(axis=1)[shallow], depth[shallow], color="#0072B2", label="Layered prior")
    axes[2, 0].plot(posterior.mean(axis=1)[shallow], depth[shallow], color="#D55E00", label="Posterior")
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlabel(r"$\epsilon_r$")
    axes[2, 0].set_ylabel("Depth (m)")
    axes[2, 0].set_title("Shallow 1D profile")
    axes[2, 0].legend(fontsize=5.2)
    for axis, image, title in zip(
        axes[2, 1:],
        (prior, posterior),
        ("Layered Neural-BVI prior", "Neural-BVI posterior mean"),
    ):
        axis.imshow(image, cmap="viridis", vmin=2.2, vmax=6.2, extent=model_extent, aspect="auto")
        axis.set_title(title)
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Depth (m)")
    for axis in axes[:2].ravel():
        axis.set_xlabel("Distance (m)")
        axis.set_ylabel("Time (ns)")
    for label, axis in zip("abcdefghi", axes.ravel()):
        axis.text(-0.14, 1.03, label, transform=axis.transAxes, fontsize=8, fontweight="bold")
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 500})):
        fig.savefig(out_dir / f"fig_field_joint_early_layered_bvi.{suffix}", bbox_inches="tight", pad_inches=0.02, **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Joint early-time layered inversion requires CUDA")
    set_seed(int(args.seed))

    field_summary = load_json(FIELD_DIR / "summary.json")
    profile = field_summary["forward_backend"]["acquisition_profile"]
    profile_raw = copy.deepcopy(profile)
    profile_raw["preprocessing"]["background_subtraction"] = False
    raw_counts, raw_observation = preprocess_raw_window(field_summary, profile_raw)
    del raw_counts
    dt_s = float(profile["observation"]["sample_interval_s"])
    dt_ns = dt_s * 1.0e9
    time_ns = np.arange(raw_observation.shape[0]) * dt_ns

    physics = load_json(ROOT / "surrogate" / "summary.json")["physics_config"]
    forward_raw = DeepwavePhysicsSurrogate(
        profile=profile_raw,
        shot_batch_size=int(physics["shot_batch_size"]),
        pml_width=int(physics["pml_width"]),
        epsilon_min=float(physics["epsilon_min"]),
        epsilon_max=float(physics["epsilon_max"]),
    ).to(device)
    starts = {
        "velocity_uniform": [5.5] * 7,
        "moderate_uniform": [5.0] * 7,
        "mild_layered": [5.4, 5.2, 4.8, 5.3, 4.7, 5.1, 5.0],
    }
    if args.quick:
        starts = {"velocity_uniform": starts["velocity_uniform"]}
    outer_iterations = 1 if args.quick else 2
    steps_per_outer = 12 if args.quick else 24
    records = []
    payloads = []
    if args.reuse_layer_inversion:
        existing_summary = load_json(out_dir / "summary.json")
        existing_arrays = np.load(out_dir / "field_joint_early_layered_bvi_arrays.npz")
        records = existing_summary["starts"]
        selected_record = next(
            record for record in records if record["name"] == existing_summary["selected_start"]
        )
        layered_model = existing_arrays["layered_1d_model"].copy()
        calibration = Calibration(
            time_zero_ns=float(existing_summary["nuisance"]["time_zero_ns"]),
            source_response=(
                existing_arrays["source_response_real"]
                + 1j * existing_arrays["source_response_imag"]
            ).astype(np.complex64),
            coupling=existing_arrays["antenna_coupling"].squeeze().copy(),
            surface=existing_arrays["surface_reflection"].squeeze().copy(),
            matched_prediction=existing_arrays["raw_matched_prediction"].squeeze().copy(),
            corrected_observation=existing_arrays["raw_corrected_observation"].squeeze().copy(),
            residual_rms=float(existing_summary["nuisance"]["corrected_residual_rms"]),
            coupling_energy_fraction=float(existing_summary["nuisance"]["coupling_energy_fraction"]),
            surface_energy_fraction=float(existing_summary["nuisance"]["surface_energy_fraction"]),
        )
        existing_arrays.close()
    else:
        for name, initial_values in starts.items():
            record, model, prediction, calibration = optimize_start(
                name=name,
                initial_values=initial_values,
                forward_raw=forward_raw,
                observation_raw=raw_observation,
                time_ns=time_ns,
                dt_s=dt_s,
                profile=profile,
                outer_iterations=outer_iterations,
                steps_per_outer=steps_per_outer,
                device=device,
            )
            records.append(record)
            payloads.append((model, prediction, calibration))
        eligible = [
            (index, record)
            for index, record in enumerate(records)
            if max(record["layer_values"]) - min(record["layer_values"]) <= 2.5
            and 3.0 <= float(np.mean(record["layer_values"][:4])) <= 6.3
        ]
        selected_index, selected_record = min(
            eligible,
            key=lambda item: (item[1]["corrected_0_15ns_rmse"], -item[1]["corrected_0_15ns_pearson"]),
        )
        layered_model, _raw_prediction, calibration = payloads[selected_index]
    layer_values = np.asarray(selected_record["layer_values"], dtype=np.float32)

    current_arrays = np.load(CURRENT_DIR / "shallow_reconciled_bvi_arrays.npz")
    current_mean = current_arrays["posterior_mean"].astype(np.float32)
    observation = torch.from_numpy(current_arrays["observation"]).float()
    forward_processed, _ = load_surrogate_checkpoint(CHECKPOINT, device)
    matched_forward = AdaptiveMatchedForward(forward_processed, observation.to(device))
    observation_np = current_arrays["observation"].squeeze().astype(np.float64)
    fwi_metrics = metric_payload(
        current_arrays["fwi_prediction"].squeeze().astype(np.float64),
        observation_np,
        dt_ns,
    )
    prior_candidates: list[dict[str, Any]] = []
    prior_payloads: list[tuple[np.ndarray, np.ndarray]] = []
    with torch.no_grad():
        for taper_end_m in (0.55, 0.65, 0.72, 0.85):
            for strength in (0.25, 0.50, 0.75, 1.00):
                candidate = insert_layered_prior(
                    current_mean,
                    layer_values,
                    float(profile["model_domain"]["target_depth_m"]),
                    strength=strength,
                    taper_end_m=taper_end_m,
                )
                candidate_prediction = matched_forward(torch.from_numpy(candidate).to(device)).cpu().numpy()
                candidate_metrics = metric_payload(
                    candidate_prediction.squeeze().astype(np.float64),
                    observation_np,
                    dt_ns,
                )
                prior_candidates.append(
                    {
                        "strength": strength,
                        "taper_end_m": taper_end_m,
                        "passes_fwi_gate": fwi_gate(candidate_metrics, fwi_metrics),
                        "metrics": candidate_metrics,
                    }
                )
                prior_payloads.append((candidate, candidate_prediction))
    eligible_priors = [
        (index, record)
        for index, record in enumerate(prior_candidates)
        if record["passes_fwi_gate"]
        and record["metrics"]["trace_ncc"] >= fwi_metrics["trace_ncc"] + 0.005
        and record["metrics"]["pearson"] >= fwi_metrics["pearson"] + 0.005
    ]
    if not eligible_priors:
        raise RuntimeError("No layered-prior candidate passes the conventional-FWI data gate")
    selected_prior_index, selected_prior = max(
        eligible_priors,
        key=lambda item: (
            item[1]["strength"],
            item[1]["taper_end_m"],
            item[1]["metrics"]["pearson"],
        ),
    )
    layered_prior, layered_prior_prediction = prior_payloads[selected_prior_index]

    arrays: dict[str, np.ndarray] = {
        "raw_observation": raw_observation[None, None],
        "raw_matched_prediction": calibration.matched_prediction[None, None],
        "raw_corrected_observation": calibration.corrected_observation[None, None],
        "antenna_coupling": calibration.coupling[None, None],
        "surface_reflection": calibration.surface[None, None],
        "source_response_real": calibration.source_response.real,
        "source_response_imag": calibration.source_response.imag,
        "layer_values_epsilon": layer_values,
        "layer_boundaries_m": np.asarray(LAYER_BOUNDARIES_M, dtype=np.float32),
        "layered_1d_model": layered_model,
        "layered_prior": layered_prior,
        "layered_prior_prediction": layered_prior_prediction,
        "observation": current_arrays["observation"],
        "current_posterior_mean": current_mean,
        "current_prediction": current_arrays["prediction"],
        "fwi_model": current_arrays["fwi_model"],
        "fwi_prediction": current_arrays["fwi_prediction"],
    }
    bvi_result: dict[str, Any] | None = None
    posterior_predictive: dict[str, Any] | None = None
    if not args.skip_bvi:
        basis_payload = torch.load(BASIS, map_location="cpu", weights_only=False)
        set_seed(int(args.seed) + 17)
        result, samples, _ = run_map_bvi_case(
            matched_forward,
            observation,
            None,
            None,
            torch.from_numpy(layered_prior),
            latent_dim=16,
            residual_scale=0.003,
            model_prior_std=0.005,
            latent_prior_weight=0.5,
            steps=8,
            lr=0.03,
            basis_type="error_pca_mean",
            posterior_latent_std=0.03,
            posterior_sample_count=int(args.posterior_samples),
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
        posterior_predictive = posterior_predictive_audit(
            samples,
            matched_forward,
            observation.to(device),
        )
        arrays.update(
            {
                "posterior_samples": samples.numpy(),
                "posterior_mean": posterior_mean.numpy(),
                "posterior_std": posterior_std.numpy(),
                "prediction": prediction.numpy(),
            }
        )
        bvi_result = result.__dict__

    layered_prior_metrics = metric_payload(
        arrays["layered_prior_prediction"].squeeze().astype(np.float64),
        observation_np,
        dt_ns,
    )
    current_metrics = metric_payload(
        arrays["current_prediction"].squeeze().astype(np.float64),
        observation_np,
        dt_ns,
    )
    posterior_metrics = (
        metric_payload(arrays["prediction"].squeeze().astype(np.float64), observation_np, dt_ns)
        if "prediction" in arrays
        else None
    )
    np.savez_compressed(out_dir / "field_joint_early_layered_bvi_arrays.npz", **arrays)
    save_diagnostic_figure(out_dir, arrays, profile, field_summary["crop"])
    summary = {
        "status": "complete",
        "seconds": time.time() - started,
        "config": {
            "raw_preprocessing": "DC removal + 32-620 MHz Butterworth + 99.5th-percentile scaling; no lateral background subtraction",
            "calibration_window_ns": [0.0, 15.0],
            "antenna_coupling_window_ns": [0.0, 6.0],
            "surface_reflection_window_ns": [5.5, 12.0],
            "layer_boundaries_m": list(LAYER_BOUNDARIES_M),
            "outer_iterations": outer_iterations,
            "steps_per_outer": steps_per_outer,
            "posterior_samples": 0 if args.skip_bvi else int(args.posterior_samples),
        },
        "nuisance": {
            "time_zero_ns": calibration.time_zero_ns,
            "coupling_energy_fraction": calibration.coupling_energy_fraction,
            "surface_energy_fraction": calibration.surface_energy_fraction,
            "corrected_residual_rms": calibration.residual_rms,
        },
        "starts": records,
        "selected_start": selected_record["name"],
        "selected_layer_values_epsilon": selected_record["layer_values"],
        "layered_prior_candidates": prior_candidates,
        "selected_layered_prior": {
            "strength": selected_prior["strength"],
            "taper_end_m": selected_prior["taper_end_m"],
            "selection_rule": "maximum 1D-profile strength and depth support among candidates passing all symmetric FWI gates with 0.005 Pearson/trace-NCC safety margins",
        },
        "layered_prior_metrics": layered_prior_metrics,
        "posterior_metrics": posterior_metrics,
        "current_metrics": current_metrics,
        "fwi_metrics": fwi_metrics,
        "bvi_result": bvi_result,
        "posterior_predictive": posterior_predictive,
        "current_result_source": str(CURRENT_DIR / "shallow_reconciled_bvi_arrays.npz"),
        "claim_boundary": (
            "The nuisance separation and 1D layering are conditional on a zero-offset scalar Deepwave model, "
            "fixed temporal windows, and one measured record. They do not uniquely separate antenna, surface, "
            "and shallow-medium effects without calibration data or multi-offset observations."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "complete",
                "time_zero_ns": calibration.time_zero_ns,
                "layer_values": selected_record["layer_values"],
                "selected_layered_prior": {
                    "strength": selected_prior["strength"],
                    "taper_end_m": selected_prior["taper_end_m"],
                },
                "layered_prior_metrics": layered_prior_metrics,
                "posterior_metrics": posterior_metrics,
                "current_metrics": current_metrics,
                "fwi_metrics": fwi_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
