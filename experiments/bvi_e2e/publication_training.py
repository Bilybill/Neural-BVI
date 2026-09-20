"""Training and deterministic evaluation for the publication protocol."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepwave_physics_surrogate import DeepwavePhysicsSurrogate
from inversion_models import BaseConditionedResidualRefiner, build_inversion_model, build_waveform_surrogate
from publication_data import DynamicNoiseDataset
from publication_metrics import deterministic_metrics, normalized_data_rmse, pearson_flat


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    py = prediction[..., 1:, :] - prediction[..., :-1, :]
    ty = target[..., 1:, :] - target[..., :-1, :]
    px = prediction[..., :, 1:] - prediction[..., :, :-1]
    tx = target[..., :, 1:] - target[..., :, :-1]
    return F.l1_loss(py, ty) + F.l1_loss(px, tx)


def predict_batches(model: torch.nn.Module, data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            outputs.append(model(data[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


class WaveformSurrogateEnsemble(torch.nn.Module):
    def __init__(self, members: list[torch.nn.Module], weights: list[float] | torch.Tensor | None = None):
        super().__init__()
        if not members:
            raise ValueError("WaveformSurrogateEnsemble requires at least one member")
        self.members = torch.nn.ModuleList(members)
        if weights is None:
            weights_tensor = torch.full((len(members),), 1.0 / float(len(members)), dtype=torch.float32)
        else:
            weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
            if weights_tensor.numel() != len(members):
                raise ValueError("Ensemble weights must match the number of members")
            weights_tensor = weights_tensor / weights_tensor.sum().clamp_min(1.0e-8)
        self.register_buffer("weights", weights_tensor.view(-1))

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        prediction = self.weights[0] * self.members[0](model)
        for weight, member in zip(self.weights[1:], self.members[1:]):
            prediction = prediction + weight * member(model)
        return prediction


class OutputCalibratedSurrogate(torch.nn.Module):
    """Apply train-fitted output calibration stored in the surrogate checkpoint."""

    def __init__(self, base: torch.nn.Module, calibration: dict[str, Any]):
        super().__init__()
        self.base = base
        self.calibration = dict(calibration)
        scale = calibration.get("scale")
        bias = calibration.get("bias")
        self.register_buffer(
            "scale",
            torch.as_tensor(scale, dtype=torch.float32).view(1, 1, -1, 1) if scale is not None else None,
        )
        self.register_buffer(
            "bias",
            torch.as_tensor(bias, dtype=torch.float32).view(1, 1, -1, 1) if bias is not None else None,
        )

    @property
    def members(self) -> Any:
        return getattr(self.base, "members", None)

    @property
    def weights(self) -> Any:
        return getattr(self.base, "weights", None)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        prediction = self.base(model)
        if self.scale is not None:
            prediction = prediction * self.scale.to(device=prediction.device, dtype=prediction.dtype)
        if self.bias is not None:
            prediction = prediction + self.bias.to(device=prediction.device, dtype=prediction.dtype)
        return prediction


class ResidualRefinedSurrogate(torch.nn.Module):
    """Frozen base surrogate plus a train-fitted residual correction network."""

    def __init__(self, base: torch.nn.Module, refiner: BaseConditionedResidualRefiner, residual_alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.refiner = refiner
        self.residual_alpha = float(residual_alpha)

    @property
    def members(self) -> Any:
        return getattr(self.base, "members", None)

    @property
    def weights(self) -> Any:
        return getattr(self.base, "weights", None)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        base_prediction = self.base(model)
        correction = self.refiner(model, base_prediction)
        return base_prediction + self.residual_alpha * correction


def scheduler_lambda(epoch: int, epochs: int, warmup: int) -> float:
    if epoch < warmup:
        return float(epoch + 1) / max(float(warmup), 1.0)
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def surrogate_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    high_weight: float = 0.15,
    low_weight: float = 0.10,
    gradient_weight: float = 0.01,
    spectral_weight: float = 0.0005,
) -> torch.Tensor:
    error2 = (prediction - target) ** 2
    global_mse = error2.mean()
    threshold = torch.quantile(target.detach().abs().flatten(1), 0.90, dim=1).view(-1, 1, 1, 1)
    high = (target.detach().abs() > threshold).float()
    low = 1.0 - high
    high_mse = (error2 * high).sum() / high.sum().clamp_min(1.0)
    low_mse = (error2 * low).sum() / low.sum().clamp_min(1.0)
    spectral = F.l1_loss(
        torch.abs(torch.fft.rfft(prediction, dim=-2)),
        torch.abs(torch.fft.rfft(target, dim=-2)),
    )
    return (
        (1.0 - high_weight - low_weight) * global_mse
        + high_weight * high_mse
        + low_weight * low_mse
        + gradient_weight * gradient_loss(prediction, target)
        + spectral_weight * spectral
    )


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    truth = truth - truth.mean(dim=1, keepdim=True)
    numerator = torch.sum(pred * truth, dim=1)
    denominator = torch.sqrt(torch.sum(pred**2, dim=1) * torch.sum(truth**2, dim=1)).clamp_min(1.0e-8)
    return torch.mean(1.0 - numerator / denominator)


def sample_normalized_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    power = torch.mean(target.detach() ** 2, dim=(1, 2, 3)).clamp_min(1.0e-8)
    return torch.mean(error / power)


def window_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: tuple[tuple[int, int], ...] = ((0, 20), (20, 50), (50, 90), (90, 140), (140, 179)),
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.15, 0.10),
    power_floor_fraction: float = 0.02,
) -> torch.Tensor:
    target_power = torch.mean(target.detach() ** 2).clamp_min(1.0e-8)
    floor = target_power * float(power_floor_fraction)
    total = target.new_tensor(0.0)
    norm = target.new_tensor(0.0)
    n_time = target.shape[-2]
    for (start, end), weight in zip(windows, weights):
        lo = max(0, min(int(start), n_time))
        hi = max(lo + 1, min(int(end), n_time))
        error = torch.mean((prediction[..., lo:hi, :] - target[..., lo:hi, :]) ** 2)
        power = torch.mean(target.detach()[..., lo:hi, :] ** 2).clamp_min(float(floor))
        total = total + float(weight) * error / power
        norm = norm + float(weight)
    return total / norm.clamp_min(1.0e-8)


def envelope_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_time = torch.sqrt(torch.mean(prediction.float() ** 2, dim=(0, 1, 3)) + 1.0e-8)
    target_time = torch.sqrt(torch.mean(target.float() ** 2, dim=(0, 1, 3)) + 1.0e-8)
    pred_trace = torch.sqrt(torch.mean(prediction.float() ** 2, dim=(0, 1, 2)) + 1.0e-8)
    target_trace = torch.sqrt(torch.mean(target.float() ** 2, dim=(0, 1, 2)) + 1.0e-8)
    return F.l1_loss(pred_time, target_time) + F.l1_loss(pred_trace, target_trace)


def fk_spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.log1p(torch.abs(torch.fft.rfft2(prediction.float(), dim=(-2, -1))))
    target_mag = torch.log1p(torch.abs(torch.fft.rfft2(target.float(), dim=(-2, -1))))
    return F.l1_loss(pred_mag, target_mag)


def late_window_underpower_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: tuple[tuple[int, int], ...] = ((50, 90), (90, 140), (140, 179)),
    weights: tuple[float, ...] = (0.45, 0.40, 0.15),
    rms_floor_fraction: float = 0.03,
) -> torch.Tensor:
    """Penalize late-window amplitude collapse without rewarding over-amplification."""

    global_rms = torch.sqrt(torch.mean(target.detach().float() ** 2)).clamp_min(1.0e-8)
    floor = float(rms_floor_fraction) * global_rms
    total = target.new_tensor(0.0)
    norm = target.new_tensor(0.0)
    n_time = target.shape[-2]
    for (start, end), weight in zip(windows, weights):
        lo = max(0, min(int(start), n_time - 1))
        hi = max(lo + 1, min(int(end), n_time))
        pred_rms = torch.sqrt(torch.mean(prediction[..., lo:hi, :].float() ** 2, dim=(1, 2, 3)) + floor**2)
        target_rms = torch.sqrt(torch.mean(target.detach()[..., lo:hi, :].float() ** 2, dim=(1, 2, 3)) + floor**2)
        underpower = F.relu(torch.log(target_rms.clamp_min(1.0e-8)) - torch.log(pred_rms.clamp_min(1.0e-8)))
        total = total + float(weight) * torch.mean(underpower**2)
        norm = norm + float(weight)
    return total / norm.clamp_min(1.0e-8)


def late_window_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: tuple[tuple[int, int], ...] = ((50, 90), (90, 140), (140, 179)),
    weights: tuple[float, ...] = (0.50, 0.35, 0.15),
) -> torch.Tensor:
    total = target.new_tensor(0.0)
    norm = target.new_tensor(0.0)
    n_time = target.shape[-2]
    for (start, end), weight in zip(windows, weights):
        lo = max(0, min(int(start), n_time - 1))
        hi = max(lo + 1, min(int(end), n_time))
        total = total + float(weight) * correlation_loss(prediction[..., lo:hi, :], target[..., lo:hi, :])
        norm = norm + float(weight)
    return total / norm.clamp_min(1.0e-8)


def late_time_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    start_fraction: float = 0.28,
    max_extra_weight: float = 2.0,
) -> torch.Tensor:
    n_time = target.shape[-2]
    t = torch.linspace(0.0, 1.0, n_time, device=target.device, dtype=target.dtype).view(1, 1, n_time, 1)
    ramp = ((t - float(start_fraction)) / max(1.0 - float(start_fraction), 1.0e-6)).clamp(0.0, 1.0)
    weight = 1.0 + float(max_extra_weight) * ramp
    return torch.mean((prediction - target) ** 2 * weight) / weight.mean().clamp_min(1.0e-8)


def time_gain_domain_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    start_fraction: float = 0.18,
    max_gain: float = 5.0,
    power: float = 1.4,
) -> torch.Tensor:
    n_time = target.shape[-2]
    t = torch.linspace(0.0, 1.0, n_time, device=target.device, dtype=target.dtype).view(1, 1, n_time, 1)
    ramp = ((t - float(start_fraction)) / max(1.0 - float(start_fraction), 1.0e-6)).clamp(0.0, 1.0)
    gain = 1.0 + (float(max_gain) - 1.0) * torch.pow(ramp, float(power))
    error = (prediction - target) * gain
    return torch.mean(error**2) / torch.mean(gain**2).clamp_min(1.0e-8)


def per_sample_window_relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: tuple[tuple[int, int], ...] = ((0, 20), (20, 50), (50, 90), (90, 140), (140, 179)),
    weights: tuple[float, ...] = (0.10, 0.20, 0.40, 0.22, 0.08),
    power_floor_fraction: float = 0.04,
) -> torch.Tensor:
    sample_power = torch.mean(target.detach().float() ** 2, dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-8)
    total = target.new_tensor(0.0)
    norm = target.new_tensor(0.0)
    n_time = target.shape[-2]
    for (start, end), weight in zip(windows, weights):
        lo = max(0, min(int(start), n_time - 1))
        hi = max(lo + 1, min(int(end), n_time))
        error = torch.mean((prediction[..., lo:hi, :] - target[..., lo:hi, :]) ** 2, dim=(1, 2, 3))
        power = torch.mean(target.detach()[..., lo:hi, :].float() ** 2, dim=(1, 2, 3))
        floor = sample_power.flatten() * float(power_floor_fraction)
        total = total + float(weight) * torch.mean(error / power.clamp_min(floor))
        norm = norm + float(weight)
    return total / norm.clamp_min(1.0e-8)


def arrival_prior_from_model(
    model: torch.Tensor,
    output_shape: tuple[int, int],
    blur_time: int = 7,
    blur_trace: int = 3,
) -> torch.Tensor:
    """Approximate reflection-arrival prior from vertical slowness contrasts."""

    with torch.no_grad():
        resized = F.interpolate(model.detach().float().clamp(0.0, 1.0), size=output_shape, mode="bilinear", align_corners=False)
        slowness = torch.sqrt(1.0 + 8.0 * resized)
        reflectivity = F.pad((slowness[..., 1:, :] - slowness[..., :-1, :]).abs(), (0, 0, 0, 1))
        travel_time = torch.cumsum(slowness, dim=-2)
        travel_time = travel_time / travel_time.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
        n_time = output_shape[0]
        arrival_index = (travel_time * float(n_time - 1)).round().long().clamp(0, n_time - 1)
        prior = torch.zeros_like(reflectivity)
        prior.scatter_add_(2, arrival_index, reflectivity)
        if blur_time > 1 or blur_trace > 1:
            prior = F.avg_pool2d(
                prior,
                kernel_size=(int(blur_time), int(blur_trace)),
                stride=1,
                padding=(int(blur_time) // 2, int(blur_trace) // 2),
            )
            prior = prior[..., : output_shape[0], : output_shape[1]]
        prior = prior / prior.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    return prior.to(device=model.device, dtype=model.dtype)


def arrival_prior_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    model: torch.Tensor | None,
    prior_strength: float = 1.0,
    late_start_fraction: float = 0.25,
) -> torch.Tensor:
    if model is None:
        return prediction.new_tensor(0.0)
    prior = arrival_prior_from_model(model, target.shape[-2:])
    n_time = target.shape[-2]
    t = torch.linspace(0.0, 1.0, n_time, device=target.device, dtype=target.dtype).view(1, 1, n_time, 1)
    late_ramp = ((t - float(late_start_fraction)) / max(1.0 - float(late_start_fraction), 1.0e-6)).clamp(0.0, 1.0)
    weight = 1.0 + float(prior_strength) * prior * late_ramp
    return torch.mean((prediction - target) ** 2 * weight) / weight.mean().clamp_min(1.0e-8)


def surrogate_physics_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    high_weight: float = 0.15,
    low_weight: float = 0.10,
    gradient_weight: float = 0.01,
    spectral_weight: float = 0.0005,
    sample_nrmse_weight: float = 0.02,
    window_relative_weight: float = 0.005,
    correlation_weight: float = 0.02,
    envelope_weight: float = 0.01,
    fk_spectral_weight: float = 0.002,
    window_power_floor_fraction: float = 0.02,
) -> torch.Tensor:
    """Balanced waveform loss with data-domain wave-physics regularizers."""

    base = surrogate_balanced_loss(
        prediction,
        target,
        high_weight=high_weight,
        low_weight=low_weight,
        gradient_weight=gradient_weight,
        spectral_weight=spectral_weight,
    )
    return (
        base
        + sample_nrmse_weight * sample_normalized_mse(prediction, target)
        + window_relative_weight
        * window_relative_mse(prediction, target, power_floor_fraction=window_power_floor_fraction)
        + correlation_weight * correlation_loss(prediction, target)
        + envelope_weight * envelope_loss(prediction, target)
        + fk_spectral_weight * fk_spectral_loss(prediction, target)
    )


def surrogate_late_tail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: dict[str, Any],
    model: torch.Tensor | None = None,
) -> torch.Tensor:
    """Light fine-tuning loss for the low-energy late-arrival failure mode."""

    loss = F.mse_loss(prediction, target)
    loss = loss + float(cfg.get("late_time_mse_weight", 0.20)) * late_time_weighted_mse(
        prediction,
        target,
        start_fraction=float(cfg.get("late_time_start_fraction", 0.28)),
        max_extra_weight=float(cfg.get("late_time_max_extra_weight", 2.0)),
    )
    loss = loss + float(cfg.get("window_relative_loss_weight", 0.0015)) * window_relative_mse(
        prediction,
        target,
        windows=((50, 90), (90, 140), (140, 179)),
        weights=(0.45, 0.40, 0.15),
        power_floor_fraction=float(cfg.get("window_power_floor_fraction", 0.08)),
    )
    loss = loss + float(cfg.get("late_underpower_loss_weight", 0.0007)) * late_window_underpower_loss(
        prediction,
        target,
        rms_floor_fraction=float(cfg.get("late_rms_floor_fraction", 0.03)),
    )
    loss = loss + float(cfg.get("late_correlation_loss_weight", 0.0015)) * late_window_correlation_loss(
        prediction,
        target,
    )
    loss = loss + float(cfg.get("arrival_prior_loss_weight", 0.05)) * arrival_prior_weighted_mse(
        prediction,
        target,
        model,
        prior_strength=float(cfg.get("arrival_prior_strength", 1.0)),
        late_start_fraction=float(cfg.get("arrival_prior_late_start_fraction", 0.25)),
    )
    loss = loss + float(cfg.get("gradient_loss_weight", 0.002)) * gradient_loss(prediction, target)
    loss = loss + float(cfg.get("spectral_loss_weight", 0.0)) * F.l1_loss(
        torch.abs(torch.fft.rfft(prediction, dim=-2)),
        torch.abs(torch.fft.rfft(target, dim=-2)),
    )
    return loss


def surrogate_tail_gain_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: dict[str, Any],
    model: torch.Tensor | None = None,
) -> torch.Tensor:
    """Loss for low-energy mid/late arrivals without abandoning global waveform fit."""

    loss = float(cfg.get("global_mse_loss_weight", 1.0)) * F.mse_loss(prediction, target)
    loss = loss + float(cfg.get("gain_domain_loss_weight", 0.25)) * time_gain_domain_mse(
        prediction,
        target,
        start_fraction=float(cfg.get("gain_domain_start_fraction", 0.18)),
        max_gain=float(cfg.get("gain_domain_max_gain", 5.0)),
        power=float(cfg.get("gain_domain_power", 1.4)),
    )
    loss = loss + float(cfg.get("sample_window_relative_loss_weight", 0.006)) * per_sample_window_relative_mse(
        prediction,
        target,
        power_floor_fraction=float(cfg.get("sample_window_power_floor_fraction", 0.04)),
    )
    loss = loss + float(cfg.get("late_underpower_loss_weight", 0.001)) * late_window_underpower_loss(
        prediction,
        target,
        rms_floor_fraction=float(cfg.get("late_rms_floor_fraction", 0.03)),
    )
    loss = loss + float(cfg.get("late_correlation_loss_weight", 0.002)) * late_window_correlation_loss(
        prediction,
        target,
    )
    loss = loss + float(cfg.get("arrival_prior_loss_weight", 0.03)) * arrival_prior_weighted_mse(
        prediction,
        target,
        model,
        prior_strength=float(cfg.get("arrival_prior_strength", 1.0)),
        late_start_fraction=float(cfg.get("arrival_prior_late_start_fraction", 0.25)),
    )
    loss = loss + float(cfg.get("gradient_loss_weight", 0.001)) * gradient_loss(prediction, target)
    return loss


def surrogate_training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: dict[str, Any],
    model: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {
        "high_weight": float(cfg.get("high_energy_loss_weight", 0.15)),
        "low_weight": float(cfg.get("low_energy_loss_weight", 0.10)),
        "gradient_weight": float(cfg.get("gradient_loss_weight", 0.01)),
        "spectral_weight": float(cfg.get("spectral_loss_weight", 0.0005)),
    }
    variant = str(cfg.get("loss_variant", "balanced")).lower()
    if variant in {"mse", "pure_mse"}:
        return F.mse_loss(prediction, target)
    if variant == "balanced":
        return surrogate_balanced_loss(prediction, target, **kwargs)
    if variant == "physics_balanced":
        return surrogate_physics_balanced_loss(
            prediction,
            target,
            **kwargs,
            sample_nrmse_weight=float(cfg.get("sample_nrmse_loss_weight", 0.02)),
            window_relative_weight=float(cfg.get("window_relative_loss_weight", 0.005)),
            correlation_weight=float(cfg.get("correlation_loss_weight", 0.02)),
            envelope_weight=float(cfg.get("envelope_loss_weight", 0.01)),
            fk_spectral_weight=float(cfg.get("fk_spectral_loss_weight", 0.002)),
            window_power_floor_fraction=float(cfg.get("window_power_floor_fraction", 0.02)),
        )
    if variant in {"late_tail", "late_tail_balanced", "late_tail_physics"}:
        return surrogate_late_tail_loss(prediction, target, cfg, model=model)
    if variant in {"tail_gain", "tail_gain_balanced", "tail_gain_physics"}:
        return surrogate_tail_gain_loss(prediction, target, cfg, model=model)
    raise ValueError(f"Unknown forward surrogate loss_variant {variant!r}")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def train_inversion(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    backbone: str,
    seed: int,
    out_dir: Path,
    noise_mode: str = "mixed",
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(protocol["training"])
    if smoke:
        cfg.update({"epochs": 1, "early_stopping_patience": 1})
    train_indices = artifacts["splits"]["train"]
    models = artifacts["models"]
    clean = artifacts["clean_bscans"]
    dataset = DynamicNoiseDataset(
        clean=clean[train_indices],
        models=models[train_indices],
        global_indices=train_indices,
        residual_bank=artifacts["residual_bank"],
        base_seed=seed,
        snr_range=tuple(map(float, protocol["noise"]["train_snr_db"])),
        mode=noise_mode,
        field_weight=float(protocol["noise"]["field_weight"]),
        gaussian_weight=float(protocol["noise"]["gaussian_weight"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        generator=torch.Generator().manual_seed(seed),
    )
    validation = artifacts["fixed"]["validation"]
    val_truth = models[validation["indices"]]
    model = build_inversion_model(backbone, output_size=256).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: scheduler_lambda(epoch, int(cfg["epochs"]), int(cfg["warmup_epochs"])),
    )
    use_amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_rmse = float("inf")
    patience = 0
    history = []
    for epoch in range(int(cfg["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        losses = []
        for observation, truth in loader:
            observation, truth = observation.to(device), truth.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(observation)
                loss = F.mse_loss(prediction, truth) + float(cfg["gradient_loss_weight"]) * gradient_loss(
                    prediction, truth
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_by_view = {}
        for view_name, val_view in validation["views"].items():
            val_prediction = predict_batches(model, val_view, int(cfg["batch_size"]), device)
            val_by_view[view_name] = float(torch.sqrt(F.mse_loss(val_prediction, val_truth)))
        val_rmse = float(np.mean(list(val_by_view.values())))
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_normalized_rmse": val_rmse,
            "validation_epsilon_rmse": 8.0 * val_rmse,
            "validation_rmse_by_view": val_by_view,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(f"{backbone} seed={seed} epoch={epoch + 1}: train={record['train_loss']:.5f} val={val_rmse:.5f}")
        if val_rmse < best_rmse - 1.0e-6:
            best_rmse = val_rmse
            patience = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "backbone": backbone,
                    "seed": seed,
                    "noise_mode": noise_mode,
                    "protocol_hash": protocol["protocol_hash"],
                    "validation_normalized_rmse": best_rmse,
                    "epoch": epoch + 1,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= int(cfg["early_stopping_patience"]):
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    records = evaluate_checkpoint(protocol, artifacts, model, backbone, seed, noise_mode, device)
    write_csv(out_dir / "test_metrics.csv", records)
    summary = {
        "status": "complete",
        "protocol_hash": protocol["protocol_hash"],
        "backbone": backbone,
        "seed": seed,
        "noise_mode": noise_mode,
        "best_epoch": checkpoint["epoch"],
        "best_validation_normalized_rmse": best_rmse,
        "history": history,
        "test_aggregate": aggregate_records(records),
        "checkpoint": str(best_path),
        "test_metrics": str(out_dir / "test_metrics.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoint(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    model: torch.nn.Module,
    backbone: str,
    seed: int,
    noise_mode: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    fixed = artifacts["fixed"]["test"]
    truth = artifacts["models"][fixed["indices"]]
    records: list[dict[str, Any]] = []
    batch_size = int(protocol["training"]["batch_size"])
    for view_name, view in fixed["views"].items():
        predictions = predict_batches(model, view, batch_size, device)
        metrics = deterministic_metrics(
            predictions, truth, event_threshold=float(protocol["uq"]["event_threshold_normalized"])
        )
        for local, metric in enumerate(metrics):
            records.append(
                metric
                | {
                    "backbone": backbone,
                    "seed": seed,
                    "noise_mode": noise_mode,
                    "view": view_name,
                    "global_index": fixed["indices"][local],
                    "model_name": fixed["model_names"][local],
                    "protocol_hash": protocol["protocol_hash"],
                }
            )
    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for view in sorted({record["view"] for record in records}):
        subset = [record for record in records if record["view"] == view]
        aggregate[view] = {}
        for key in ("epsilon_rmse", "epsilon_mae", "ssim", "gradient_rmse", "high_eps_iou", "high_eps_f1"):
            values = np.asarray([float(record[key]) for record in subset], dtype=np.float64)
            aggregate[view][key] = {"mean": float(values.mean()), "std": float(values.std())}
    return aggregate


def train_forward_surrogate(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    out_dir: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    seed = int(protocol["split"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(protocol["forward_surrogate"])
    if smoke:
        cfg.update({"epochs": 1, "early_stopping_patience": 1})
    train_indices = artifacts["splits"]["train"]
    val_indices = artifacts["splits"]["validation"]
    models, clean = artifacts["models"], artifacts["clean_bscans"]
    train_tensor_indices = torch.as_tensor(train_indices, dtype=torch.long)
    val_tensor_indices = torch.as_tensor(val_indices, dtype=torch.long)
    models_device = models.to(device)
    clean_device = clean.to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    ensemble_seeds = [int(value) for value in cfg.get("ensemble_seeds", [seed])]
    if smoke:
        ensemble_seeds = ensemble_seeds[:1]
    history: list[dict[str, Any]] = []
    member_summaries: list[dict[str, Any]] = []
    member_state_dicts: list[dict[str, torch.Tensor]] = []
    val_predictions: list[torch.Tensor] = []
    architecture = str(cfg.get("surrogate_architecture", "WaveformSurrogate"))

    for member_index, member_seed in enumerate(ensemble_seeds):
        set_seed(member_seed)
        surrogate = build_waveform_surrogate(
            architecture,
            n_time=clean.shape[-2],
            n_traces=clean.shape[-1],
            reflectivity_features=bool(cfg.get("reflectivity_features", True)),
        ).to(device)
        optimizer = torch.optim.AdamW(
            surrogate.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg["epochs"]), eta_min=float(cfg.get("min_learning_rate", 4.0e-5))
        )
        best_member_nrmse = float("inf")
        best_member: dict[str, Any] | None = None
        best_member_state: dict[str, torch.Tensor] | None = None
        patience = 0
        generator = torch.Generator().manual_seed(member_seed)
        for epoch in range(int(cfg["epochs"])):
            surrogate.train()
            order = train_tensor_indices[torch.randperm(len(train_tensor_indices), generator=generator)]
            losses = []
            for start in range(0, len(order), int(cfg["batch_size"])):
                batch_indices = order[start : start + int(cfg["batch_size"])].to(device)
                model_batch = models_device[batch_indices]
                target = clean_device[batch_indices]
                if bool(cfg.get("horizontal_flip_augmentation", True)) and bool(
                    torch.randint(0, 2, (1,), generator=generator).item()
                ):
                    model_batch = torch.flip(model_batch, dims=(-1,))
                    target = torch.flip(target, dims=(-1,))
                prediction = surrogate(model_batch)
                loss = surrogate_training_loss(prediction, target, cfg, model=model_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(surrogate.parameters(), float(cfg.get("grad_clip_norm", 1.0)))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            prediction = predict_batches(surrogate, models[val_indices], int(cfg["batch_size"]), device)
            target = clean[val_indices]
            nrmse = normalized_data_rmse(prediction, target)
            pearson = pearson_flat(prediction, target)
            record = {
                "member": member_index,
                "seed": member_seed,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_nrmse": nrmse,
                "validation_pearson": pearson,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            history.append(record)
            print(
                f"surrogate member={member_index} seed={member_seed} epoch={epoch + 1}: "
                f"nrmse={nrmse:.5f} pearson={pearson:.5f}"
            )
            if nrmse < best_member_nrmse - 1.0e-6:
                best_member_nrmse, patience = nrmse, 0
                best_member = record
                best_member_state = {key: value.detach().cpu() for key, value in surrogate.state_dict().items()}
            else:
                patience += 1
                if patience >= int(cfg["early_stopping_patience"]):
                    break
        if best_member is None or best_member_state is None:
            raise RuntimeError("Forward surrogate training did not produce a checkpoint")
        surrogate.load_state_dict(best_member_state)
        member_state_dicts.append(best_member_state)
        member_summaries.append(best_member)
        val_predictions.append(predict_batches(surrogate, models[val_indices], int(cfg["batch_size"]), device))

    ensemble_prediction = torch.stack(val_predictions, dim=0).mean(dim=0)
    ensemble_nrmse = normalized_data_rmse(ensemble_prediction, clean[val_indices])
    ensemble_pearson = pearson_flat(ensemble_prediction, clean[val_indices])
    checkpoint = {
        "state_dict": member_state_dicts[0],
        "state_dicts": member_state_dicts,
        "protocol_hash": protocol["protocol_hash"],
        "architecture": architecture,
        "member_architectures": [architecture for _ in member_state_dicts],
        "reflectivity_features": bool(cfg.get("reflectivity_features", True)),
        "loss_variant": str(cfg.get("loss_variant", "balanced")),
        "loss_config": {
            key: cfg[key]
            for key in [
                "high_energy_loss_weight",
                "low_energy_loss_weight",
                "gradient_loss_weight",
                "spectral_loss_weight",
                "sample_nrmse_loss_weight",
                "window_relative_loss_weight",
                "correlation_loss_weight",
                "envelope_loss_weight",
                "fk_spectral_loss_weight",
                "window_power_floor_fraction",
                "global_mse_loss_weight",
                "gain_domain_loss_weight",
                "gain_domain_start_fraction",
                "gain_domain_max_gain",
                "gain_domain_power",
                "sample_window_relative_loss_weight",
                "sample_window_power_floor_fraction",
                "late_underpower_loss_weight",
                "late_correlation_loss_weight",
                "arrival_prior_loss_weight",
                "arrival_prior_strength",
                "arrival_prior_late_start_fraction",
            ]
            if key in cfg
        },
        "ensemble_seeds": ensemble_seeds,
        "member_summaries": member_summaries,
        "epoch": max(int(record["epoch"]) for record in member_summaries),
        "validation_nrmse": ensemble_nrmse,
        "validation_pearson": ensemble_pearson,
    }
    torch.save(checkpoint, best_path)
    gate_pass = (
        float(checkpoint["validation_nrmse"]) <= float(cfg["validation_nrmse_max"])
        and float(checkpoint["validation_pearson"]) >= float(cfg["validation_pearson_min"])
    )
    summary = {
        "status": "pass" if gate_pass else "fail",
        "protocol_hash": protocol["protocol_hash"],
        "checkpoint": str(best_path),
        "best_epoch": checkpoint["epoch"],
        "validation_nrmse": checkpoint["validation_nrmse"],
        "validation_pearson": checkpoint["validation_pearson"],
        "architecture": checkpoint["architecture"],
        "reflectivity_features": checkpoint["reflectivity_features"],
        "loss_variant": checkpoint["loss_variant"],
        "loss_config": checkpoint["loss_config"],
        "ensemble_seeds": ensemble_seeds,
        "member_summaries": member_summaries,
        "gate": {
            "nrmse_max": cfg["validation_nrmse_max"],
            "pearson_min": cfg["validation_pearson_min"],
        },
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_inversion_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_inversion_model(checkpoint["backbone"], output_size=256).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _build_surrogate_from_payload(
    checkpoint: dict[str, Any],
    device: torch.device,
    n_time: int = 179,
    n_traces: int = 137,
) -> torch.nn.Module:
    if checkpoint.get("surrogate_kind") == "deepwave_physics":
        cfg = dict(checkpoint.get("physics_config", {}))
        return DeepwavePhysicsSurrogate(
            profile=checkpoint["acquisition_profile"],
            shot_batch_size=int(cfg.get("shot_batch_size", 16)),
            pml_width=int(cfg.get("pml_width", 20)),
            epsilon_min=float(cfg.get("epsilon_min", 2.0)),
            epsilon_max=float(cfg.get("epsilon_max", 10.0)),
        ).to(device)

    if checkpoint.get("surrogate_kind") == "residual_refiner":
        base_payload = checkpoint.get("base_checkpoint_payload")
        if base_payload is not None:
            base = _build_surrogate_from_payload(base_payload, device, n_time=n_time, n_traces=n_traces)
        else:
            base_path = Path(checkpoint["base_checkpoint"])
            base, _ = load_surrogate_checkpoint(base_path, device, n_time=n_time, n_traces=n_traces)
        refiner_config = dict(checkpoint.get("refiner_config", {}))
        refiner = BaseConditionedResidualRefiner(
            n_time=n_time,
            n_traces=n_traces,
            reflectivity_features=bool(refiner_config.get("reflectivity_features", True)),
            correction_scale=float(refiner_config.get("correction_scale", 0.35)),
            correction_start_fraction=float(refiner_config.get("correction_start_fraction", 0.0)),
            correction_ramp_width_fraction=float(refiner_config.get("correction_ramp_width_fraction", 0.15)),
        ).to(device)
        refiner.load_state_dict(checkpoint["refiner_state_dict"])
        surrogate = ResidualRefinedSurrogate(
            base,
            refiner,
            residual_alpha=float(checkpoint.get("residual_alpha", 1.0)),
        ).to(device)
        if checkpoint.get("output_calibration") is not None:
            surrogate = OutputCalibratedSurrogate(surrogate, checkpoint["output_calibration"]).to(device)
        return surrogate

    state_dicts = checkpoint.get("state_dicts") or [checkpoint["state_dict"]]
    architecture = str(checkpoint.get("architecture", "WaveformSurrogate"))
    member_architectures = checkpoint.get("member_architectures") or [architecture for _ in state_dicts]
    members = []
    for state_dict, member_architecture in zip(state_dicts, member_architectures):
        member = build_waveform_surrogate(
            str(member_architecture),
            n_time=n_time,
            n_traces=n_traces,
            reflectivity_features=bool(checkpoint.get("reflectivity_features", True)),
        ).to(device)
        member.load_state_dict(state_dict)
        members.append(member)
    weights = checkpoint.get("ensemble_weights")
    surrogate = members[0] if len(members) == 1 else WaveformSurrogateEnsemble(members, weights=weights).to(device)
    if checkpoint.get("output_calibration") is not None:
        surrogate = OutputCalibratedSurrogate(surrogate, checkpoint["output_calibration"]).to(device)
    return surrogate


def load_surrogate_checkpoint(path: Path, device: torch.device, n_time: int = 179, n_traces: int = 137) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    surrogate = _build_surrogate_from_payload(checkpoint, device, n_time=n_time, n_traces=n_traces)
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    return surrogate, checkpoint
