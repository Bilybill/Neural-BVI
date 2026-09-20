"""Lightweight Deepwave scalar forward module for LA010010 surrogate gates.

This file intentionally avoids importing ``build_deepwave_dataset.py`` because
that script imports image-denoising dependencies that can fail during normal
surrogate loading on this Windows environment. The numerical operations below
mirror the dataset generator's Deepwave simulation and LA010010 preprocessing.

The module exposes two execution paths:

* an exact NumPy/SciPy path used when gradients are not required, preserving the
  byte-level gate comparison with prepared clean B-scans;
* a Torch-only differentiable path used by BVI when the input model requires
  gradients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.ndimage
import scipy.signal
import torch
import torch.nn.functional as F
from torch import nn


C0 = 299_792_458.0
UNIT_SCALE = 1.0e6


def blackman_harris_derivative(frequency: float, time: np.ndarray) -> np.ndarray:
    coefficients = [0.35322222, -0.488, 0.145, -0.010222222]
    duration = 1.14 / frequency
    window = np.zeros_like(time)
    active = time < duration
    for order, coefficient in enumerate(coefficients):
        window[active] += coefficient * np.cos(2.0 * order * np.pi * time[active] / duration)
    pulse = np.concatenate((window[1:], np.zeros(1))) - window
    return (pulse / max(float(np.max(np.abs(pulse))), 1.0e-12)).astype(np.float32)


def torch_blackman_harris_derivative(frequency: float, n_time: int, dt: float, device: torch.device) -> torch.Tensor:
    coefficients = [0.35322222, -0.488, 0.145, -0.010222222]
    time = torch.arange(n_time, dtype=torch.float32, device=device) * float(dt)
    duration = 1.14 / float(frequency)
    active = time < duration
    window = torch.zeros_like(time)
    for order, coefficient in enumerate(coefficients):
        window = window + torch.where(
            active,
            float(coefficient) * torch.cos(2.0 * order * torch.pi * time / duration),
            torch.zeros_like(time),
        )
    pulse = torch.cat((window[1:], window.new_zeros(1))) - window
    return pulse / pulse.abs().amax().clamp_min(1.0e-12)


def preprocess_bscan(data: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    cfg = profile["preprocessing"]
    result = data.astype(np.float32)
    if cfg.get("dc_removal", False):
        result -= result.mean(axis=0, keepdims=True)
    butterworth = cfg.get("bandpass_butterworth_hz")
    if butterworth:
        dt = float(profile["observation"]["sample_interval_s"])
        nyquist = 0.5 / dt
        low_hz, high_hz = map(float, butterworth)
        low = max(low_hz / nyquist, 1.0e-4)
        high = min(high_hz / nyquist, 0.98)
        if high <= low:
            raise ValueError(f"Invalid Butterworth passband {butterworth} for dt={dt}")
        sos = scipy.signal.butter(int(cfg.get("bandpass_order", 4)), [low, high], btype="bandpass", output="sos")
        result = scipy.signal.sosfiltfilt(sos, result, axis=0).astype(np.float32)
    dewow = int(cfg.get("dewow_samples", 0))
    if dewow > 1:
        result -= scipy.ndimage.uniform_filter1d(result, size=dewow, axis=0, mode="nearest")
    background = cfg.get("background_subtraction", False)
    if background == "median":
        result -= np.median(result, axis=1, keepdims=True)
    elif background:
        result -= result.mean(axis=1, keepdims=True)
    if cfg.get("sqrt_time_gain", False):
        result *= np.sqrt(np.arange(result.shape[0], dtype=np.float32) + 1.0)[:, None]
    if cfg.get("gain") == "linear":
        gain_end = float(cfg.get("linear_gain_end", 4.0))
        result *= np.linspace(1.0, gain_end, result.shape[0], dtype=np.float32)[:, None]
    percentile = float(cfg.get("clip_percentile", 99.0))
    scale = float(np.percentile(np.abs(result), percentile))
    result = np.clip(result / max(scale, 1.0e-8), -1.0, 1.0)
    return result.astype(np.float32)


def torch_odd_extension(data: torch.Tensor, padlen: int) -> torch.Tensor:
    if padlen <= 0:
        return data
    if data.shape[0] <= padlen:
        raise ValueError(f"Cannot apply filtfilt padding {padlen} to {data.shape[0]} samples")
    left = 2.0 * data[:1] - data[1 : padlen + 1].flip(0)
    right = 2.0 * data[-1:] - data[-padlen - 1 : -1].flip(0)
    return torch.cat((left, data, right), dim=0)


def torch_sosfilt(signal: torch.Tensor, sos: torch.Tensor, zi: torch.Tensor, zi_scale: torch.Tensor) -> torch.Tensor:
    current = signal
    section_count = sos.shape[0]
    for section in range(section_count):
        b0, b1, b2, _a0, a1, a2 = [sos[section, idx] for idx in range(6)]
        z1 = zi[section, 0] * zi_scale
        z2 = zi[section, 1] * zi_scale
        outputs = []
        for sample in current.unbind(dim=0):
            value = b0 * sample + z1
            z1, z2 = b1 * sample - a1 * value + z2, b2 * sample - a2 * value
            outputs.append(value)
        current = torch.stack(outputs, dim=0)
    return current


def torch_sosfiltfilt(data: torch.Tensor, sos_np: np.ndarray) -> torch.Tensor:
    original_dtype = data.dtype
    work = data.to(torch.float64)
    sos = torch.as_tensor(sos_np, dtype=work.dtype, device=work.device)
    zi_np = scipy.signal.sosfilt_zi(sos_np)
    zi = torch.as_tensor(zi_np, dtype=work.dtype, device=work.device)
    section_count = int(sos.shape[0])
    zeros_b2 = int(np.sum(np.isclose(sos_np[:, 2], 0.0)))
    zeros_a2 = int(np.sum(np.isclose(sos_np[:, 5], 0.0)))
    padlen = 3 * (2 * section_count + 1 - min(zeros_b2, zeros_a2))
    extended = torch_odd_extension(work, padlen)
    forward = torch_sosfilt(extended, sos, zi, extended[0])
    backward = torch_sosfilt(forward.flip(0), sos, zi, forward[-1])
    filtered = backward.flip(0)[padlen:-padlen]
    return filtered.to(dtype=original_dtype)


def torch_bandpass(data: torch.Tensor, profile: dict[str, Any], order: int) -> torch.Tensor:
    butterworth = profile["preprocessing"].get("bandpass_butterworth_hz")
    if not butterworth:
        return data
    dt = float(profile["observation"]["sample_interval_s"])
    nyquist = 0.5 / dt
    low_hz, high_hz = map(float, butterworth)
    low_hz = max(low_hz, nyquist * 1.0e-4)
    high_hz = min(high_hz, nyquist * 0.98)
    if high_hz <= low_hz:
        raise ValueError(f"Invalid Butterworth passband {butterworth} for dt={dt}")
    sos = scipy.signal.butter(int(order), [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return torch_sosfiltfilt(data, sos)


def torch_uniform_filter1d_time(data: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return data
    pad_left = int(size) // 2
    pad_right = int(size) - 1 - pad_left
    channels = data.shape[1]
    padded = F.pad(data.T.unsqueeze(0), (pad_left, pad_right), mode="replicate")
    kernel = data.new_full((channels, 1, int(size)), 1.0 / float(size))
    return F.conv1d(padded, kernel, groups=channels).squeeze(0).T


def torch_preprocess_bscan(data: torch.Tensor, profile: dict[str, Any]) -> torch.Tensor:
    cfg = profile["preprocessing"]
    result = data.float()
    if cfg.get("dc_removal", False):
        result = result - result.mean(dim=0, keepdim=True)
    result = torch_bandpass(result, profile, int(cfg.get("bandpass_order", 4)))
    dewow = int(cfg.get("dewow_samples", 0))
    if dewow > 1:
        result = result - torch_uniform_filter1d_time(result, dewow)
    background = cfg.get("background_subtraction", False)
    if background == "median":
        result = result - result.median(dim=1, keepdim=True).values
    elif background:
        result = result - result.mean(dim=1, keepdim=True)
    if cfg.get("sqrt_time_gain", False):
        gain = torch.sqrt(torch.arange(result.shape[0], dtype=result.dtype, device=result.device) + 1.0)
        result = result * gain[:, None]
    if cfg.get("gain") == "linear":
        gain_end = float(cfg.get("linear_gain_end", 4.0))
        gain = torch.linspace(1.0, gain_end, result.shape[0], dtype=result.dtype, device=result.device)
        result = result * gain[:, None]
    percentile = float(cfg.get("clip_percentile", 99.0)) / 100.0
    scale = torch.quantile(result.abs().flatten(), percentile).clamp_min(1.0e-8)
    return torch.clamp(result / scale, -1.0, 1.0)


def simulate_one(
    normalized_model: torch.Tensor,
    profile: dict[str, Any],
    device: torch.device,
    shot_batch_size: int = 16,
    pml_width: int = 20,
    epsilon_min: float = 2.0,
    epsilon_max: float = 10.0,
) -> np.ndarray:
    from deepwave import scalar

    model_cfg = profile["model_domain"]
    obs = profile["observation"]
    permittivity = (
        normalized_model.detach().float().clamp(0.0, 1.0).cpu().numpy()
        * (float(epsilon_max) - float(epsilon_min))
        + float(epsilon_min)
    ).astype(np.float32)
    nz, nx = permittivity.shape
    target_depth = float(model_cfg.get("target_depth_m", model_cfg.get("depth_m")))
    dz = target_depth / nz
    dx = float(model_cfg["width_m"]) / nx
    dt_real = float(obs["sample_interval_s"])
    n_time = int(obs["n_time"])
    n_traces = int(obs["n_traces"])
    trace_spacing = float(obs["trace_spacing_m"])
    left_margin = float(model_cfg.get("left_margin_m", 0.0))
    tx_rx_offset = float(obs["tx_rx_offset_m"])
    antenna_depth = float(obs["antenna_depth_m"])
    frequency_real = float(profile["source"]["center_frequency_hz"])

    source_x = left_margin + np.arange(n_traces, dtype=np.float64) * trace_spacing
    receiver_x = source_x + tx_rx_offset
    if receiver_x.max() >= float(model_cfg["width_m"]):
        raise ValueError("Acquisition aperture exceeds the configured model width")

    fixed_arrays: list[np.ndarray] = []
    for layer in model_cfg.get("fixed_layers", []):
        layer_cells = max(1, int(round(float(layer["thickness_m"]) / dz)))
        fixed_arrays.append(np.full((layer_cells, nx), float(layer["relative_permittivity"]), dtype=np.float32))
    simulation_permittivity = np.concatenate([*fixed_arrays, permittivity], axis=0) if fixed_arrays else permittivity

    source_z_index = int(round(antenna_depth / dz))
    source_x_index = np.rint(source_x / dx).astype(np.int64)
    receiver_x_index = np.rint(receiver_x / dx).astype(np.int64)
    velocity = torch.as_tensor(C0 / np.sqrt(simulation_permittivity) / UNIT_SCALE, dtype=torch.float32, device=device)
    dt_scaled = dt_real * UNIT_SCALE
    frequency_scaled = frequency_real / UNIT_SCALE
    time_scaled = np.arange(n_time, dtype=np.float64) * dt_scaled
    wavelet = torch.as_tensor(blackman_harris_derivative(frequency_scaled, time_scaled), dtype=torch.float32, device=device)

    records: list[torch.Tensor] = []
    for start in range(0, n_traces, int(shot_batch_size)):
        stop = min(start + int(shot_batch_size), n_traces)
        count = stop - start
        source_locations = torch.zeros((count, 1, 2), dtype=torch.long, device=device)
        receiver_locations = torch.zeros((count, 1, 2), dtype=torch.long, device=device)
        source_locations[:, 0, 0] = source_z_index
        receiver_locations[:, 0, 0] = source_z_index
        source_locations[:, 0, 1] = torch.as_tensor(source_x_index[start:stop], device=device)
        receiver_locations[:, 0, 1] = torch.as_tensor(receiver_x_index[start:stop], device=device)
        amplitudes = wavelet.view(1, 1, -1).expand(count, 1, -1).contiguous()
        receiver = scalar(
            velocity,
            [dz, dx],
            dt=dt_scaled,
            source_amplitudes=amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=int(pml_width),
            pml_freq=frequency_scaled,
            accuracy=4,
        )[-1]
        records.append(receiver[:, 0].detach().cpu())
    bscan = torch.cat(records, dim=0).numpy().T
    return preprocess_bscan(bscan, profile)


def simulate_one_differentiable(
    normalized_model: torch.Tensor,
    profile: dict[str, Any],
    shot_batch_size: int = 16,
    pml_width: int = 20,
    epsilon_min: float = 2.0,
    epsilon_max: float = 10.0,
) -> torch.Tensor:
    from deepwave import scalar

    model_cfg = profile["model_domain"]
    obs = profile["observation"]
    device = normalized_model.device
    dtype = normalized_model.dtype
    permittivity = normalized_model.float().clamp(0.0, 1.0) * (
        float(epsilon_max) - float(epsilon_min)
    ) + float(epsilon_min)
    nz, nx = permittivity.shape
    target_depth = float(model_cfg.get("target_depth_m", model_cfg.get("depth_m")))
    dz = target_depth / nz
    dx = float(model_cfg["width_m"]) / nx
    dt_real = float(obs["sample_interval_s"])
    n_time = int(obs["n_time"])
    n_traces = int(obs["n_traces"])
    trace_spacing = float(obs["trace_spacing_m"])
    left_margin = float(model_cfg.get("left_margin_m", 0.0))
    tx_rx_offset = float(obs["tx_rx_offset_m"])
    antenna_depth = float(obs["antenna_depth_m"])
    frequency_real = float(profile["source"]["center_frequency_hz"])

    source_x = left_margin + np.arange(n_traces, dtype=np.float64) * trace_spacing
    receiver_x = source_x + tx_rx_offset
    if receiver_x.max() >= float(model_cfg["width_m"]):
        raise ValueError("Acquisition aperture exceeds the configured model width")

    fixed_tensors: list[torch.Tensor] = []
    for layer in model_cfg.get("fixed_layers", []):
        layer_cells = max(1, int(round(float(layer["thickness_m"]) / dz)))
        fixed_tensors.append(
            torch.full(
                (layer_cells, nx),
                float(layer["relative_permittivity"]),
                dtype=permittivity.dtype,
                device=device,
            )
        )
    simulation_permittivity = torch.cat([*fixed_tensors, permittivity], dim=0) if fixed_tensors else permittivity

    source_z_index = int(round(antenna_depth / dz))
    source_x_index = np.rint(source_x / dx).astype(np.int64)
    receiver_x_index = np.rint(receiver_x / dx).astype(np.int64)
    velocity = C0 / torch.sqrt(simulation_permittivity) / UNIT_SCALE
    dt_scaled = dt_real * UNIT_SCALE
    frequency_scaled = frequency_real / UNIT_SCALE
    wavelet = torch_blackman_harris_derivative(frequency_scaled, n_time, dt_scaled, device).to(dtype=torch.float32)

    records: list[torch.Tensor] = []
    for start in range(0, n_traces, int(shot_batch_size)):
        stop = min(start + int(shot_batch_size), n_traces)
        count = stop - start
        source_locations = torch.zeros((count, 1, 2), dtype=torch.long, device=device)
        receiver_locations = torch.zeros((count, 1, 2), dtype=torch.long, device=device)
        source_locations[:, 0, 0] = source_z_index
        receiver_locations[:, 0, 0] = source_z_index
        source_locations[:, 0, 1] = torch.as_tensor(source_x_index[start:stop], device=device)
        receiver_locations[:, 0, 1] = torch.as_tensor(receiver_x_index[start:stop], device=device)
        amplitudes = wavelet.view(1, 1, -1).expand(count, 1, -1).contiguous()
        receiver = scalar(
            velocity,
            [dz, dx],
            dt=dt_scaled,
            source_amplitudes=amplitudes,
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            pml_width=int(pml_width),
            pml_freq=frequency_scaled,
            accuracy=4,
        )[-1]
        records.append(receiver[:, 0])
    bscan = torch.cat(records, dim=0).T
    return torch_preprocess_bscan(bscan, profile).to(dtype=dtype)


class DeepwavePhysicsSurrogate(nn.Module):
    """Exact Deepwave scalar forward backend for prepared LA010010 models."""

    def __init__(
        self,
        profile: dict[str, Any],
        shot_batch_size: int = 16,
        pml_width: int = 20,
        epsilon_min: float = 2.0,
        epsilon_max: float = 10.0,
    ):
        super().__init__()
        self.profile = profile
        self.shot_batch_size = int(shot_batch_size)
        self.pml_width = int(pml_width)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        device = model.device
        outputs = []
        differentiable = torch.is_grad_enabled() and model.requires_grad
        if differentiable:
            for sample in model:
                outputs.append(
                    simulate_one_differentiable(
                        sample[0],
                        self.profile,
                        shot_batch_size=self.shot_batch_size,
                        pml_width=self.pml_width,
                        epsilon_min=self.epsilon_min,
                        epsilon_max=self.epsilon_max,
                    ).unsqueeze(0)
                )
        else:
            for sample in model:
                bscan = simulate_one(
                    sample[0],
                    self.profile,
                    device,
                    shot_batch_size=self.shot_batch_size,
                    pml_width=self.pml_width,
                    epsilon_min=self.epsilon_min,
                    epsilon_max=self.epsilon_max,
                )
                outputs.append(torch.from_numpy(bscan).to(device=device, dtype=model.dtype).unsqueeze(0))
        return torch.stack(outputs, dim=0)
