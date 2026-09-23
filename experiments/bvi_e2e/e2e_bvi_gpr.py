"""Neural-BVI inversion trained from an explicit Deepwave GPR dataset.

The offline dataset is created by ``build_deepwave_dataset.py``. A validated
differentiable surrogate of those Deepwave records is used inside BVI because
directly running thousands of 256x256 wave simulations per posterior case is
not computationally practical.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepwave_dataset import DeepwaveForwardSurrogate, load_deepwave_dataset
from path_utils import resolve_data_path


def numeric_model_key(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_permittivity_models(model_dir: Path, count: int, image_size: int) -> np.ndarray:
    paths = sorted(model_dir.glob("model_*.mat"), key=numeric_model_key)[:count]
    if not paths:
        raise FileNotFoundError(f"No model_*.mat files found in {model_dir}")

    models: List[np.ndarray] = []
    for path in paths:
        mat = sio.loadmat(path)
        if "model" in mat:
            arr = np.asarray(mat["model"], dtype=np.float32)
        elif "ep" in mat:
            arr = np.asarray(mat["ep"], dtype=np.float32)
        else:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"{path} does not contain 'model' or 'ep'; keys={keys}")

        # Center crop square if needed, then downsample to the experiment grid.
        h, w = arr.shape
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        arr = arr[y0 : y0 + size, x0 : x0 + size]
        zoom = image_size / float(size)
        arr = scipy.ndimage.zoom(arr, zoom=zoom, order=1)
        models.append(arr.astype(np.float32))

    data = np.stack(models, axis=0)
    # Fixed normalization for these generated models: epsilon_r is mostly [2, 10].
    data = np.clip((data - 2.0) / 8.0, 0.0, 1.0)
    return data[:, None, :, :]


class HyperbolaForward(nn.Module):
    """Differentiable common-offset-like GPR projection.

    The operator maps a 2-D permittivity image to a B-scan by summing vertical
    reflectivity along hyperbolic travel-time curves for each trace location.
    It is not a Maxwell solver; it is a controlled proxy for fast method tests.
    """

    def __init__(self, image_size: int, n_traces: int, n_time: int, sigma_t: float = 1.25):
        super().__init__()
        x = torch.linspace(0.0, 1.0, image_size)
        z = torch.linspace(0.02, 1.02, image_size)
        xx, zz = torch.meshgrid(x, z, indexing="xy")
        xx = xx.T.contiguous()
        zz = zz.T.contiguous()

        trace_x = torch.linspace(0.0, 1.0, n_traces)
        time_axis = torch.arange(n_time, dtype=torch.float32)
        kernels = []
        t_max = 2.0 * math.sqrt(2.0)
        for sx in trace_x:
            dist = torch.sqrt((xx - sx) ** 2 + zz**2)
            t_index = (2.0 * dist / t_max) * (n_time - 1)
            weight = torch.exp(-0.5 * ((time_axis[:, None, None] - t_index) / sigma_t) ** 2)
            weight = weight / (dist[None, :, :] + 0.05)
            weight = weight / (weight.sum(dim=0, keepdim=True) + 1e-6)
            kernels.append(weight)

        kernel = torch.stack(kernels, dim=0)  # traces, time, h, w
        self.register_buffer("kernel", kernel)

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        # model: [B, 1, H, W], returns [B, 1, T, X]
        m = model[:, 0]
        refl = torch.zeros_like(m)
        refl[:, 1:, :] = m[:, 1:, :] - m[:, :-1, :]
        bscan = torch.einsum("bhw,xthw->btx", refl, self.kernel)
        bscan = bscan / (bscan.abs().amax(dim=(1, 2), keepdim=True) + 1e-6)
        return bscan[:, None]


class TinyInversionNet(nn.Module):
    def __init__(self, image_size: int):
        super().__init__()
        self.image_size = image_size
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, bscan: torch.Tensor) -> torch.Tensor:
        x = self.enc(bscan)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.dec(x)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    py = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    px = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.l1_loss(py, ty) + F.l1_loss(px, tx)


def train_network(
    net: nn.Module,
    train_loader: DataLoader,
    val: Tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    device: torch.device,
) -> List[dict]:
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    history: List[dict] = []
    val_x, val_y = val

    for epoch in range(1, epochs + 1):
        net.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            loss = F.mse_loss(pred, yb) + 0.05 * gradient_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        net.eval()
        with torch.no_grad():
            squared_error = 0.0
            absolute_error = 0.0
            element_count = 0
            for start in range(0, len(val_x), train_loader.batch_size or 1):
                vx = val_x[start : start + (train_loader.batch_size or 1)].to(device)
                vy = val_y[start : start + (train_loader.batch_size or 1)].to(device)
                vp = net(vx)
                squared_error += float(F.mse_loss(vp, vy, reduction="sum").cpu())
                absolute_error += float(F.l1_loss(vp, vy, reduction="sum").cpu())
                element_count += vy.numel()
            val_rmse = math.sqrt(squared_error / element_count)
            val_mae = absolute_error / element_count
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_rmse": val_rmse, "val_mae": val_mae}
        history.append(rec)
        print(f"epoch {epoch:03d}: train_loss={rec['train_loss']:.5f} val_rmse={val_rmse:.5f} val_mae={val_mae:.5f}")
    return history


def train_forward_surrogate(
    forward: nn.Module,
    train_loader: DataLoader,
    val: Tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    device: torch.device,
) -> List[dict]:
    optimizer = torch.optim.AdamW(forward.parameters(), lr=lr, weight_decay=1e-4)
    val_models, val_bscans = val
    history: List[dict] = []
    for epoch in range(1, epochs + 1):
        forward.train()
        losses = []
        for bscans, models in train_loader:
            models = models.to(device)
            bscans = bscans.to(device)
            prediction = forward(models)
            loss = F.mse_loss(prediction, bscans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        forward.eval()
        with torch.no_grad():
            error_sum = 0.0
            element_count = 0
            batch_size = train_loader.batch_size or 1
            for start in range(0, len(val_models), batch_size):
                models = val_models[start : start + batch_size].to(device)
                target = val_bscans[start : start + batch_size].to(device)
                prediction = forward(models)
                error_sum += float(F.mse_loss(prediction, target, reduction="sum").cpu())
                element_count += target.numel()
            val_rmse = math.sqrt(error_sum / element_count)
        record = {"epoch": epoch, "train_mse": float(np.mean(losses)), "val_rmse": val_rmse}
        history.append(record)
        print(
            f"forward epoch {epoch:03d}: train_mse={record['train_mse']:.6f} "
            f"val_rmse={val_rmse:.6f}"
        )
    return history


def predict_in_batches(net: nn.Module, inputs: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            outputs.append(net(inputs[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


def make_residual_basis(image_size: int, latent_dim: int, device: torch.device, basis_type: str) -> torch.Tensor:
    """Create mixed global/local residual bases for BVI.

    Low-frequency cosine atoms describe broad layer uncertainty. Local RBF atoms
    allow posterior samples to express compact GPR anomalies such as pipes,
    cavities, roots, or lens-shaped dielectric inclusions.
    """
    coords = torch.linspace(0.0, math.pi, image_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    basis = []
    n_cos = latent_dim if basis_type == "cosine" else max(4, latent_dim // 2)
    freqs = []
    max_freq = 1
    while len(freqs) < n_cos:
        for fy in range(max_freq + 1):
            for fx in range(max_freq + 1):
                if fy == 0 and fx == 0:
                    continue
                pair = (fy, fx)
                if pair not in freqs:
                    freqs.append(pair)
                if len(freqs) >= n_cos:
                    break
            if len(freqs) >= n_cos:
                break
        max_freq += 1

    for fy, fx in freqs[:n_cos]:
        b = torch.cos(fy * yy) * torch.cos(fx * xx)
        b = b - b.mean()
        b = b / (b.std() + 1e-6)
        basis.append(b)

    if basis_type == "mixed":
        grid_n = int(math.ceil(math.sqrt(max(1, latent_dim - len(basis)))))
        y_centers = torch.linspace(0.18, math.pi * 0.92, grid_n, device=device)
        x_centers = torch.linspace(0.08, math.pi * 0.92, grid_n, device=device)
        sigma = math.pi / (grid_n + 1)
        for cy in y_centers:
            for cx in x_centers:
                if len(basis) >= latent_dim:
                    break
                b = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
                b = b - b.mean()
                b = b / (b.std() + 1e-6)
                basis.append(b)
            if len(basis) >= latent_dim:
                break
    return torch.stack(basis, dim=0)


def log_normal_diag(z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    var_term = ((z - mean) / log_std.exp()) ** 2
    return -0.5 * (var_term + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=-1)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


@dataclass
class BVIResult:
    rmse_nn: float
    rmse_bvi_mean: float
    mae_nn: float
    mae_bvi_mean: float
    coverage_1sigma: float
    coverage_2sigma: float
    interval_width_2sigma: float
    gaussian_nll: float
    mean_std: float
    final_loss: float
    data_misfit_nn: float | None = None
    data_misfit_bvi_mean: float | None = None
    high_eps_threshold: float = 0.5
    high_eps_area_mean: float | None = None
    high_eps_area_std: float | None = None
    central_high_eps_prob: float | None = None
    max_high_eps_prob: float | None = None
    mixture_effective_components: float | None = None
    mixture_max_weight: float | None = None
    stage_new_weights: List[float] | None = None
    stage_line_search_objectives: List[float] | None = None
    stage_line_search_gains: List[float] | None = None


def run_bvi(
    forward: nn.Module,
    observation: torch.Tensor,
    true_model: torch.Tensor | None,
    nn_mean: torch.Tensor,
    latent_dim: int,
    components: int,
    steps: int,
    samples_per_component: int,
    residual_scale: float,
    noise_std: float,
    kl_weight: float,
    model_prior_std: float,
    interrogation_threshold: float,
    basis_type: str,
    device: torch.device,
    posterior_sample_count: int | None = None,
    return_samples: bool = False,
    basis_override: torch.Tensor | None = None,
    latent_prior_mean: torch.Tensor | None = None,
    latent_prior_std: torch.Tensor | None = None,
    model_prior_center_override: torch.Tensor | None = None,
    learning_rate: float = 0.03,
    initial_log_std: float = -1.1,
    antithetic_samples: bool = False,
    forward_batch_size: int | None = None,
    streaming_backward: bool = False,
    initial_component_weight: float = 0.1,
    component_weight_learning_rate: float = 0.3,
    mixture_line_search_samples: int = 4,
) -> tuple:
    if components < 1:
        raise ValueError("components must be at least one")
    if steps < 1:
        raise ValueError("steps must be at least one")

    if basis_override is None:
        basis = make_residual_basis(nn_mean.shape[-1], latent_dim, device, basis_type)
    else:
        basis = basis_override.to(device=device, dtype=nn_mean.dtype)
        expected_shape = (int(latent_dim), int(nn_mean.shape[-2]), int(nn_mean.shape[-1]))
        if tuple(basis.shape) != expected_shape:
            raise ValueError(f"Residual basis shape {tuple(basis.shape)} does not match {expected_shape}")
    obs = observation.to(device)
    truth = true_model.to(device) if true_model is not None else None
    base = nn_mean.to(device).detach()
    prior_mean = (
        torch.zeros(int(latent_dim), device=device, dtype=base.dtype)
        if latent_prior_mean is None
        else latent_prior_mean.to(device=device, dtype=base.dtype).flatten()
    )
    prior_std = (
        torch.ones(int(latent_dim), device=device, dtype=base.dtype)
        if latent_prior_std is None
        else latent_prior_std.to(device=device, dtype=base.dtype).flatten()
    )
    if prior_mean.numel() != int(latent_dim) or prior_std.numel() != int(latent_dim):
        raise ValueError("Latent prior tensors must match latent_dim")
    prior_std = prior_std.clamp_min(1.0e-6)
    prior_log_std = prior_std.log()
    model_prior_center = (
        base
        if model_prior_center_override is None
        else model_prior_center_override.to(device=device, dtype=base.dtype).detach()
    )
    if tuple(model_prior_center.shape) != tuple(base.shape):
        raise ValueError("Model-prior center must match the neural prediction shape")
    final_loss = 0.0

    frozen_means: List[torch.Tensor] = []
    frozen_log_stds: List[torch.Tensor] = []
    mixture_weights = torch.ones(1, device=device)
    stage_new_weights: List[float] = []
    stage_line_search_objectives: List[float] = []
    stage_line_search_gains: List[float] = []

    def line_search_new_weight(
        stage_means: List[torch.Tensor],
        stage_log_stds: List[torch.Tensor],
        old_weights: torch.Tensor,
        optimized_weight: float,
    ) -> tuple[float, float, float]:
        sample_count = max(1, int(mixture_line_search_samples))
        latent_samples: List[torch.Tensor] = []
        non_kl_objectives: List[torch.Tensor] = []
        chunk_size = max(1, int(forward_batch_size or sample_count))
        with torch.no_grad():
            for mean_k, log_std_k in zip(stage_means, stage_log_stds):
                if antithetic_samples and sample_count >= 2:
                    half = (sample_count + 1) // 2
                    positive = torch.randn(half, latent_dim, device=device)
                    eps = torch.cat((positive, -positive), dim=0)[:sample_count]
                else:
                    eps = torch.randn(sample_count, latent_dim, device=device)
                z = mean_k + log_std_k.exp() * eps
                latent_samples.append(z)
                per_sample: List[torch.Tensor] = []
                for z_chunk in z.split(chunk_size, dim=0):
                    residual = torch.einsum("sd,dhw->shw", z_chunk, basis)[:, None]
                    candidates = torch.clamp(base + residual_scale * residual, 0.0, 1.0)
                    prediction = forward(candidates)
                    data_value = ((prediction - obs) ** 2).mean(dim=(1, 2, 3)) / (2.0 * noise_std**2)
                    if model_prior_std > 0.0:
                        model_value = ((candidates - model_prior_center) ** 2).mean(dim=(1, 2, 3)) / (
                            2.0 * model_prior_std**2
                        )
                    else:
                        model_value = torch.zeros_like(data_value)
                    per_sample.append(data_value + model_value)
                non_kl_objectives.append(torch.cat(per_sample))

            candidate_weights = sorted(
                {
                    0.0,
                    0.01,
                    0.025,
                    0.05,
                    0.1,
                    0.2,
                    0.35,
                    0.5,
                    0.75,
                    1.0,
                    min(max(float(optimized_weight), 0.0), 1.0),
                }
            )
            best_weight = 0.0
            best_objective = float("inf")
            zero_weight_objective = float("nan")
            for candidate_weight in candidate_weights:
                weight = torch.as_tensor(candidate_weight, device=device, dtype=base.dtype)
                weights = torch.cat(((1.0 - weight) * old_weights, weight[None]))
                objective = torch.zeros((), device=device)
                for component_index, z in enumerate(latent_samples):
                    log_p = log_normal_diag(z, prior_mean, prior_log_std)
                    component_logs = [
                        torch.log(weights[index].clamp_min(1.0e-8))
                        + log_normal_diag(z, stage_means[index], stage_log_stds[index])
                        for index in range(len(stage_means))
                    ]
                    log_q = torch.logsumexp(torch.stack(component_logs, dim=0), dim=0)
                    sample_objective = non_kl_objectives[component_index] + float(kl_weight) * (log_q - log_p)
                    objective = objective + weights[component_index] * sample_objective.mean()
                score = float(objective.cpu())
                if candidate_weight == 0.0:
                    zero_weight_objective = score
                if score < best_objective:
                    best_weight = candidate_weight
                    best_objective = score
        return float(best_weight), float(best_objective), float(zero_weight_objective - best_objective)

    # Greedy BVI fits one component and then freezes the existing mixture.
    for stage in range(components):
        new_mean = nn.Parameter(prior_mean + 0.05 * prior_std * torch.randn(latent_dim, device=device))
        new_log_std = nn.Parameter(prior_log_std + float(initial_log_std))
        params: List[nn.Parameter] = [new_mean, new_log_std]
        raw_new_weight = None
        if stage > 0:
            initial_weight = min(max(float(initial_component_weight), 1.0e-4), 1.0 - 1.0e-4)
            raw_new_weight = nn.Parameter(
                torch.tensor(math.log(initial_weight / (1.0 - initial_weight)), device=device)
            )
            opt = torch.optim.Adam(
                [
                    {"params": params, "lr": float(learning_rate)},
                    {"params": [raw_new_weight], "lr": float(component_weight_learning_rate)},
                ]
            )
        else:
            opt = torch.optim.Adam(params, lr=float(learning_rate))
        best_stage_loss = float("inf")
        best_stage_mean: torch.Tensor | None = None
        best_stage_log_std: torch.Tensor | None = None
        best_stage_raw_weight: torch.Tensor | None = None

        for step in range(1, steps + 1):
            stage_means = frozen_means + [new_mean]
            stage_log_stds = frozen_log_stds + [new_log_std]

            def current_stage_weights() -> torch.Tensor:
                if raw_new_weight is None:
                    return torch.ones(1, device=device)
                current_new_weight = torch.sigmoid(raw_new_weight)
                return torch.cat(
                    ((1.0 - current_new_weight) * mixture_weights, current_new_weight[None])
                )

            eps_by_component = []
            for _ in stage_means:
                if antithetic_samples and samples_per_component >= 2:
                    half = (int(samples_per_component) + 1) // 2
                    positive = torch.randn(half, latent_dim, device=device)
                    eps = torch.cat((positive, -positive), dim=0)[: int(samples_per_component)]
                else:
                    eps = torch.randn(samples_per_component, latent_dim, device=device)
                eps_by_component.append(eps)

            if streaming_backward:
                opt.zero_grad()
                detached_data = 0.0
                detached_model = 0.0
                detached_kl = 0.0
                chunk_size = max(1, int(forward_batch_size or samples_per_component))

                # Recompute each chunk before backward so Deepwave wavefields are released immediately.
                for component_index, (mean_k, log_std_k, eps_all) in enumerate(
                    zip(stage_means, stage_log_stds, eps_by_component)
                ):
                    for eps_chunk in eps_all.split(chunk_size, dim=0):
                        weights = current_stage_weights()
                        z_chunk = mean_k + log_std_k.exp() * eps_chunk
                        residual = torch.einsum("sd,dhw->shw", z_chunk, basis)[:, None]
                        candidates = torch.clamp(base + residual_scale * residual, 0.0, 1.0)
                        prediction = forward(candidates)
                        data_chunk = ((prediction - obs) ** 2).mean(dim=(1, 2, 3)) / (2.0 * noise_std**2)
                        if model_prior_std > 0.0:
                            model_chunk = ((candidates - model_prior_center) ** 2).mean(dim=(1, 2, 3)) / (
                                2.0 * model_prior_std**2
                            )
                        else:
                            model_chunk = torch.zeros_like(data_chunk)
                        coefficient = weights[component_index] / float(samples_per_component)
                        chunk_loss = coefficient * (data_chunk + model_chunk).sum()
                        chunk_loss.backward()
                        detached_data += float((coefficient.detach() * data_chunk.detach().sum()).cpu())
                        detached_model += float((coefficient.detach() * model_chunk.detach().sum()).cpu())

                # The KL graph is cheap; recomputing latent samples avoids retaining any wavefield graph.
                for component_index, (mean_k, log_std_k, eps_all) in enumerate(
                    zip(stage_means, stage_log_stds, eps_by_component)
                ):
                    for eps_chunk in eps_all.split(chunk_size, dim=0):
                        weights = current_stage_weights()
                        z_chunk = mean_k + log_std_k.exp() * eps_chunk
                        log_p = log_normal_diag(z_chunk, prior_mean, prior_log_std)
                        comp_logs = [
                            torch.log(weights[k].clamp_min(1e-8))
                            + log_normal_diag(z_chunk, stage_means[k], stage_log_stds[k])
                            for k in range(stage + 1)
                        ]
                        log_q = torch.logsumexp(torch.stack(comp_logs, dim=0), dim=0)
                        coefficient = weights[component_index] / float(samples_per_component)
                        kl_chunk = coefficient * float(kl_weight) * (log_q - log_p).sum()
                        kl_chunk.backward()
                        detached_kl += float(kl_chunk.detach().cpu())

                current_loss = detached_data + detached_model + detached_kl
                if current_loss < best_stage_loss:
                    best_stage_loss = current_loss
                    best_stage_mean = new_mean.detach().clone()
                    best_stage_log_std = new_log_std.detach().clone()
                    best_stage_raw_weight = (
                        raw_new_weight.detach().clone() if raw_new_weight is not None else None
                    )
                if step < steps:
                    opt.step()
                final_loss = current_loss
                weighted_data_value = detached_data
            else:
                stage_weights = current_stage_weights()
                samples_by_component = [
                    mean_k + log_std_k.exp() * eps
                    for mean_k, log_std_k, eps in zip(stage_means, stage_log_stds, eps_by_component)
                ]
                flat_z = torch.cat(samples_by_component, dim=0)
                residual = torch.einsum("sd,dhw->shw", flat_z, basis)[:, None]
                candidates = torch.clamp(base + residual_scale * residual, 0.0, 1.0)
                if forward_batch_size is not None and int(forward_batch_size) > 0:
                    pred_obs = torch.cat(
                        [forward(chunk) for chunk in candidates.split(int(forward_batch_size), dim=0)], dim=0
                    )
                else:
                    pred_obs = forward(candidates)
                data_loss = ((pred_obs - obs) ** 2).mean(dim=(1, 2, 3)) / (2.0 * noise_std**2)
                if model_prior_std > 0.0:
                    model_prior_loss = ((candidates - model_prior_center) ** 2).mean(dim=(1, 2, 3)) / (
                        2.0 * model_prior_std**2
                    )
                else:
                    model_prior_loss = torch.zeros_like(data_loss)

                log_p = log_normal_diag(flat_z, prior_mean, prior_log_std)
                comp_logs = [
                    torch.log(stage_weights[k].clamp_min(1e-8))
                    + log_normal_diag(flat_z, stage_means[k], stage_log_stds[k])
                    for k in range(stage + 1)
                ]
                log_q = torch.logsumexp(torch.stack(comp_logs, dim=0), dim=0)
                sample_objective = data_loss + model_prior_loss + kl_weight * (log_q - log_p)
                component_objectives = torch.stack(
                    [chunk.mean() for chunk in sample_objective.split(samples_per_component)]
                )
                loss = torch.sum(stage_weights * component_objectives)

                opt.zero_grad()
                loss.backward()
                current_loss = float(loss.detach().cpu())
                if current_loss < best_stage_loss:
                    best_stage_loss = current_loss
                    best_stage_mean = new_mean.detach().clone()
                    best_stage_log_std = new_log_std.detach().clone()
                    best_stage_raw_weight = (
                        raw_new_weight.detach().clone() if raw_new_weight is not None else None
                    )
                if step < steps:
                    opt.step()
                final_loss = current_loss
                weighted_data_value = float(
                    torch.sum(
                        stage_weights.detach()
                        * torch.stack([chunk.mean() for chunk in data_loss.split(samples_per_component)])
                    ).detach().cpu()
                )
            if step == 1 or step % max(1, steps // 5) == 0:
                print(
                    f"BVI stage {stage + 1}/{components} step {step:04d}/{steps}: "
                    f"objective={final_loss:.5f} data={weighted_data_value:.5f}"
                )

        if best_stage_mean is None or best_stage_log_std is None:
            raise RuntimeError("BVI stage did not produce a finite objective")
        new_mean.data.copy_(best_stage_mean)
        new_log_std.data.copy_(best_stage_log_std)
        if raw_new_weight is not None and best_stage_raw_weight is not None:
            raw_new_weight.data.copy_(best_stage_raw_weight)

        frozen_means.append(new_mean.detach().clone())
        frozen_log_stds.append(new_log_std.detach().clone())
        if raw_new_weight is None:
            mixture_weights = torch.ones(1, device=device)
            selected_new_weight = 1.0
            selected_objective = best_stage_loss
            selected_gain = 0.0
        else:
            optimized_weight = float(torch.sigmoid(raw_new_weight.detach()).cpu())
            selected_new_weight, selected_objective, selected_gain = line_search_new_weight(
                frozen_means,
                frozen_log_stds,
                mixture_weights,
                optimized_weight,
            )
            new_weight = torch.as_tensor(selected_new_weight, device=device, dtype=base.dtype)
            mixture_weights = torch.cat(((1.0 - new_weight) * mixture_weights, new_weight[None]))
        stage_new_weights.append(float(selected_new_weight))
        stage_line_search_objectives.append(float(selected_objective))
        stage_line_search_gains.append(float(selected_gain))
        final_loss = float(selected_objective)

    means = torch.stack(frozen_means)
    log_stds = torch.stack(frozen_log_stds)
    print(f"BVI mixture weights: {[round(float(w), 4) for w in mixture_weights.cpu()]}")

    with torch.no_grad():
        n_samples = int(posterior_sample_count or max(components * samples_per_component * 8, 128))
        component_ids = torch.multinomial(mixture_weights, n_samples, replacement=True)
        eps = torch.randn(n_samples, latent_dim, device=device)
        flat_z = means[component_ids] + log_stds[component_ids].exp() * eps
        residual = torch.einsum("sd,dhw->shw", flat_z, basis)[:, None]
        samples = torch.clamp(base + residual_scale * residual, 0.0, 1.0)
        post_mean = samples.mean(dim=0, keepdim=True)
        post_std = samples.std(dim=0, keepdim=True)
        event_map = (samples > interrogation_threshold).float().mean(dim=0, keepdim=True)
        high_area = (samples > interrogation_threshold).float().mean(dim=(1, 2, 3))
        h, w = samples.shape[-2:]
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
        central_hit = (samples[:, :, y0:y1, x0:x1].amax(dim=(1, 2, 3)) > interrogation_threshold).float()
        nn_data = torch.sqrt(F.mse_loss(forward(base), obs)).item()
        bvi_data = torch.sqrt(F.mse_loss(forward(post_mean.to(device)), obs)).item()
        if truth is not None:
            nn_rmse = torch.sqrt(F.mse_loss(base, truth)).item()
            bvi_rmse = torch.sqrt(F.mse_loss(post_mean, truth)).item()
            nn_mae = F.l1_loss(base, truth).item()
            bvi_mae = F.l1_loss(post_mean, truth).item()
            coverage_1sigma = ((truth >= post_mean - post_std) & (truth <= post_mean + post_std)).float().mean().item()
            coverage_2sigma = ((truth >= post_mean - 2.0 * post_std) & (truth <= post_mean + 2.0 * post_std)).float().mean().item()
            safe_std = post_std.clamp_min(1e-4)
            gaussian_nll = (
                0.5
                * (((truth - post_mean) / safe_std) ** 2 + 2.0 * torch.log(safe_std) + math.log(2.0 * math.pi))
            ).mean().item()
        else:
            nn_rmse = float("nan")
            bvi_rmse = float("nan")
            nn_mae = float("nan")
            bvi_mae = float("nan")
            coverage_1sigma = float("nan")
            coverage_2sigma = float("nan")
            gaussian_nll = float("nan")
        interval_width_2sigma = (4.0 * post_std).mean().item()
        result = BVIResult(
            rmse_nn=nn_rmse,
            rmse_bvi_mean=bvi_rmse,
            mae_nn=nn_mae,
            mae_bvi_mean=bvi_mae,
            coverage_1sigma=coverage_1sigma,
            coverage_2sigma=coverage_2sigma,
            interval_width_2sigma=interval_width_2sigma,
            gaussian_nll=gaussian_nll,
            mean_std=post_std.mean().item(),
            final_loss=final_loss,
            data_misfit_nn=nn_data,
            data_misfit_bvi_mean=bvi_data,
            high_eps_threshold=interrogation_threshold,
            high_eps_area_mean=high_area.mean().item(),
            high_eps_area_std=high_area.std().item(),
            central_high_eps_prob=central_hit.mean().item(),
            max_high_eps_prob=event_map.max().item(),
            mixture_effective_components=float(
                torch.exp(-(mixture_weights * torch.log(mixture_weights.clamp_min(1e-8))).sum()).cpu()
            ),
            mixture_max_weight=float(mixture_weights.max().cpu()),
            stage_new_weights=stage_new_weights,
            stage_line_search_objectives=stage_line_search_objectives,
            stage_line_search_gains=stage_line_search_gains,
        )
    payload = (result, post_mean.cpu(), post_std.cpu(), event_map.cpu())
    if return_samples:
        return (*payload, samples.cpu())
    return payload


def save_figure(
    out_path: Path,
    obs: torch.Tensor,
    true_model: torch.Tensor,
    nn_pred: torch.Tensor,
    bvi_mean: torch.Tensor,
    bvi_std: torch.Tensor,
    event_prob: torch.Tensor,
    interrogation_threshold: float,
) -> None:
    obs_np = obs.squeeze().cpu().numpy()
    true_np = true_model.squeeze().cpu().numpy()
    nn_np = nn_pred.squeeze().cpu().numpy()
    mean_np = bvi_mean.squeeze().cpu().numpy()
    std_np = bvi_std.squeeze().cpu().numpy()
    prob_np = event_prob.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    panels = [
        (obs_np, "Synthetic B-scan", "seismic"),
        (true_np, "True permittivity", "viridis"),
        (nn_np, "E2E CNN mean", "viridis"),
        (mean_np, "BVI posterior mean", "viridis"),
        (std_np, "BVI posterior std.", "magma"),
        (prob_np, f"P(m>{interrogation_threshold:.2f})", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(img, cmap=cmap, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("experiments/bvi_e2e/datasets/deepwave_ljysh_ygj_256/manifest.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bvi_e2e/results"))
    parser.add_argument("--model-count", type=int, default=1000)
    parser.add_argument("--train-count", type=int, default=700)
    parser.add_argument("--val-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--forward-checkpoint-in", type=Path, default=None)
    parser.add_argument("--forward-epochs", type=int, default=20)
    parser.add_argument("--forward-lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--bvi-steps", type=int, default=180)
    parser.add_argument("--samples-per-component", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.04)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--model-prior-std", type=float, default=0.0)
    parser.add_argument("--interrogation-threshold", type=float, default=0.35)
    parser.add_argument("--basis-type", choices=["cosine", "mixed"], default="cosine")
    parser.add_argument("--bvi-cases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    total = args.train_count + args.val_count + args.test_count
    if not args.dataset_manifest.exists():
        raise FileNotFoundError(
            f"Missing Deepwave dataset manifest {args.dataset_manifest}. "
            "Run build_deepwave_dataset.py first."
        )
    models_t, obs_t, clean_obs_t, model_names, dataset_manifest = load_deepwave_dataset(
        args.dataset_manifest, count=max(args.model_count, total), return_clean=True
    )
    if len(models_t) < total:
        raise ValueError(f"Dataset has {len(models_t)} samples, but split requires {total}")
    if tuple(models_t.shape[-2:]) != (256, 256):
        raise ValueError(f"Expected 256 x 256 models, got {tuple(models_t.shape[-2:])}")
    args.image_size = 256
    args.n_time, args.n_traces = map(int, obs_t.shape[-2:])

    permutation = torch.randperm(len(models_t), generator=torch.Generator().manual_seed(args.seed))
    models_t = models_t[permutation]
    obs_t = obs_t[permutation]
    clean_obs_t = clean_obs_t[permutation]
    model_names = [model_names[index] for index in permutation.tolist()]

    train_x = obs_t[: args.train_count]
    train_y = models_t[: args.train_count]
    val_x = obs_t[args.train_count : args.train_count + args.val_count]
    val_y = models_t[args.train_count : args.train_count + args.val_count]
    test_x = obs_t[args.train_count + args.val_count : total]
    test_y = models_t[args.train_count + args.val_count : total]
    clean_train_x = clean_obs_t[: args.train_count]
    clean_val_x = clean_obs_t[args.train_count : args.train_count + args.val_count]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    forward_train_loader = DataLoader(
        TensorDataset(clean_train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    forward = DeepwaveForwardSurrogate(args.n_time, args.n_traces).to(device)
    if args.forward_checkpoint_in is not None:
        if not args.forward_checkpoint_in.exists():
            raise FileNotFoundError(f"Missing forward checkpoint {args.forward_checkpoint_in}")
        forward.load_state_dict(torch.load(args.forward_checkpoint_in, map_location=device))
        forward_history = [{"loaded_checkpoint": str(args.forward_checkpoint_in)}]
    else:
        forward_history = train_forward_surrogate(
            forward,
            forward_train_loader,
            (val_y, clean_val_x),
            args.forward_epochs,
            args.forward_lr,
            device,
        )
    forward.eval()
    for parameter in forward.parameters():
        parameter.requires_grad_(False)

    net = TinyInversionNet(args.image_size).to(device)
    if args.checkpoint_in is not None:
        if not args.checkpoint_in.exists():
            raise FileNotFoundError(f"Missing checkpoint {args.checkpoint_in}")
        net.load_state_dict(torch.load(args.checkpoint_in, map_location=device))
        history = [{"loaded_checkpoint": str(args.checkpoint_in)}]
        print(f"loaded checkpoint: {args.checkpoint_in}")
    else:
        history = train_network(net, train_loader, (val_x, val_y), args.epochs, args.lr, device)
    test_pred = predict_in_batches(net, test_x, args.batch_size, device)
    test_rmse = torch.sqrt(F.mse_loss(test_pred, test_y)).item()
    test_mae = F.l1_loss(test_pred, test_y).item()
    print(f"test deterministic: rmse={test_rmse:.5f} mae={test_mae:.5f}")

    bvi_case_results = []
    fig_path = args.out_dir / "e2e_bvi_gpr_result.png"
    n_bvi_cases = min(args.bvi_cases, len(test_x))
    first_case_payload = None
    for idx in range(n_bvi_cases):
        print(f"Running BVI case {idx + 1}/{n_bvi_cases}")
        obs_one = test_x[idx : idx + 1]
        true_one = test_y[idx : idx + 1]
        nn_one = test_pred[idx : idx + 1]
        bvi_result, bvi_mean, bvi_std, event_prob = run_bvi(
            forward=forward,
            observation=obs_one,
            true_model=true_one,
            nn_mean=nn_one,
            latent_dim=args.latent_dim,
            components=args.components,
            steps=args.bvi_steps,
            samples_per_component=args.samples_per_component,
            residual_scale=args.residual_scale,
            noise_std=args.noise_std,
            kl_weight=args.kl_weight,
            model_prior_std=args.model_prior_std,
            interrogation_threshold=args.interrogation_threshold,
            basis_type=args.basis_type,
            device=device,
        )
        bvi_case_results.append(asdict(bvi_result))
        case_fig = args.out_dir / f"e2e_bvi_gpr_case_{idx + 1:02d}.png"
        save_figure(case_fig, obs_one, true_one, nn_one, bvi_mean, bvi_std, event_prob, args.interrogation_threshold)
        np.savez_compressed(
            args.out_dir / f"e2e_bvi_gpr_case_{idx + 1:02d}_arrays.npz",
            observation=obs_one.numpy(),
            true_model=true_one.numpy(),
            nn_prediction=nn_one.numpy(),
            bvi_mean=bvi_mean.numpy(),
            bvi_std=bvi_std.numpy(),
            high_eps_probability=event_prob.numpy(),
        )
        if idx == 0:
            save_figure(fig_path, obs_one, true_one, nn_one, bvi_mean, bvi_std, event_prob, args.interrogation_threshold)
            first_case_payload = (obs_one, true_one, nn_one, bvi_mean, bvi_std, event_prob)

    bvi_aggregate = {}
    if bvi_case_results:
        for key in bvi_case_results[0].keys():
            vals = np.array([r[key] for r in bvi_case_results if r[key] is not None and not np.isnan(r[key])], dtype=np.float64)
            if vals.size:
                bvi_aggregate[f"{key}_mean"] = float(vals.mean())
                bvi_aggregate[f"{key}_std"] = float(vals.std())
    torch.save(net.state_dict(), args.out_dir / "tiny_inversion_net.pt")
    torch.save(forward.state_dict(), args.out_dir / "deepwave_forward_surrogate.pt")
    if first_case_payload is not None:
        first_obs, first_true, first_nn, first_mean, first_std, first_event = first_case_payload
        np.savez_compressed(
            args.out_dir / "e2e_bvi_gpr_arrays.npz",
            observation=first_obs.numpy(),
            true_model=first_true.numpy(),
            nn_prediction=first_nn.numpy(),
            bvi_mean=first_mean.numpy(),
            bvi_std=first_std.numpy(),
            high_eps_probability=first_event.numpy(),
        )

    summary = {
        "config": json_safe(
            vars(args)
            | {
                "device": str(device),
                "dataset_manifest": str(args.dataset_manifest),
                "out_dir": str(args.out_dir),
                "bvi_algorithm": "sequential_greedy",
                "bvi_steps_per_stage": args.bvi_steps,
            }
        ),
        "history": history,
        "forward_surrogate_history": forward_history,
        "forward_surrogate_validation_rmse": forward_history[-1].get("val_rmse") if forward_history else None,
        "dataset": {
            "format": dataset_manifest["format"],
            "profile_name": dataset_manifest["profile_name"],
            "physics": dataset_manifest["physics"],
            "model_shape": dataset_manifest["model_shape"],
            "bscan_shape": dataset_manifest["bscan_shape"],
            "split_model_names": {
                "train": model_names[: args.train_count],
                "validation": model_names[args.train_count : args.train_count + args.val_count],
                "test": model_names[args.train_count + args.val_count : total],
            },
        },
        "test_deterministic": {"rmse": test_rmse, "mae": test_mae},
        "bvi_cases": bvi_case_results,
        "bvi_aggregate": bvi_aggregate,
        "figure": str(fig_path),
        "notes": (
            "B-scans were generated offline with Deepwave's scalar wave propagator under the "
            "acquisition contract stored in the dataset manifest. The inverse network consumes "
            "the configured noisy channel, while the differentiable forward surrogate is trained "
            "against the clean Deepwave channel. BVI uses that differentiable "
            "surrogate trained on those Deepwave records; report its held-out validation error. "
            "Deepwave scalar remains an approximation rather than a Maxwell TM solver."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["bvi_aggregate"], indent=2))
    print(f"saved results to {args.out_dir}")


if __name__ == "__main__":
    main()
