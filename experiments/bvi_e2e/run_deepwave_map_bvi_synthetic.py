"""MAP-centered trust-region Neural-BVI probe with differentiable Deepwave.

This is an experimental accuracy probe. It keeps the deterministic U-Net
inversion as the prior mean, optimizes a low-dimensional posterior center with
the differentiable Deepwave likelihood, then samples a local Gaussian posterior
around that center. Hyperparameters must be selected on validation before any
held-out test claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from e2e_bvi_gpr import make_residual_basis
from publication_data import farthest_point_subset, load_protocol, synthesize_noise_view
from publication_metrics import deterministic_metrics, uq_metrics
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, set_seed, write_csv
from publication_uq import save_posterior_arrays, uq_cases


HERE = Path(__file__).resolve().parent


@dataclass
class MapBVIResult:
    rmse_nn: float | None
    rmse_bvi_mean: float | None
    data_misfit_nn: float
    data_misfit_bvi_mean: float
    final_loss: float
    final_data_loss: float
    final_model_prior_loss: float
    final_latent_prior_loss: float
    z_norm: float
    z_deviation_norm: float
    latent_prior_mean_norm: float
    latent_prior_std_mean: float
    correction_rms: float
    model_prior_center_delta_rms: float
    mean_std: float
    posterior_latent_std_mean: float
    posterior_latent_std_min: float
    posterior_latent_std_max: float
    posterior_precision_condition: float
    posterior_precision_data_trace: float


def smooth_bscan(data: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return data
    if kernel_size % 2 == 0:
        raise ValueError("data_smooth_kernel must be odd so the B-scan shape is preserved")
    pad = kernel_size // 2
    return F.avg_pool2d(data, kernel_size=kernel_size, stride=1, padding=pad, count_include_pad=False)


def parse_snrs(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def read_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def record_key(record: dict[str, Any]) -> tuple[int, str]:
    return int(record["global_index"]), str(record["view"])


def fixed_split_cases(protocol: dict[str, Any], artifacts: dict[str, Any], split: str) -> list[dict[str, Any]]:
    fixed = artifacts["fixed"][split]
    clean = fixed["views"]["clean"]
    cases = []
    for local_index in range(len(fixed["indices"])):
        for snr in protocol["uq"]["snrs_db"]:
            view = f"mixed_{float(snr):g}db"
            cases.append(
                {
                    "split": split,
                    "local_index": int(local_index),
                    "global_index": int(fixed["indices"][local_index]),
                    "model_name": fixed["model_names"][local_index],
                    "view": view,
                    "snr_db": float(snr),
                    "observation": fixed["views"][view][local_index : local_index + 1],
                    "clean_observation": clean[local_index : local_index + 1],
                    "truth": artifacts["models"][fixed["indices"][local_index] : fixed["indices"][local_index] + 1],
                }
            )
    return cases


def train_split_cases(protocol: dict[str, Any], artifacts: dict[str, Any], source: str) -> list[dict[str, Any]]:
    train_indices = list(artifacts["splits"]["train"])
    if source == "uq":
        subset_count = min(int(protocol["uq"].get("validation_models", 10)), len(train_indices))
        local_indices = farthest_point_subset(artifacts["models"][train_indices], subset_count)
        selected_indices = [int(train_indices[index]) for index in local_indices]
    elif source == "all":
        selected_indices = [int(index) for index in train_indices]
    else:
        raise ValueError(f"Unknown case source {source!r}")

    cases = []
    residual_bank = artifacts["residual_bank"]
    noise_cfg = protocol["noise"]
    for local_index, global_index in enumerate(selected_indices):
        clean = artifacts["clean_bscans"][global_index]
        truth = artifacts["models"][global_index : global_index + 1]
        for snr in protocol["uq"]["snrs_db"]:
            view_name = f"mixed_{float(snr):g}db"
            noise_seed = int(noise_cfg["evaluation_seed"]) + int(global_index) * 1009 + int(float(snr) * 10)
            view, _ = synthesize_noise_view(
                clean,
                residual_bank,
                noise_seed,
                float(snr),
                "mixed",
                float(noise_cfg["field_weight"]),
                float(noise_cfg["gaussian_weight"]),
            )
            cases.append(
                {
                    "split": "train",
                    "local_index": int(local_index),
                    "global_index": int(global_index),
                    "model_name": artifacts["model_names"][global_index],
                    "view": view_name,
                    "snr_db": float(snr),
                    "observation": view[None],
                    "clean_observation": clean[None],
                    "truth": truth,
                }
            )
    return cases


def select_cases(protocol: dict[str, Any], artifacts: dict[str, Any], split: str, source: str) -> list[dict[str, Any]]:
    if split == "train":
        return train_split_cases(protocol, artifacts, source)
    if source == "uq":
        return list(uq_cases(protocol, artifacts, split))
    if source == "all":
        return fixed_split_cases(protocol, artifacts, split)
    raise ValueError(f"Unknown case source {source!r}")


def normalize_basis_atom(atom: torch.Tensor) -> torch.Tensor:
    atom = atom - atom.mean()
    return atom / (atom.std() + 1.0e-6)


def cache_matches(payload: dict[str, Any], expected: dict[str, Any]) -> bool:
    meta = payload.get("meta", {})
    return all(meta.get(key) == value for key, value in expected.items())


def fit_basis_latent_raw_stats(residuals: torch.Tensor, basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit residual coefficients in model units for an empirical latent prior."""

    target = residuals[:, 0].flatten(1).float()
    atoms = basis.flatten(1).float()
    gram = atoms @ atoms.T
    ridge = 1.0e-4 * torch.trace(gram) / float(max(1, atoms.shape[0]))
    eye = torch.eye(atoms.shape[0], dtype=gram.dtype, device=gram.device)
    rhs = target @ atoms.T
    coeffs = torch.linalg.solve(gram + ridge * eye, rhs.T).T
    std = coeffs.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    return coeffs.mean(dim=0).cpu(), std.cpu()


def build_error_pca_basis(
    *,
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    model: torch.nn.Module,
    latent_dim: int,
    include_mean: bool,
    train_count: int,
    snrs: list[float],
    seed: int,
    cache_path: Path,
    device: torch.device,
    batch_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Learn residual basis atoms from training-set NN inversion errors.

    The basis uses only training labels and the fixed deterministic inversion
    checkpoint, so validation/test labels remain untouched for selection and
    reporting. Atoms are normalized like the hand-written cosine/RBF basis; the
    residual scale therefore keeps the same interpretation.
    """

    expected_meta = {
        "kind": "training_error_pca_basis_v2",
        "protocol_hash": protocol["protocol_hash"],
        "latent_dim": int(latent_dim),
        "include_mean": bool(include_mean),
        "train_count": int(train_count),
        "snrs": [float(snr) for snr in snrs],
        "seed": int(seed),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(payload, dict)
            and "basis" in payload
            and "latent_raw_mean" in payload
            and "latent_raw_std" in payload
            and cache_matches(payload, expected_meta)
        ):
            print(f"Loaded cached training-error PCA basis: {cache_path}", flush=True)
            return {
                "basis": payload["basis"].float().to(device),
                "latent_raw_mean": payload["latent_raw_mean"].float().to(device),
                "latent_raw_std": payload["latent_raw_std"].float().to(device),
            }

    train_indices = list(artifacts["splits"]["train"])[: int(train_count)]
    if not train_indices:
        raise ValueError("Cannot build an error PCA basis without training indices")
    residual_bank = artifacts["residual_bank"]
    observations: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []
    noise_cfg = protocol["noise"]
    for global_index in train_indices:
        clean = artifacts["clean_bscans"][int(global_index)]
        truth = artifacts["models"][int(global_index)]
        for snr in snrs:
            noise_seed = int(noise_cfg["evaluation_seed"]) + 9_000_000 + int(global_index) * 1009 + int(float(snr) * 10)
            view, _ = synthesize_noise_view(
                clean,
                residual_bank,
                noise_seed,
                float(snr),
                "mixed",
                float(noise_cfg["field_weight"]),
                float(noise_cfg["gaussian_weight"]),
            )
            observations.append(view)
            truths.append(truth)

    obs_tensor = torch.stack(observations)
    truth_tensor = torch.stack(truths)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(obs_tensor), int(batch_size)):
            batch = obs_tensor[start : start + int(batch_size)].to(device)
            predictions.append(model(batch).cpu())
    prediction_tensor = torch.cat(predictions, dim=0)
    residuals = truth_tensor - prediction_tensor

    atoms: list[torch.Tensor] = []
    if include_mean:
        atoms.append(normalize_basis_atom(residuals.mean(dim=0)[0]))

    remaining = int(latent_dim) - len(atoms)
    if remaining > 0:
        flat = residuals.flatten(1).float()
        centered = flat - flat.mean(dim=0, keepdim=True)
        if centered.shape[0] < 2:
            raise ValueError("At least two training residuals are needed for PCA")
        gram = centered @ centered.T / float(centered.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(gram)
        order = torch.argsort(eigvals, descending=True)
        h, w = residuals.shape[-2:]
        for item in order[:remaining]:
            value = eigvals[item].clamp_min(1.0e-12)
            component = centered.T @ eigvecs[:, item] / torch.sqrt(value * float(centered.shape[0] - 1))
            atoms.append(normalize_basis_atom(component.reshape(h, w)))

    basis = torch.stack(atoms[: int(latent_dim)], dim=0).float().cpu()
    latent_raw_mean, latent_raw_std = fit_basis_latent_raw_stats(residuals, basis)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "basis": basis,
            "latent_raw_mean": latent_raw_mean,
            "latent_raw_std": latent_raw_std,
            "meta": expected_meta,
        },
        cache_path,
    )
    print(f"Built training-error PCA basis: {cache_path}", flush=True)
    return {
        "basis": basis.to(device),
        "latent_raw_mean": latent_raw_mean.to(device),
        "latent_raw_std": latent_raw_std.to(device),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "mean_epsilon_rmse": None,
            "mean_rmse_delta_bvi_minus_nn": None,
            "rmse_improved_count": 0,
            "mean_data_misfit_delta_bvi_minus_nn": None,
            "data_misfit_improved_count": 0,
        }
    rmse_delta = [float(record["rmse_delta_bvi_minus_nn"]) for record in records]
    data_delta = [float(record["data_misfit_delta_bvi_minus_nn"]) for record in records]
    return {
        "record_count": len(records),
        "mean_epsilon_rmse": float(np.mean([float(record["epsilon_rmse"]) for record in records])),
        "mean_rmse_delta_bvi_minus_nn": float(np.mean(rmse_delta)),
        "rmse_improved_count": int(sum(value < 0.0 for value in rmse_delta)),
        "mean_data_misfit_delta_bvi_minus_nn": float(np.mean(data_delta)),
        "data_misfit_improved_count": int(sum(value < 0.0 for value in data_delta)),
    }


def candidate_from_latent(
    base: torch.Tensor,
    basis: torch.Tensor,
    z: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    residual = torch.einsum("d,dhw->hw", z, basis)[None, None]
    return torch.clamp(base + float(residual_scale) * residual, 0.0, 1.0)


def prediction_for_loss(forward: torch.nn.Module, candidate: torch.Tensor, data_loss: str, kernel_size: int) -> torch.Tensor:
    prediction = forward(candidate)
    return smooth_bscan(prediction, kernel_size) if data_loss == "smooth_mse" else prediction


def estimate_laplace_precision(
    *,
    forward: torch.nn.Module,
    base: torch.Tensor,
    basis: torch.Tensor,
    z: torch.Tensor,
    residual_scale: float,
    noise_std: torch.Tensor,
    model_prior_std: float,
    latent_prior_weight: float,
    z_prior_std: torch.Tensor,
    data_loss: str,
    data_smooth_kernel: int,
    fd_eps: float,
    precision_damping: float,
) -> tuple[torch.Tensor, float, float]:
    """Approximate MAP posterior precision in latent space.

    The data block is a finite-difference Gauss-Newton/Fisher term around the
    optimized latent point. It makes the posterior spread depend on the local
    Deepwave observation sensitivity instead of a fixed isotropic jitter.
    """

    latent_dim = int(z.numel())
    fd_eps = float(fd_eps)
    if fd_eps <= 0.0:
        raise ValueError("laplace_fd_eps must be positive")

    pred_columns: list[torch.Tensor] = []
    model_columns: list[torch.Tensor] = []
    with torch.no_grad():
        for dim in range(latent_dim):
            step = fd_eps * max(1.0, float(abs(z[dim]).detach().cpu()))
            delta = torch.zeros_like(z)
            delta[dim] = step
            plus = candidate_from_latent(base, basis, z + delta, residual_scale)
            minus = candidate_from_latent(base, basis, z - delta, residual_scale)
            pred_plus = prediction_for_loss(forward, plus, data_loss, data_smooth_kernel)
            pred_minus = prediction_for_loss(forward, minus, data_loss, data_smooth_kernel)
            pred_columns.append(((pred_plus - pred_minus) / (2.0 * step)).reshape(-1).double())
            if model_prior_std > 0.0:
                model_columns.append(((plus - minus) / (2.0 * step)).reshape(-1).double())

    jac = torch.stack(pred_columns, dim=1)
    noise_var = float(noise_std.detach().cpu()) ** 2
    data_precision = (jac.T @ jac) / (max(noise_var, 1.0e-12) * float(jac.shape[0]))

    if model_columns:
        model_jac = torch.stack(model_columns, dim=1)
        model_precision = (model_jac.T @ model_jac) / (float(model_prior_std) ** 2 * float(model_jac.shape[0]))
    else:
        model_precision = torch.zeros_like(data_precision)

    prior_std = z_prior_std.detach().double().clamp_min(1.0e-8)
    prior_diag = float(latent_prior_weight) / (float(latent_dim) * prior_std.square())
    prior_precision = torch.diag(prior_diag)

    precision = data_precision + model_precision + prior_precision
    trace_per_dim = float(torch.trace(precision).detach().cpu()) / float(max(1, latent_dim))
    damping = max(float(precision_damping) * max(trace_per_dim, 1.0e-12), 1.0e-12)
    precision = precision + damping * torch.eye(latent_dim, dtype=precision.dtype, device=precision.device)

    eigvals = torch.linalg.eigvalsh(0.5 * (precision + precision.T)).clamp_min(1.0e-12)
    condition = float((eigvals.max() / eigvals.min()).detach().cpu())
    data_trace = float(torch.trace(data_precision).detach().cpu())
    return precision.float(), condition, data_trace


def posterior_factor_from_precision(
    precision: torch.Tensor,
    sampler: str,
    temperature: float,
    std_min: float,
    std_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    precision = 0.5 * (precision + precision.T)
    temperature = max(float(temperature), 1.0e-12)
    std_min = max(float(std_min), 0.0)
    std_max = max(float(std_max), std_min + 1.0e-12)
    if sampler == "laplace_diag":
        diag_precision = torch.diag(precision).clamp_min(1.0e-12)
        latent_std = torch.sqrt(temperature / diag_precision).clamp(std_min, std_max)
        return torch.diag(latent_std), latent_std
    if sampler == "laplace_full":
        eigvals, eigvecs = torch.linalg.eigh(precision.double())
        latent_std = torch.sqrt(temperature / eigvals.clamp_min(1.0e-12)).clamp(std_min, std_max).float()
        factor = eigvecs.float() @ torch.diag(latent_std)
        return factor, latent_std
    raise ValueError(f"Unknown posterior sampler {sampler!r}")


def sample_images_from_latent_factor(
    *,
    base: torch.Tensor,
    basis: torch.Tensor,
    z: torch.Tensor,
    residual_scale: float,
    factor: torch.Tensor,
    sample_count: int,
    include_map_center_sample: bool = True,
) -> torch.Tensor:
    sample_count = max(int(sample_count), 1)
    eps = torch.randn(sample_count, int(z.numel()), device=z.device)
    flat_z = z.detach()[None] + eps @ factor.to(z.device).T
    sampled_residual = torch.einsum("sd,dhw->shw", flat_z, basis)[:, None]
    samples = torch.clamp(base + float(residual_scale) * sampled_residual, 0.0, 1.0)
    if include_map_center_sample:
        samples[0] = candidate_from_latent(base, basis, z.detach(), residual_scale)[0]
    return samples


def write_summary(
    path: Path,
    status: str,
    root: Path,
    split: str,
    snrs: list[float],
    config: dict[str, Any],
    records: list[dict[str, Any]],
    started: float,
) -> None:
    payload = {
        "status": status,
        "root": str(root),
        "split": split,
        "snrs": snrs,
        "config": config,
        "seconds": time.time() - started,
        "metrics": str(path.with_name("metrics.csv")),
        "aggregate": aggregate(records),
        "records": records,
        "claim_boundary": (
            "MAP-centered trust-region Neural-BVI probe. Hyperparameters and "
            "acceptance rules must be selected on validation before held-out test use."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_map_bvi_case(
    forward: torch.nn.Module,
    observation: torch.Tensor,
    clean_observation: torch.Tensor | None,
    truth: torch.Tensor | None,
    nn_prediction: torch.Tensor,
    *,
    latent_dim: int,
    residual_scale: float,
    model_prior_std: float,
    latent_prior_weight: float,
    steps: int,
    lr: float,
    basis_type: str,
    posterior_latent_std: float,
    posterior_sample_count: int,
    event_threshold: float,
    device: torch.device,
    data_loss: str = "raw_mse",
    data_smooth_kernel: int = 1,
    noise_std_floor: float = 0.01,
    noise_std_override: float | None = None,
    basis_override: torch.Tensor | None = None,
    latent_prior_mean: torch.Tensor | None = None,
    latent_prior_std: torch.Tensor | None = None,
    latent_prior_min_std: float = 0.25,
    latent_prior_max_std: float = 20.0,
    latent_init: str = "zero",
    model_prior_center: str = "nn",
    posterior_sampler: str = "laplace_full",
    laplace_fd_eps: float = 1.0e-2,
    laplace_std_min: float = 1.0e-4,
    laplace_std_max: float = 5.0,
    posterior_temperature: float = 1.0,
    laplace_precision_damping: float = 1.0e-3,
    include_map_center_sample: bool = True,
) -> tuple[MapBVIResult, torch.Tensor, dict[str, float]]:
    basis = (
        basis_override.to(device)
        if basis_override is not None
        else make_residual_basis(nn_prediction.shape[-1], latent_dim, device, basis_type)
    )
    obs = observation.to(device)
    clean = clean_observation.to(device) if clean_observation is not None else None
    base = nn_prediction.to(device).detach()
    target = truth.to(device) if truth is not None else None
    obs_for_loss = smooth_bscan(obs, data_smooth_kernel) if data_loss == "smooth_mse" else obs
    if noise_std_override is not None:
        noise_std = max(float(noise_std_override), float(noise_std_floor))
    elif clean is not None:
        clean_for_loss = smooth_bscan(clean, data_smooth_kernel) if data_loss == "smooth_mse" else clean
        noise = obs_for_loss - clean_for_loss
        noise_std = max(float(torch.sqrt(torch.mean(noise**2))), float(noise_std_floor))
    else:
        raise ValueError("Field cases without a clean observation require noise_std_override")

    if latent_prior_mean is None:
        z_prior_mean = torch.zeros(latent_dim, device=device)
    else:
        z_prior_mean = latent_prior_mean.to(device).float().flatten()
    if latent_prior_std is None:
        z_prior_std = torch.ones(latent_dim, device=device)
    else:
        z_prior_std = latent_prior_std.to(device).float().flatten()
    if z_prior_mean.numel() != int(latent_dim) or z_prior_std.numel() != int(latent_dim):
        raise ValueError("Latent prior tensors must match latent_dim")
    z_prior_std = torch.clamp(
        z_prior_std,
        min=float(latent_prior_min_std),
        max=float(latent_prior_max_std),
    )
    if latent_init == "prior_mean":
        z_initial = z_prior_mean.detach().clone()
    elif latent_init == "zero":
        z_initial = torch.zeros(latent_dim, device=device)
    else:
        raise ValueError(f"Unknown latent_init {latent_init!r}")

    if model_prior_center == "latent_prior_mean":
        prior_residual = torch.einsum("d,dhw->hw", z_prior_mean, basis)[None, None]
        model_center = torch.clamp(base + float(residual_scale) * prior_residual, 0.0, 1.0).detach()
    elif model_prior_center == "nn":
        model_center = base
    else:
        raise ValueError(f"Unknown model_prior_center {model_prior_center!r}")
    model_center_delta = model_center - base

    z = torch.nn.Parameter(z_initial)
    optimizer = torch.optim.Adam([z], lr=float(lr))
    final_loss = torch.tensor(float("nan"), device=device)
    final_data_loss = torch.tensor(float("nan"), device=device)
    final_model_prior_loss = torch.tensor(float("nan"), device=device)
    final_latent_prior_loss = torch.tensor(float("nan"), device=device)

    for step in range(1, int(steps) + 1):
        residual = torch.einsum("d,dhw->hw", z, basis)[None, None]
        candidate = torch.clamp(base + float(residual_scale) * residual, 0.0, 1.0)
        prediction = forward(candidate)
        prediction_for_loss = smooth_bscan(prediction, data_smooth_kernel) if data_loss == "smooth_mse" else prediction
        data_loss_value = F.mse_loss(prediction_for_loss, obs_for_loss) / (2.0 * noise_std**2)
        if model_prior_std > 0.0:
            model_prior_loss = F.mse_loss(candidate, model_center) / (2.0 * float(model_prior_std) ** 2)
        else:
            model_prior_loss = torch.zeros((), device=device)
        latent_prior_loss = 0.5 * ((z - z_prior_mean) / z_prior_std).pow(2).mean()
        loss = data_loss_value + model_prior_loss + float(latent_prior_weight) * latent_prior_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.detach()
        final_data_loss = data_loss_value.detach()
        final_model_prior_loss = model_prior_loss.detach()
        final_latent_prior_loss = latent_prior_loss.detach()
        if step == 1 or step % max(1, int(steps) // 4) == 0:
            print(
                f"MAP-BVI step {step:04d}/{steps}: objective={float(final_loss):.5f} "
                f"data={float(final_data_loss):.5f} prior={float(final_model_prior_loss):.5f}",
                flush=True,
            )

    sampler = str(posterior_sampler)
    with torch.no_grad():
        center = candidate_from_latent(base, basis, z.detach(), residual_scale)

    if sampler == "fixed":
        latent_std = torch.full((int(latent_dim),), float(posterior_latent_std), device=device)
        factor = torch.diag(latent_std)
        posterior_precision_condition = 1.0
        posterior_precision_data_trace = 0.0
    elif sampler in {"laplace_diag", "laplace_full"}:
        precision, posterior_precision_condition, posterior_precision_data_trace = estimate_laplace_precision(
            forward=forward,
            base=base,
            basis=basis,
            z=z.detach(),
            residual_scale=float(residual_scale),
            noise_std=torch.as_tensor(noise_std, device=device),
            model_prior_std=float(model_prior_std),
            latent_prior_weight=float(latent_prior_weight),
            z_prior_std=z_prior_std,
            data_loss=str(data_loss),
            data_smooth_kernel=int(data_smooth_kernel),
            fd_eps=float(laplace_fd_eps),
            precision_damping=float(laplace_precision_damping),
        )
        factor, latent_std = posterior_factor_from_precision(
            precision.to(device),
            sampler,
            temperature=float(posterior_temperature),
            std_min=float(laplace_std_min),
            std_max=float(laplace_std_max),
        )
    else:
        raise ValueError(f"Unknown posterior_sampler {posterior_sampler!r}")

    with torch.no_grad():
        samples = sample_images_from_latent_factor(
            base=base,
            basis=basis,
            z=z.detach(),
            residual_scale=float(residual_scale),
            factor=factor,
            sample_count=int(posterior_sample_count),
            include_map_center_sample=bool(include_map_center_sample),
        )
        post_mean = samples.mean(dim=0, keepdim=True)
        post_std = samples.std(dim=0, keepdim=True, unbiased=False)
        nn_data = torch.sqrt(F.mse_loss(forward(base), obs)).item()
        bvi_data = torch.sqrt(F.mse_loss(forward(post_mean), obs)).item()
        nn_rmse = torch.sqrt(F.mse_loss(base, target)).item() if target is not None else None
        bvi_rmse = torch.sqrt(F.mse_loss(post_mean, target)).item() if target is not None else None
        correction = post_mean - base
        result = MapBVIResult(
            rmse_nn=nn_rmse,
            rmse_bvi_mean=bvi_rmse,
            data_misfit_nn=nn_data,
            data_misfit_bvi_mean=bvi_data,
            final_loss=float(final_loss.cpu()),
            final_data_loss=float(final_data_loss.cpu()),
            final_model_prior_loss=float(final_model_prior_loss.cpu()),
            final_latent_prior_loss=float(final_latent_prior_loss.cpu()),
            z_norm=float(torch.linalg.vector_norm(z.detach()).cpu()),
            z_deviation_norm=float(torch.linalg.vector_norm((z.detach() - z_prior_mean) / z_prior_std).cpu()),
            latent_prior_mean_norm=float(torch.linalg.vector_norm(z_prior_mean).cpu()),
            latent_prior_std_mean=float(z_prior_std.mean().cpu()),
            correction_rms=float(torch.sqrt(torch.mean(correction**2)).cpu()),
            model_prior_center_delta_rms=float(torch.sqrt(torch.mean(model_center_delta**2)).cpu()),
            mean_std=float(post_std.mean().cpu()),
            posterior_latent_std_mean=float(latent_std.mean().detach().cpu()),
            posterior_latent_std_min=float(latent_std.min().detach().cpu()),
            posterior_latent_std_max=float(latent_std.max().detach().cpu()),
            posterior_precision_condition=float(posterior_precision_condition),
            posterior_precision_data_trace=float(posterior_precision_data_trace),
        )
    metrics = uq_metrics(samples.cpu(), truth, event_threshold) if truth is not None else {}
    return result, samples.cpu(), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "publication_la010010" / "full")
    parser.add_argument("--protocol", type=Path, default=HERE / "la010010_protocol.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--case-source", choices=["uq", "all"], default="uq")
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--snrs", default="0,5,10")
    parser.add_argument("--name", default="map_bvi")
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--residual-scale", type=float, default=0.01)
    parser.add_argument("--model-prior-std", type=float, default=0.01)
    parser.add_argument("--latent-prior-weight", type=float, default=0.02)
    parser.add_argument("--latent-prior-mode", choices=["standard", "empirical"], default="standard")
    parser.add_argument("--latent-init", choices=["zero", "prior_mean"], default="zero")
    parser.add_argument("--latent-prior-min-std", type=float, default=0.25)
    parser.add_argument("--latent-prior-max-std", type=float, default=20.0)
    parser.add_argument("--model-prior-center", choices=["nn", "latent_prior_mean"], default="nn")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--basis-type", choices=["cosine", "mixed", "error_pca", "error_pca_mean"], default="cosine")
    parser.add_argument("--pca-train-count", type=int, default=96)
    parser.add_argument("--pca-snrs", default="0,5,10")
    parser.add_argument("--pca-basis-path", type=Path, default=None)
    parser.add_argument("--posterior-sampler", choices=["fixed", "laplace_diag", "laplace_full"], default="laplace_full")
    parser.add_argument("--posterior-latent-std", type=float, default=0.03)
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--laplace-fd-eps", type=float, default=1.0e-2)
    parser.add_argument("--laplace-std-min", type=float, default=1.0e-4)
    parser.add_argument("--laplace-std-max", type=float, default=5.0)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--laplace-precision-damping", type=float, default=1.0e-3)
    parser.add_argument("--data-loss", choices=["raw_mse", "smooth_mse"], default="raw_mse")
    parser.add_argument("--data-smooth-kernel", type=int, default=1)
    parser.add_argument("--noise-std-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = (
        json.loads((root / "protocol.snapshot.json").read_text(encoding="utf-8"))
        if (root / "protocol.snapshot.json").exists()
        else load_protocol(args.protocol.resolve())
    )
    artifacts = torch.load(root / "prepared" / "artifacts.pt", map_location="cpu", weights_only=False)
    if artifacts["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Prepared artifacts do not match the protocol hash")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate, _ = load_surrogate_checkpoint(root / "surrogate" / "best.pt", device)
    seed = int(args.seed if args.seed is not None else protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)
    pca_basis_path = (
        args.pca_basis_path.resolve()
        if args.pca_basis_path is not None
        else root
        / "basis"
        / (
            f"{args.basis_type}_latent{int(args.latent_dim)}_"
            f"train{int(args.pca_train_count)}_snrs{str(args.pca_snrs).replace(',', '-')}_seed{seed}.pt"
        )
    )
    basis_override = None
    latent_prior_mean = None
    latent_prior_std = None
    if str(args.basis_type).startswith("error_pca"):
        basis_payload = build_error_pca_basis(
            protocol=protocol,
            artifacts=artifacts,
            model=model,
            latent_dim=int(args.latent_dim),
            include_mean=str(args.basis_type) == "error_pca_mean",
            train_count=int(args.pca_train_count),
            snrs=parse_snrs(args.pca_snrs),
            seed=seed,
            cache_path=pca_basis_path,
            device=device,
        )
        basis_override = basis_payload["basis"]
        if args.latent_prior_mode == "empirical":
            scale = max(float(args.residual_scale), 1.0e-8)
            latent_prior_mean = basis_payload["latent_raw_mean"] / scale
            latent_prior_std = basis_payload["latent_raw_std"] / scale
    elif args.latent_prior_mode == "empirical":
        raise ValueError("Empirical latent prior requires an error_pca or error_pca_mean basis")
    config = {
        "name": str(args.name),
        "latent_dim": int(args.latent_dim),
        "residual_scale": float(args.residual_scale),
        "model_prior_std": float(args.model_prior_std),
        "latent_prior_weight": float(args.latent_prior_weight),
        "latent_prior_mode": str(args.latent_prior_mode),
        "latent_init": str(args.latent_init),
        "latent_prior_min_std": float(args.latent_prior_min_std),
        "latent_prior_max_std": float(args.latent_prior_max_std),
        "model_prior_center": str(args.model_prior_center),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "basis_type": str(args.basis_type),
        "pca_train_count": int(args.pca_train_count),
        "pca_snrs": parse_snrs(args.pca_snrs),
        "pca_basis_path": str(pca_basis_path) if str(args.basis_type).startswith("error_pca") else None,
        "posterior_sampler": str(args.posterior_sampler),
        "posterior_latent_std": float(args.posterior_latent_std),
        "posterior_samples": int(args.posterior_samples),
        "include_map_center_sample": True,
        "laplace_fd_eps": float(args.laplace_fd_eps),
        "laplace_std_min": float(args.laplace_std_min),
        "laplace_std_max": float(args.laplace_std_max),
        "posterior_temperature": float(args.posterior_temperature),
        "laplace_precision_damping": float(args.laplace_precision_damping),
        "data_loss": str(args.data_loss),
        "data_smooth_kernel": int(args.data_smooth_kernel),
        "noise_std_floor": float(args.noise_std_floor),
        "seed": seed,
        "case_source": str(args.case_source),
    }

    wanted_snrs = set(parse_snrs(args.snrs))
    protocol_for_run = dict(protocol)
    protocol_for_run["uq"] = dict(protocol["uq"])
    protocol_for_run["uq"]["snrs_db"] = parse_snrs(args.snrs)
    all_cases = [
        case for case in select_cases(protocol_for_run, artifacts, args.split, str(args.case_source)) if case["snr_db"] in wanted_snrs
    ]
    selected_cases = all_cases[int(args.case_offset) : int(args.case_offset) + int(args.case_count)]
    metrics_path = out_dir / "metrics.csv"
    records = read_existing_records(metrics_path) if args.resume else []
    completed = {record_key(record) for record in records}
    started = time.time()

    for index, case in enumerate(selected_cases, start=1):
        key = (int(case["global_index"]), str(case["view"]))
        if args.resume and key in completed:
            print(f"[{index}/{len(selected_cases)}] skip existing model={case['model_name']} view={case['view']}")
            continue
        print(
            f"[{index}/{len(selected_cases)}] model={case['model_name']} view={case['view']} "
            f"residual={args.residual_scale} prior={args.model_prior_std}",
            flush=True,
        )
        set_seed(seed + int(case["global_index"]) * 101 + int(case["snr_db"] * 10))
        with torch.no_grad():
            nn_prediction = model(case["observation"].to(device)).cpu()
        result, samples, metrics = run_map_bvi_case(
            surrogate,
            case["observation"],
            case["clean_observation"],
            case["truth"],
            nn_prediction,
            latent_dim=int(args.latent_dim),
            residual_scale=float(args.residual_scale),
            model_prior_std=float(args.model_prior_std),
            latent_prior_weight=float(args.latent_prior_weight),
            steps=int(args.steps),
            lr=float(args.lr),
            basis_type=str(args.basis_type),
            posterior_latent_std=float(args.posterior_latent_std),
            posterior_sample_count=int(args.posterior_samples),
            event_threshold=float(protocol["uq"]["event_threshold_normalized"]),
            device=device,
            data_loss=str(args.data_loss),
            data_smooth_kernel=int(args.data_smooth_kernel),
            noise_std_floor=float(args.noise_std_floor),
            basis_override=basis_override,
            latent_prior_mean=latent_prior_mean,
            latent_prior_std=latent_prior_std,
            latent_prior_min_std=float(args.latent_prior_min_std),
            latent_prior_max_std=float(args.latent_prior_max_std),
            latent_init=str(args.latent_init),
            model_prior_center=str(args.model_prior_center),
            posterior_sampler=str(args.posterior_sampler),
            laplace_fd_eps=float(args.laplace_fd_eps),
            laplace_std_min=float(args.laplace_std_min),
            laplace_std_max=float(args.laplace_std_max),
            posterior_temperature=float(args.posterior_temperature),
            laplace_precision_damping=float(args.laplace_precision_damping),
        )
        arrays_path = out_dir / f"model_{case['global_index']}_{case['view']}.npz"
        save_posterior_arrays(
            arrays_path,
            case,
            nn_prediction,
            samples,
            float(protocol["uq"]["event_threshold_normalized"]),
        )
        nn_metric = deterministic_metrics(
            nn_prediction,
            case["truth"],
            float(protocol["uq"]["event_threshold_normalized"]),
        )[0]
        record = metrics | {
            "global_index": int(case["global_index"]),
            "model_name": case["model_name"],
            "view": case["view"],
            "snr_db": float(case["snr_db"]),
            "arrays": str(arrays_path),
            "method": "map_centered_deepwave_neural_bvi",
            "config_name": str(args.name),
            "posterior_sampler": str(args.posterior_sampler),
            "seed": seed,
            "nn_epsilon_rmse": float(nn_metric["epsilon_rmse"]),
            "rmse_delta_bvi_minus_nn": float(metrics["epsilon_rmse"]) - float(nn_metric["epsilon_rmse"]),
            "data_misfit_nn": result.data_misfit_nn,
            "data_misfit_bvi_mean": result.data_misfit_bvi_mean,
            "data_misfit_delta_bvi_minus_nn": result.data_misfit_bvi_mean - result.data_misfit_nn,
            **asdict(result),
        }
        records.append(record)
        completed.add(key)
        write_csv(metrics_path, records)
        write_summary(out_dir / "summary.json", "running", root, args.split, parse_snrs(args.snrs), config, records, started)
        print(
            f"  epsilon_rmse={record['epsilon_rmse']:.5f} "
            f"delta={record['rmse_delta_bvi_minus_nn']:+.6f} "
            f"data_delta={record['data_misfit_delta_bvi_minus_nn']:+.6f}",
            flush=True,
        )

    write_csv(metrics_path, records)
    write_summary(out_dir / "summary.json", "complete", root, args.split, parse_snrs(args.snrs), config, records, started)
    print((out_dir / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
