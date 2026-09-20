"""Unified matched-physics comparison for Neural-BVI.

The benchmark answers three questions under one Deepwave scalar-wave operator:
deterministic neural inversion, neural predictive UQ, and the incremental value
of boosting plus the model-space trust region. Residual-space methods share one
training-error PCA basis and one test protocol.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from e2e_bvi_gpr import run_bvi
from inversion_models import enable_mc_dropout
from publication_data import load_protocol
from publication_metrics import ause, deterministic_metrics, uq_metrics
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, set_seed, write_csv
from publication_uq import save_posterior_arrays, uq_cases
from run_deepwave_map_bvi_synthetic import build_error_pca_basis, run_map_bvi_case


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "publication_la010010" / "full"
DEFAULT_OUT = DEFAULT_ROOT / "neural_bvi_unified_comparison_v2"

METHODS = (
    "nn",
    "mc_dropout",
    "deep_ensemble",
    "residual_map",
    "neural_vi_k1",
    "residual_laplace",
    "neural_bvi_no_trust",
    "neural_bvi",
)
UQ_METHODS = {
    "mc_dropout",
    "deep_ensemble",
    "neural_vi_k1",
    "residual_laplace",
    "neural_bvi_no_trust",
    "neural_bvi",
}
LOWER_IS_BETTER = {
    "normalized_rmse",
    "epsilon_mae",
    "data_misfit",
    "crps",
    "ause",
    "event_brier",
    "calibrated_coverage_error",
    "calibrated_mpiw_95",
    "calibrated_gaussian_crps",
    "calibrated_gaussian_nll",
}
SUMMARY_METRICS = (
    "normalized_rmse",
    "epsilon_mae",
    "data_misfit",
    "crps",
    "calibrated_coverage_95",
    "calibrated_coverage_error",
    "calibrated_mpiw_95",
    "calibrated_gaussian_crps",
    "calibrated_gaussian_nll",
    "std_error_spearman",
    "ause",
    "event_auroc",
    "event_auprc",
    "event_brier",
    "wall_seconds",
    "physics_evaluations",
)

PLANNED_COMPARISONS = (
    ("RQ1_reconstruction", "primary", "nn", "normalized_rmse"),
    ("RQ1_reconstruction", "secondary", "nn", "epsilon_mae"),
    ("RQ1_reconstruction", "secondary", "nn", "data_misfit"),
    ("RQ2_neural_uq", "primary", "mc_dropout", "crps"),
    ("RQ2_neural_uq", "primary", "deep_ensemble", "crps"),
    ("RQ2_neural_uq", "secondary", "mc_dropout", "calibrated_gaussian_crps"),
    ("RQ2_neural_uq", "secondary", "deep_ensemble", "calibrated_gaussian_crps"),
    ("RQ2_neural_uq", "secondary", "mc_dropout", "calibrated_coverage_error"),
    ("RQ2_neural_uq", "secondary", "deep_ensemble", "calibrated_coverage_error"),
    ("RQ2_neural_uq", "secondary", "mc_dropout", "std_error_spearman"),
    ("RQ2_neural_uq", "secondary", "deep_ensemble", "std_error_spearman"),
    ("RQ2_neural_uq", "secondary", "mc_dropout", "ause"),
    ("RQ2_neural_uq", "secondary", "deep_ensemble", "ause"),
    ("RQ2_physics_uq", "secondary", "residual_laplace", "crps"),
    ("RQ2_physics_uq", "secondary", "residual_laplace", "ause"),
    ("RQ3_boosting", "primary", "neural_vi_k1", "crps"),
    ("RQ3_boosting", "secondary", "neural_vi_k1", "normalized_rmse"),
    ("RQ3_boosting", "secondary", "neural_vi_k1", "data_misfit"),
    ("RQ3_boosting", "secondary", "neural_vi_k1", "ause"),
    ("RQ3_trust_region", "primary", "neural_bvi_no_trust", "normalized_rmse"),
    ("RQ3_trust_region", "secondary", "neural_bvi_no_trust", "data_misfit"),
    ("RQ3_trust_region", "secondary", "neural_bvi_no_trust", "crps"),
    ("RQ3_trust_region", "secondary", "neural_bvi_no_trust", "ause"),
)


class EnsembleMeanModel(torch.nn.Module):
    """Expose a fixed ensemble mean through the inversion-model interface."""

    def __init__(self, members: list[torch.nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("EnsembleMeanModel requires at least one member")
        self.members = torch.nn.ModuleList(members)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(observation) for member in self.members], dim=0).mean(dim=0)


def apply_center_transform(
    center: torch.Tensor,
    mean_shift: float = 0.0,
    contrast_correction: float = 0.0,
) -> torch.Tensor:
    """Apply a frozen low-degree correction to an ensemble center."""

    spatial_mean = center.mean(dim=(-2, -1), keepdim=True)
    return (
        center
        + float(mean_shift)
        + float(contrast_correction) * (center - spatial_mean)
    ).clamp(0.0, 1.0)


def project_samples_to_mean(samples: torch.Tensor, desired_mean: torch.Tensor) -> torch.Tensor:
    """Preserve the requested mean while maximally retaining bounded deviations."""

    deviations = samples - samples.mean(dim=0, keepdim=True)
    maximum_positive = deviations.amax(dim=0, keepdim=True).clamp_min(0.0)
    maximum_negative = (-deviations.amin(dim=0, keepdim=True)).clamp_min(0.0)
    ones = torch.ones_like(desired_mean)
    positive_scale = torch.where(
        maximum_positive > 0.0,
        (1.0 - desired_mean) / maximum_positive.clamp_min(1.0e-12),
        ones,
    )
    negative_scale = torch.where(
        maximum_negative > 0.0,
        desired_mean / maximum_negative.clamp_min(1.0e-12),
        ones,
    )
    feasible_scale = torch.minimum(
        ones,
        torch.minimum(positive_scale, negative_scale),
    ).clamp(0.0, 1.0)
    return (desired_mean + feasible_scale * deviations).clamp(0.0, 1.0)


class CalibratedEnsembleModel(torch.nn.Module):
    """Validation-fitted convex ensemble with a fixed spatial bias correction."""

    def __init__(
        self,
        members: list[torch.nn.Module],
        weights: torch.Tensor,
        bias: torch.Tensor,
        bias_scale: float,
        mean_shift: float = 0.0,
        contrast_correction: float = 0.0,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("CalibratedEnsembleModel requires at least one member")
        if int(weights.numel()) != len(members):
            raise ValueError("Ensemble weight count does not match the checkpoint count")
        self.members = torch.nn.ModuleList(members)
        self.register_buffer("weights", weights.detach().float().flatten())
        self.register_buffer("bias", bias.detach().float().reshape(1, 1, *bias.shape[-2:]))
        self.bias_scale = float(bias_scale)
        self.mean_shift = float(mean_shift)
        self.contrast_correction = float(contrast_correction)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        predictions = torch.stack([member(observation) for member in self.members], dim=0)
        center = torch.einsum("m,mbchw->bchw", self.weights, predictions)
        center = (center + self.bias_scale * self.bias).clamp(0.0, 1.0)
        return apply_center_transform(center, self.mean_shift, self.contrast_correction)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_std(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = np.asarray(
        [number for value in values if (number := finite_float(value)) is not None],
        dtype=np.float64,
    )
    if not array.size:
        return {"count": 0, "mean": None, "std": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
    }


def average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    probabilities = probabilities.astype(np.float64).ravel()
    positive_count = int(labels.sum())
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].sum() / positive_count)


def posterior_extras(samples: torch.Tensor, truth: torch.Tensor, event_threshold: float) -> dict[str, float]:
    sample_np = samples.detach().cpu().numpy().astype(np.float64)
    truth_np = truth.detach().cpu().numpy().astype(np.float64)
    low = np.quantile(sample_np, 0.025, axis=0, keepdims=True)
    high = np.quantile(sample_np, 0.975, axis=0, keepdims=True)
    event_probability = np.mean(sample_np >= float(event_threshold), axis=0, keepdims=True)
    event_truth = truth_np >= float(event_threshold)
    return {
        "sample_quantile_coverage_95": float(np.mean((truth_np >= low) & (truth_np <= high))),
        "event_auprc": average_precision(event_truth, event_probability),
        "event_prevalence": float(event_truth.mean()),
    }


def exact_data_misfit(
    forward: torch.nn.Module,
    estimate: torch.Tensor,
    observation: torch.Tensor,
    device: torch.device,
) -> float:
    with torch.no_grad():
        prediction = forward(estimate.to(device))
        return float(torch.sqrt(F.mse_loss(prediction, observation.to(device))).cpu())


def base_record(case: dict[str, Any], method: str, split: str) -> dict[str, Any]:
    return {
        "split": split,
        "method": method,
        "global_index": int(case["global_index"]),
        "model_name": str(case["model_name"]),
        "view": str(case["view"]),
        "snr_db": float(case["snr_db"]),
    }


def posterior_record(
    *,
    case: dict[str, Any],
    method: str,
    split: str,
    samples: torch.Tensor,
    nn_prediction: torch.Tensor,
    forward: torch.nn.Module,
    event_threshold: float,
    device: torch.device,
    arrays_path: Path,
    started: float,
    physics_evaluations: int,
    diagnostics: dict[str, Any] | None = None,
    std_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    samples = samples.detach().cpu()
    metrics = uq_metrics(samples, case["truth"], event_threshold)
    if std_override is not None:
        mean_np = samples.numpy().astype(np.float64).mean(axis=0, keepdims=True)
        truth_np = case["truth"].numpy().astype(np.float64)
        std_np = np.maximum(std_override.detach().cpu().numpy().astype(np.float64), 1.0e-8)
        abs_error = np.abs(mean_np - truth_np)
        standardized = abs_error / std_np
        metrics |= {
            "gaussian_nll": float(
                np.mean(
                    0.5 * standardized**2
                    + np.log(std_np)
                    + 0.5 * math.log(2.0 * math.pi)
                )
            ),
            "coverage_95": float(
                np.mean(
                    (truth_np >= mean_np - 1.96 * std_np)
                    & (truth_np <= mean_np + 1.96 * std_np)
                )
            ),
            "mpiw_95": float(np.mean(3.92 * std_np)),
            "std_error_spearman": float(
                stats.spearmanr(std_np.ravel(), abs_error.ravel()).statistic
            ),
            "ause": ause(std_np, abs_error),
        }
    extras = posterior_extras(samples, case["truth"], event_threshold)
    mean = samples.mean(dim=0, keepdim=True)
    save_posterior_arrays(
        arrays_path,
        case,
        nn_prediction,
        samples,
        event_threshold,
        std_override=std_override,
    )
    return base_record(case, method, split) | metrics | extras | {
        "normalized_rmse": float(metrics["epsilon_rmse"]) / 8.0,
        "data_misfit": exact_data_misfit(forward, mean, case["observation"], device),
        "arrays": str(arrays_path.resolve()),
        "wall_seconds": float(time.time() - started),
        "physics_evaluations": int(physics_evaluations),
    } | (diagnostics or {})


def deterministic_record(
    *,
    case: dict[str, Any],
    method: str,
    split: str,
    estimate: torch.Tensor,
    forward: torch.nn.Module,
    device: torch.device,
    started: float,
    physics_evaluations: int,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metric = deterministic_metrics(estimate, case["truth"])[0]
    return base_record(case, method, split) | {
        "epsilon_rmse": float(metric["epsilon_rmse"]),
        "normalized_rmse": float(metric["epsilon_rmse"]) / 8.0,
        "epsilon_mae": float(metric["epsilon_mae"]),
        "data_misfit": exact_data_misfit(forward, estimate, case["observation"], device),
        "arrays": "",
        "wall_seconds": float(time.time() - started),
        "physics_evaluations": int(physics_evaluations),
    } | (diagnostics or {})


def run_bvi_method(
    *,
    case: dict[str, Any],
    method: str,
    split: str,
    components: int,
    trust_std: float,
    nn_prediction: torch.Tensor,
    basis: torch.Tensor,
    latent_prior_mean: torch.Tensor,
    latent_prior_std: torch.Tensor,
    forward: torch.nn.Module,
    event_threshold: float,
    device: torch.device,
    arrays_path: Path,
    config: dict[str, Any],
    predictive_samples: torch.Tensor | None = None,
    mc_dropout_samples: torch.Tensor | None = None,
    reference_prediction: torch.Tensor | None = None,
    posterior_std_prior: torch.Tensor | None = None,
    posterior_std_prior_scale: float = 0.0,
) -> dict[str, Any]:
    started = time.time()
    noise = case["observation"] - case["clean_observation"]
    noise_std = max(float(torch.sqrt(torch.mean(noise**2))), float(config["noise_std_floor"]))
    result, _, _, _, samples = run_bvi(
        forward=forward,
        observation=case["observation"],
        true_model=case["truth"],
        nn_mean=nn_prediction,
        latent_dim=int(config["latent_dim"]),
        components=int(components),
        steps=int(config["bvi_steps"]),
        samples_per_component=int(config["samples_per_component"]),
        residual_scale=float(config["residual_scale"]),
        noise_std=noise_std,
        kl_weight=float(config["kl_weight"]),
        model_prior_std=float(trust_std),
        interrogation_threshold=float(event_threshold),
        basis_type="cosine",
        device=device,
        posterior_sample_count=int(config["posterior_samples"]),
        return_samples=True,
        basis_override=basis,
        latent_prior_mean=latent_prior_mean,
        latent_prior_std=latent_prior_std,
        learning_rate=float(config["learning_rate"]),
        initial_log_std=float(config["initial_log_std"]),
        antithetic_samples=True,
        forward_batch_size=int(config["candidate_batch_size"]),
        streaming_backward=True,
        initial_component_weight=float(config["initial_component_weight"]),
        component_weight_learning_rate=float(config["component_weight_learning_rate"]),
        mixture_line_search_samples=int(config["mixture_line_search_samples"]),
    )
    raw_posterior_mean = samples.mean(dim=0, keepdim=True)
    predictive_scale = float(config["ensemble_dispersion_scale"]) if method == "neural_bvi" else 0.0
    mc_dropout_scale = float(config["mc_dropout_dispersion_scale"]) if method == "neural_bvi" else 0.0

    def add_centered_dispersion(
        posterior_samples: torch.Tensor,
        source_samples: torch.Tensor | None,
        scale: float,
        label: str,
    ) -> torch.Tensor:
        if scale <= 0.0:
            return posterior_samples
        if source_samples is None or int(source_samples.shape[0]) < 2:
            raise ValueError(f"{label} dispersion requires at least two predictive samples")
        source_samples = source_samples.to(dtype=posterior_samples.dtype, device=posterior_samples.device)
        deviations = source_samples - source_samples.mean(dim=0, keepdim=True)
        indices = torch.arange(int(posterior_samples.shape[0]), device=posterior_samples.device) % int(
            deviations.shape[0]
        )
        selected = deviations[indices]
        selected = selected - selected.mean(dim=0, keepdim=True)
        target_mean = posterior_samples.mean(dim=0, keepdim=True)
        augmented = posterior_samples + float(scale) * selected
        return target_mean + augmented - augmented.mean(dim=0, keepdim=True)

    samples = add_centered_dispersion(samples, predictive_samples, predictive_scale, "Ensemble")
    samples = add_centered_dispersion(samples, mc_dropout_samples, mc_dropout_scale, "MC-dropout")
    bvi_std_mean = float(samples.std(dim=0, unbiased=False, keepdim=True).mean().cpu())
    posterior_mean_policy = str(config["posterior_mean_policy"])
    mean_line_search: list[dict[str, Any]] = []
    mean_line_search_evaluations = 0
    selected_mean_label = ""
    if posterior_mean_policy == "fixed":
        posterior_mean_alpha = float(config["posterior_mean_alpha"])
        desired_mean = nn_prediction.to(samples.device) + posterior_mean_alpha * (
            raw_posterior_mean - nn_prediction.to(samples.device)
        )
        selected_mean_label = f"bvi_alpha_{posterior_mean_alpha:g}"
    elif posterior_mean_policy == "bvi_std_threshold":
        posterior_mean_alpha = (
            float(config["posterior_mean_low_std_alpha"])
            if bvi_std_mean <= float(config["posterior_mean_std_threshold"])
            else float(config["posterior_mean_high_std_alpha"])
        )
        desired_mean = nn_prediction.to(samples.device) + posterior_mean_alpha * (
            raw_posterior_mean - nn_prediction.to(samples.device)
        )
        selected_mean_label = f"bvi_alpha_{posterior_mean_alpha:g}"
    elif posterior_mean_policy == "physics_line_search":
        if reference_prediction is None:
            raise ValueError("physics_line_search requires an explicit reference prediction")
        center = nn_prediction.to(samples.device)
        reference = reference_prediction.to(samples.device).clamp(0.0, 1.0)
        direction = raw_posterior_mean - center
        reference_data = exact_data_misfit(forward, reference, case["observation"], device)
        mean_line_search_evaluations += 1
        seen: set[float] = set()
        for alpha_value in config["posterior_mean_line_search_alphas"]:
            alpha = float(alpha_value)
            if alpha in seen:
                continue
            seen.add(alpha)
            candidate = (center + alpha * direction).clamp(0.0, 1.0)
            data_rmse = exact_data_misfit(forward, candidate, case["observation"], device)
            mean_line_search_evaluations += 1
            mean_line_search.append(
                {
                    "label": "center" if alpha == 0.0 else f"bvi_alpha_{alpha:g}",
                    "alpha": alpha,
                    "data_rmse": data_rmse,
                    "candidate": candidate,
                }
            )
        minimum_gain = float(config["posterior_mean_line_search_min_gain"])
        eligible = [
            item
            for item in mean_line_search
            if float(item["data_rmse"]) <= reference_data - minimum_gain
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda item: (abs(float(item["alpha"])), float(item["data_rmse"])),
            )
            desired_mean = selected["candidate"]
            posterior_mean_alpha = float(selected["alpha"])
            selected_mean_label = str(selected["label"])
        else:
            desired_mean = reference
            posterior_mean_alpha = -1.0
            selected_mean_label = "reference_ensemble_mean"
        mean_line_search.insert(
            0,
            {
                "label": "reference_ensemble_mean",
                "alpha": None,
                "data_rmse": reference_data,
                "candidate": reference,
            },
        )
    else:
        raise ValueError(f"Unknown posterior mean policy {posterior_mean_policy}")
    desired_mean = desired_mean.clamp(0.0, 1.0)
    if posterior_mean_policy == "physics_line_search":
        samples = project_samples_to_mean(samples, desired_mean)
    else:
        samples = desired_mean + samples - samples.mean(dim=0, keepdim=True)
        for _ in range(8):
            samples = samples.clamp(0.0, 1.0)
            samples = samples + desired_mean - samples.mean(dim=0, keepdim=True)
        samples = samples.clamp(0.0, 1.0)
    posterior_std_override: torch.Tensor | None = None
    posterior_sample_std = samples.std(dim=0, unbiased=False, keepdim=True)
    std_prior_requested_mean = 0.0
    if posterior_std_prior is not None and float(posterior_std_prior_scale) > 0.0:
        prior = posterior_std_prior.to(dtype=samples.dtype, device=samples.device)
        posterior_std_override = torch.sqrt(
            posterior_sample_std.square()
            + (float(posterior_std_prior_scale) * prior).square()
        )
        std_prior_requested_mean = float(posterior_std_override.mean().cpu())
    posterior_mean_error = float((samples.mean(dim=0, keepdim=True) - desired_mean).abs().max().cpu())
    spp = int(config["samples_per_component"])
    line_search_solves = int(config["mixture_line_search_samples"]) * sum(
        range(2, int(components) + 1)
    )
    equivalent_solves = (
        int(config["bvi_steps"]) * spp * sum(range(1, int(components) + 1))
        + line_search_solves
        + 3
        + mean_line_search_evaluations
    )
    diagnostics = {
        "components": int(components),
        "neural_center": str(config["neural_center"]),
        "ensemble_dispersion_scale": predictive_scale,
        "mc_dropout_dispersion_scale": mc_dropout_scale,
        "posterior_mean_alpha": posterior_mean_alpha,
        "posterior_mean_policy": posterior_mean_policy,
        "posterior_mean_selected_label": selected_mean_label,
        "posterior_mean_line_search": json.dumps(
            [
                {
                    "label": item["label"],
                    "alpha": item["alpha"],
                    "data_rmse": item["data_rmse"],
                }
                for item in mean_line_search
            ]
        ),
        "bvi_std_mean": bvi_std_mean,
        "posterior_mean_projection_max_error": posterior_mean_error,
        "posterior_std_prior_scale": float(posterior_std_prior_scale),
        "posterior_std_prior_requested_mean": std_prior_requested_mean,
        "posterior_std_prior_actual_mean": float(
            (
                posterior_std_override
                if posterior_std_override is not None
                else posterior_sample_std
            ).mean().cpu()
        ),
        "posterior_sample_std_mean": float(posterior_sample_std.mean().cpu()),
        "trust_std": float(trust_std),
        "noise_std": float(noise_std),
        "mixture_effective_components": result.mixture_effective_components,
        "mixture_max_weight": result.mixture_max_weight,
        "stage_new_weights": json.dumps(result.stage_new_weights),
        "stage_line_search_objectives": json.dumps(result.stage_line_search_objectives),
        "stage_line_search_gains": json.dumps(result.stage_line_search_gains),
        "final_objective": result.final_loss,
    }
    return posterior_record(
        case=case,
        method=method,
        split=split,
        samples=samples,
        nn_prediction=nn_prediction,
        forward=forward,
        event_threshold=event_threshold,
        device=device,
        arrays_path=arrays_path,
        started=started,
        physics_evaluations=equivalent_solves,
        diagnostics=diagnostics,
        std_override=posterior_std_override,
    )


def run_map_method(
    *,
    case: dict[str, Any],
    method: str,
    split: str,
    sampler: str,
    nn_prediction: torch.Tensor,
    basis: torch.Tensor,
    latent_prior_mean: torch.Tensor,
    latent_prior_std: torch.Tensor,
    forward: torch.nn.Module,
    event_threshold: float,
    device: torch.device,
    arrays_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    sample_count = 1 if sampler == "fixed" else int(config["posterior_samples"])
    result, samples, _ = run_map_bvi_case(
        forward,
        case["observation"],
        case["clean_observation"],
        case["truth"],
        nn_prediction,
        latent_dim=int(config["latent_dim"]),
        residual_scale=float(config["residual_scale"]),
        model_prior_std=float(config["trust_std"]),
        latent_prior_weight=float(config["map_latent_prior_weight"]),
        steps=int(config["map_steps"]),
        lr=float(config["learning_rate"]),
        basis_type="cosine",
        posterior_latent_std=0.0 if sampler == "fixed" else float(config["posterior_latent_std"]),
        posterior_sample_count=sample_count,
        event_threshold=float(event_threshold),
        device=device,
        noise_std_floor=float(config["noise_std_floor"]),
        basis_override=basis,
        latent_prior_mean=latent_prior_mean,
        latent_prior_std=latent_prior_std,
        latent_init="prior_mean",
        model_prior_center="nn",
        posterior_sampler=sampler,
        laplace_fd_eps=float(config["laplace_fd_eps"]),
        laplace_std_min=float(config["laplace_std_min"]),
        laplace_std_max=float(config["laplace_std_max"]),
        laplace_precision_damping=float(config["laplace_precision_damping"]),
        include_map_center_sample=True if sampler == "fixed" else False,
    )
    diagnostics = {
        "components": 0,
        "trust_std": float(config["trust_std"]),
        "noise_std": "",
        "final_objective": result.final_loss,
        "posterior_precision_condition": result.posterior_precision_condition,
        "posterior_precision_data_trace": result.posterior_precision_data_trace,
    }
    if sampler == "fixed":
        estimate = samples.mean(dim=0, keepdim=True)
        return deterministic_record(
            case=case,
            method=method,
            split=split,
            estimate=estimate,
            forward=forward,
            device=device,
            started=started,
            physics_evaluations=int(config["map_steps"]) + 3,
            diagnostics=diagnostics,
        )
    save_evaluations = int(config["map_steps"]) + 2 * int(config["latent_dim"]) + 3
    return posterior_record(
        case=case,
        method=method,
        split=split,
        samples=samples,
        nn_prediction=nn_prediction,
        forward=forward,
        event_threshold=event_threshold,
        device=device,
        arrays_path=arrays_path,
        started=started,
        physics_evaluations=save_evaluations,
        diagnostics=diagnostics,
    )


def calibration_scale_from_records(records: list[dict[str, Any]], target: float = 0.95) -> float:
    ratios: list[np.ndarray] = []
    for record in records:
        arrays_path = record.get("arrays")
        if not arrays_path:
            continue
        arrays = np.load(arrays_path)
        truth = arrays["truth"].astype(np.float64)
        mean = arrays["mean"].astype(np.float64)
        std = np.maximum(arrays["std"].astype(np.float64), 1.0e-8)
        ratios.append((np.abs(truth - mean) / std).ravel())
    if not ratios:
        return 1.0
    return max(float(np.quantile(np.concatenate(ratios), target)) / 1.96, 1.0e-3)


def add_calibrated_metrics(record: dict[str, Any], scale: float) -> dict[str, Any]:
    arrays_path = record.get("arrays")
    if not arrays_path:
        return record
    arrays = np.load(arrays_path)
    truth = arrays["truth"].astype(np.float64)
    mean = arrays["mean"].astype(np.float64)
    raw_std = np.maximum(arrays["std"].astype(np.float64), 1.0e-8)
    std = np.maximum(float(scale) * raw_std, 1.0e-8)
    low = mean - 1.96 * std
    high = mean + 1.96 * std
    standardized = (truth - mean) / std
    standardized_tensor = torch.from_numpy(standardized)
    normal_cdf = torch.special.ndtr(standardized_tensor).numpy()
    normal_pdf = np.exp(-0.5 * standardized**2) / math.sqrt(2.0 * math.pi)
    gaussian_crps = std * (
        standardized * (2.0 * normal_cdf - 1.0)
        + 2.0 * normal_pdf
        - 1.0 / math.sqrt(math.pi)
    )
    q025 = mean + float(scale) * (arrays["q025"].astype(np.float64) - mean)
    q975 = mean + float(scale) * (arrays["q975"].astype(np.float64) - mean)
    calibrated_coverage = float(np.mean((truth >= low) & (truth <= high)))
    return record | {
        "calibration_scale": float(scale),
        "calibrated_coverage_95": calibrated_coverage,
        "calibrated_coverage_error": abs(calibrated_coverage - 0.95),
        "calibrated_mpiw_95": float(np.mean(high - low)),
        "calibrated_gaussian_crps": float(np.mean(gaussian_crps)),
        "calibrated_gaussian_nll": float(
            np.mean(0.5 * standardized**2 + np.log(std) + 0.5 * math.log(2.0 * math.pi))
        ),
        "calibrated_sample_quantile_coverage_95": float(np.mean((truth >= q025) & (truth <= q975))),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method in METHODS:
        subset = [record for record in records if record.get("method") == method]
        if not subset:
            continue
        payload[method] = {
            "record_count": len(subset),
            "model_count": len({int(record["global_index"]) for record in subset}),
            "metrics": {metric: mean_std(record.get(metric) for record in subset) for metric in SUMMARY_METRICS},
        }
    return payload


def clustered_paired_bootstrap(
    records: list[dict[str, Any]],
    proposed: str,
    baseline: str,
    metric: str,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    by_method_model: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in records:
        value = finite_float(record.get(metric))
        if value is not None:
            by_method_model[(str(record["method"]), int(record["global_index"]))].append(value)
    model_ids = sorted(
        model_id
        for model_id in {key[1] for key in by_method_model}
        if (proposed, model_id) in by_method_model and (baseline, model_id) in by_method_model
    )
    if len(model_ids) < 2:
        return None
    differences = np.asarray(
        [
            np.mean(by_method_model[(proposed, model_id)])
            - np.mean(by_method_model[(baseline, model_id)])
            for model_id in model_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(int(resamples), len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "proposed": proposed,
        "baseline": baseline,
        "model_count": len(model_ids),
        "difference_proposed_minus_baseline": float(differences.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "favorable_direction": "negative" if metric in LOWER_IS_BETTER else "positive",
        "proposed_better_model_count": int(
            np.sum(differences < 0.0) if metric in LOWER_IS_BETTER else np.sum(differences > 0.0)
        ),
    }


def build_statistics(records: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    baselines = [method for method in METHODS if method != "neural_bvi"]
    for baseline in baselines:
        for metric in SUMMARY_METRICS:
            result = clustered_paired_bootstrap(records, "neural_bvi", baseline, metric, resamples, seed)
            if result is not None:
                comparisons.append(result)
    return comparisons


def build_planned_statistics(records: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for question, role, baseline, metric in PLANNED_COMPARISONS:
        result = clustered_paired_bootstrap(records, "neural_bvi", baseline, metric, resamples, seed)
        if result is not None:
            comparisons.append({"research_question": question, "role": role} | result)
    return comparisons


def write_comparison_table(path: Path, summary: dict[str, Any]) -> None:
    records = []
    for method, method_summary in summary.items():
        row: dict[str, Any] = {
            "method": method,
            "model_count": method_summary["model_count"],
            "record_count": method_summary["record_count"],
        }
        for metric, metric_summary in method_summary["metrics"].items():
            row[f"{metric}_mean"] = metric_summary["mean"]
            row[f"{metric}_std"] = metric_summary["std"]
        records.append(row)
    write_csv(path, records)


def choose_cases(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    split: str,
    model_count: int,
    local_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    if local_indices is None:
        cases = list(uq_cases(protocol, artifacts, split))
    else:
        fixed = artifacts["fixed"][split]
        clean = fixed["views"]["clean"]
        if len(set(local_indices)) != len(local_indices):
            raise ValueError(f"Duplicate {split} local indices are not allowed")
        if any(index < 0 or index >= len(fixed["indices"]) for index in local_indices):
            raise ValueError(f"{split} local indices are outside the prepared split")
        cases = []
        for local_index in local_indices:
            for snr in protocol["uq"]["snrs_db"]:
                view = f"mixed_{float(snr):g}db"
                global_index = int(fixed["indices"][local_index])
                cases.append(
                    {
                        "split": split,
                        "local_index": int(local_index),
                        "global_index": global_index,
                        "model_name": fixed["model_names"][local_index],
                        "view": view,
                        "snr_db": float(snr),
                        "observation": fixed["views"][view][local_index : local_index + 1],
                        "clean_observation": clean[local_index : local_index + 1],
                        "truth": artifacts["models"][global_index : global_index + 1],
                    }
                )
    selected_models: list[int] = []
    for case in cases:
        model_id = int(case["global_index"])
        if model_id not in selected_models and len(selected_models) < int(model_count):
            selected_models.append(model_id)
    return [case for case in cases if int(case["global_index"]) in selected_models]


def parse_local_indices(value: str) -> list[int] | None:
    if not value.strip():
        return None
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("Local-index list must not be empty")
    return indices


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Floating-point list must not be empty")
    if any(not math.isfinite(item) for item in values):
        raise ValueError("Floating-point list contains a non-finite value")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scale_for_record(scales: dict[str, Any], record: dict[str, Any]) -> float:
    method_scale = scales.get(str(record.get("method")), 1.0)
    if isinstance(method_scale, dict):
        if method_scale.get("kind") == "linear_record_policy":
            values: list[float] = []
            for feature in method_scale["features"]:
                if feature == "posterior_mean_reference_fallback":
                    values.append(
                        float(
                            float(record.get("posterior_mean_alpha", 0.0)) < 0.0
                            or str(record.get("posterior_mean_selected_label", ""))
                            == "reference_ensemble_mean"
                        )
                    )
                else:
                    values.append(float(record[feature]))
            means = np.asarray(method_scale["feature_means"], dtype=np.float64)
            stds = np.maximum(
                np.asarray(method_scale["feature_stds"], dtype=np.float64), 1.0e-12
            )
            coefficients = np.asarray(method_scale["coefficients"], dtype=np.float64)
            standardized = (np.asarray(values, dtype=np.float64) - means) / stds
            raw_scale = float(coefficients[0] + standardized @ coefficients[1:])
            lower, upper = [float(value) for value in method_scale["clip"]]
            return float(
                np.clip(raw_scale, lower, upper)
                * float(method_scale["bias_multiplier"])
            )
        snr = float(record.get("snr_db"))
        for key in (f"{snr:g}", str(snr)):
            if key in method_scale:
                return float(method_scale[key])
        raise KeyError(f"No calibration scale for {record.get('method')} at {snr:g} dB")
    return float(method_scale)


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; valid methods are {METHODS}")
    return methods


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [split for split in splits if split not in {"validation", "test"}]
    if unknown:
        raise ValueError(f"Unknown splits {unknown}; valid splits are validation,test")
    return splits


def run_split(
    *,
    split: str,
    cases: list[dict[str, Any]],
    methods: list[str],
    records: list[dict[str, Any]],
    metrics_path: Path,
    out_dir: Path,
    model: torch.nn.Module,
    ensemble_models: list[torch.nn.Module],
    ensemble_center_weights: torch.Tensor | None,
    ensemble_center_bias: torch.Tensor | None,
    ensemble_center_bias_scale: float,
    ensemble_center_mean_shift: float,
    ensemble_center_contrast_correction: float,
    posterior_std_prior: torch.Tensor | None,
    posterior_std_prior_scale: float,
    forward: torch.nn.Module,
    basis: torch.Tensor,
    latent_prior_mean: torch.Tensor,
    latent_prior_std: torch.Tensor,
    event_threshold: float,
    device: torch.device,
    config: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    completed = {
        (str(record["split"]), str(record["method"]), int(record["global_index"]), str(record["view"]))
        for record in records
    }
    for case_index, case in enumerate(cases, start=1):
        case_seed = seed + int(case["global_index"]) * 101 + int(case["snr_db"] * 10)
        set_seed(case_seed)
        with torch.no_grad():
            nn_prediction = model(case["observation"].to(device)).cpu()
            ensemble_predictions = torch.cat(
                [member(case["observation"].to(device)).cpu() for member in ensemble_models], dim=0
            )
            if config["neural_center"] == "single":
                bvi_center_prediction = nn_prediction
            elif config["neural_center"] == "ensemble_mean":
                bvi_center_prediction = ensemble_predictions.mean(dim=0, keepdim=True)
            elif config["neural_center"] == "ensemble_calibrated":
                if ensemble_center_weights is None or ensemble_center_bias is None:
                    raise RuntimeError("Calibrated ensemble center was not loaded")
                bvi_center_prediction = torch.einsum(
                    "m,mchw->chw", ensemble_center_weights, ensemble_predictions
                ).unsqueeze(0)
                bvi_center_prediction = (
                    bvi_center_prediction
                    + float(ensemble_center_bias_scale) * ensemble_center_bias
                ).clamp(0.0, 1.0)
                bvi_center_prediction = apply_center_transform(
                    bvi_center_prediction,
                    ensemble_center_mean_shift,
                    ensemble_center_contrast_correction,
                )
            else:
                raise ValueError(f"Unknown neural center {config['neural_center']}")
        mc_dropout_predictions = None
        if float(config["mc_dropout_dispersion_scale"]) > 0.0:
            set_seed(case_seed + 71_117)
            enable_mc_dropout(model)
            with torch.no_grad():
                mc_dropout_predictions = torch.cat(
                    [model(case["observation"].to(device)).cpu() for _ in range(int(config["mc_samples"]))],
                    dim=0,
                )
            model.eval()
        print(
            f"[{split} {case_index}/{len(cases)}] {case['model_name']} {case['view']}",
            flush=True,
        )
        for method in methods:
            key = (split, method, int(case["global_index"]), str(case["view"]))
            if key in completed:
                print(f"  skip {method}", flush=True)
                continue
            print(f"  run {method}", flush=True)
            set_seed(case_seed)
            arrays_path = out_dir / split / "arrays" / method / f"model_{case['global_index']}_{case['view']}.npz"
            if method == "nn":
                started = time.time()
                record = deterministic_record(
                    case=case,
                    method=method,
                    split=split,
                    estimate=nn_prediction,
                    forward=forward,
                    device=device,
                    started=started,
                    physics_evaluations=1,
                )
            elif method == "mc_dropout":
                started = time.time()
                enable_mc_dropout(model)
                with torch.no_grad():
                    samples = torch.cat(
                        [model(case["observation"].to(device)).cpu() for _ in range(int(config["mc_samples"]))],
                        dim=0,
                    )
                model.eval()
                record = posterior_record(
                    case=case,
                    method=method,
                    split=split,
                    samples=samples,
                    nn_prediction=nn_prediction,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    started=started,
                    physics_evaluations=1,
                )
            elif method == "deep_ensemble":
                started = time.time()
                record = posterior_record(
                    case=case,
                    method=method,
                    split=split,
                    samples=ensemble_predictions,
                    nn_prediction=nn_prediction,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    started=started,
                    physics_evaluations=1,
                )
            elif method == "residual_map":
                record = run_map_method(
                    case=case,
                    method=method,
                    split=split,
                    sampler="fixed",
                    nn_prediction=nn_prediction,
                    basis=basis,
                    latent_prior_mean=latent_prior_mean,
                    latent_prior_std=latent_prior_std,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    config=config,
                )
            elif method == "residual_laplace":
                record = run_map_method(
                    case=case,
                    method=method,
                    split=split,
                    sampler="laplace_full",
                    nn_prediction=nn_prediction,
                    basis=basis,
                    latent_prior_mean=latent_prior_mean,
                    latent_prior_std=latent_prior_std,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    config=config,
                )
            elif method == "neural_vi_k1":
                record = run_bvi_method(
                    case=case,
                    method=method,
                    split=split,
                    components=1,
                    trust_std=float(config["trust_std"]),
                    nn_prediction=bvi_center_prediction,
                    basis=basis,
                    latent_prior_mean=latent_prior_mean,
                    latent_prior_std=latent_prior_std,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    config=config,
                )
            elif method in {"neural_bvi", "neural_bvi_no_trust"}:
                record = run_bvi_method(
                    case=case,
                    method=method,
                    split=split,
                    components=int(config["components"]),
                    trust_std=float(config["trust_std"]) if method == "neural_bvi" else 0.0,
                    nn_prediction=bvi_center_prediction,
                    basis=basis,
                    latent_prior_mean=latent_prior_mean,
                    latent_prior_std=latent_prior_std,
                    forward=forward,
                    event_threshold=event_threshold,
                    device=device,
                    arrays_path=arrays_path,
                    config=config,
                    predictive_samples=ensemble_predictions,
                    mc_dropout_samples=mc_dropout_predictions,
                    reference_prediction=ensemble_predictions.mean(dim=0, keepdim=True),
                    posterior_std_prior=posterior_std_prior,
                    posterior_std_prior_scale=posterior_std_prior_scale,
                )
            else:
                raise AssertionError(method)
            records.append(record)
            completed.add(key)
            write_csv(metrics_path, records)
            print(
                f"    image={finite_float(record.get('normalized_rmse')):.5f} "
                f"data={finite_float(record.get('data_misfit')):.5f} "
                f"seconds={finite_float(record.get('wall_seconds')):.1f}",
                flush=True,
            )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--protocol", type=Path, default=HERE / "la010010_protocol.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--model-count", type=int, default=10)
    parser.add_argument("--validation-local-indices", default="")
    parser.add_argument("--test-local-indices", default="")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument(
        "--neural-center",
        choices=["single", "ensemble_mean", "ensemble_calibrated"],
        default="single",
    )
    parser.add_argument("--ensemble-center-json", type=Path)
    parser.add_argument("--ensemble-dispersion-scale", type=float, default=0.0)
    parser.add_argument("--mc-dropout-dispersion-scale", type=float, default=0.0)
    parser.add_argument("--posterior-mean-alpha", type=float, default=1.0)
    parser.add_argument(
        "--posterior-mean-policy",
        choices=["fixed", "bvi_std_threshold", "physics_line_search"],
        default="fixed",
    )
    parser.add_argument("--posterior-mean-std-threshold", type=float, default=0.0)
    parser.add_argument("--posterior-mean-low-std-alpha", type=float, default=1.0)
    parser.add_argument("--posterior-mean-high-std-alpha", type=float, default=0.0)
    parser.add_argument(
        "--posterior-mean-line-search-alphas",
        default="0,0.025,0.05,0.1,0.2",
    )
    parser.add_argument("--posterior-mean-line-search-min-gain", type=float, default=0.0)
    parser.add_argument("--posterior-std-prior-json", type=Path)
    parser.add_argument("--external-calibration-json", type=Path)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--pca-train-count", type=int, default=96)
    parser.add_argument("--residual-scale", type=float, default=0.04)
    parser.add_argument("--trust-std", type=float, default=0.03)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--map-latent-prior-weight", type=float, default=0.6)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--bvi-steps", type=int, default=4)
    parser.add_argument("--map-steps", type=int, default=4)
    parser.add_argument("--samples-per-component", type=int, default=2)
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--initial-log-std", type=float, default=0.0)
    parser.add_argument("--candidate-batch-size", type=int, default=2)
    parser.add_argument("--initial-component-weight", type=float, default=0.1)
    parser.add_argument("--component-weight-learning-rate", type=float, default=0.3)
    parser.add_argument("--mixture-line-search-samples", type=int, default=4)
    parser.add_argument("--posterior-latent-std", type=float, default=0.03)
    parser.add_argument("--noise-std-floor", type=float, default=0.01)
    parser.add_argument("--laplace-fd-eps", type=float, default=0.01)
    parser.add_argument("--laplace-std-min", type=float, default=1.0e-4)
    parser.add_argument("--laplace-std-max", type=float, default=5.0)
    parser.add_argument("--laplace-precision-damping", type=float, default=1.0e-3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    methods = parse_methods(args.methods)
    splits = parse_splits(args.splits)
    protocol = (
        json.loads((root / "protocol.snapshot.json").read_text(encoding="utf-8"))
        if (root / "protocol.snapshot.json").exists()
        else load_protocol(args.protocol.resolve())
    )
    artifacts = torch.load(root / "prepared" / "artifacts.pt", map_location="cpu", weights_only=False)
    if artifacts["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Prepared artifacts do not match the protocol hash")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    forward, forward_payload = load_surrogate_checkpoint(root / "surrogate" / "best.pt", device)
    if forward_payload.get("architecture") != "DeepwavePhysicsSurrogate":
        raise ValueError("Unified comparison requires the exact differentiable Deepwave physics backend")
    model, _ = load_inversion_checkpoint(root / "train" / "unet" / f"seed_{args.seed}" / "best.pt", device)
    ensemble_models = [
        load_inversion_checkpoint(root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)[0]
        for seed in protocol["ensemble_seeds"]
    ]
    ensemble_center_path = args.ensemble_center_json.resolve() if args.ensemble_center_json else None
    ensemble_center_payload = None
    ensemble_center_weights = None
    ensemble_center_bias = None
    ensemble_center_bias_scale = 0.0
    ensemble_center_mean_shift = 0.0
    ensemble_center_contrast_correction = 0.0
    ensemble_center_bias_path = None
    if args.neural_center == "ensemble_calibrated":
        if ensemble_center_path is None or not ensemble_center_path.is_file():
            raise ValueError("--neural-center ensemble_calibrated requires --ensemble-center-json")
        ensemble_center_payload = json.loads(ensemble_center_path.read_text(encoding="utf-8"))
        ensemble_center_weights = torch.tensor(
            ensemble_center_payload["global_weights"], dtype=torch.float32
        )
        ensemble_center_bias_path = Path(ensemble_center_payload["bias_path"]).resolve()
        if not ensemble_center_bias_path.is_file():
            raise FileNotFoundError(ensemble_center_bias_path)
        if file_sha256(ensemble_center_bias_path) != ensemble_center_payload["bias_sha256"]:
            raise ValueError("Calibrated ensemble bias hash does not match its manifest")
        with np.load(ensemble_center_bias_path, allow_pickle=False) as bias_payload:
            ensemble_center_bias = torch.from_numpy(bias_payload["bias"]).float().reshape(1, 1, 256, 256)
        ensemble_center_bias_scale = float(ensemble_center_payload["bias_scale"])
        ensemble_center_mean_shift = float(ensemble_center_payload.get("mean_shift", 0.0))
        ensemble_center_contrast_correction = float(
            ensemble_center_payload.get("contrast_correction", 0.0)
        )
        bvi_center_model = CalibratedEnsembleModel(
            ensemble_models,
            ensemble_center_weights.to(device),
            ensemble_center_bias.to(device),
            ensemble_center_bias_scale,
            ensemble_center_mean_shift,
            ensemble_center_contrast_correction,
        ).to(device).eval()
    elif args.neural_center == "ensemble_mean":
        bvi_center_model = EnsembleMeanModel(ensemble_models).to(device).eval()
    else:
        if ensemble_center_path is not None:
            raise ValueError("--ensemble-center-json is only valid with ensemble_calibrated")
        bvi_center_model = model

    posterior_std_prior_json_path = (
        args.posterior_std_prior_json.resolve() if args.posterior_std_prior_json else None
    )
    posterior_std_prior_payload = None
    posterior_std_prior_path = None
    posterior_std_prior = None
    posterior_std_prior_scale = 0.0
    if posterior_std_prior_json_path is not None:
        if not posterior_std_prior_json_path.is_file():
            raise FileNotFoundError(posterior_std_prior_json_path)
        posterior_std_prior_payload = json.loads(
            posterior_std_prior_json_path.read_text(encoding="utf-8")
        )
        posterior_std_prior_path = Path(posterior_std_prior_payload["prior_path"]).resolve()
        if not posterior_std_prior_path.is_file():
            raise FileNotFoundError(posterior_std_prior_path)
        if file_sha256(posterior_std_prior_path) != posterior_std_prior_payload["prior_sha256"]:
            raise ValueError("Posterior std-prior hash does not match its manifest")
        with np.load(posterior_std_prior_path, allow_pickle=False) as prior_payload:
            posterior_std_prior = torch.from_numpy(
                prior_payload[str(posterior_std_prior_payload["prior_key"])]
            ).float().reshape(1, 1, 256, 256)
        posterior_std_prior_scale = float(posterior_std_prior_payload["scale"])

    config = {
        "latent_dim": 12 if args.smoke else int(args.latent_dim),
        "pca_train_count": int(args.pca_train_count),
        "residual_scale": float(args.residual_scale),
        "trust_std": float(args.trust_std),
        "kl_weight": float(args.kl_weight),
        "map_latent_prior_weight": float(args.map_latent_prior_weight),
        "components": min(3, int(args.components)) if args.smoke else int(args.components),
        "bvi_steps": 1 if args.smoke else int(args.bvi_steps),
        "map_steps": 1 if args.smoke else int(args.map_steps),
        "samples_per_component": 2 if args.smoke else int(args.samples_per_component),
        "posterior_samples": 16 if args.smoke else int(args.posterior_samples),
        "mc_samples": 3 if args.smoke else int(args.mc_samples),
        "learning_rate": float(args.learning_rate),
        "initial_log_std": float(args.initial_log_std),
        "candidate_batch_size": int(args.candidate_batch_size),
        "initial_component_weight": float(args.initial_component_weight),
        "component_weight_learning_rate": float(args.component_weight_learning_rate),
        "mixture_line_search_samples": (
            2 if args.smoke else int(args.mixture_line_search_samples)
        ),
        "posterior_latent_std": float(args.posterior_latent_std),
        "noise_std_floor": float(args.noise_std_floor),
        "laplace_fd_eps": float(args.laplace_fd_eps),
        "laplace_std_min": float(args.laplace_std_min),
        "laplace_std_max": float(args.laplace_std_max),
        "laplace_precision_damping": float(args.laplace_precision_damping),
        "basis_type": "error_pca_mean",
        "basis_source": "training_errors_only",
        "latent_prior_source": "training_error_coefficients_only",
        "neural_center": str(args.neural_center),
        "ensemble_dispersion_scale": float(args.ensemble_dispersion_scale),
        "mc_dropout_dispersion_scale": float(args.mc_dropout_dispersion_scale),
        "posterior_mean_alpha": float(args.posterior_mean_alpha),
        "posterior_mean_policy": str(args.posterior_mean_policy),
        "posterior_mean_std_threshold": float(args.posterior_mean_std_threshold),
        "posterior_mean_low_std_alpha": float(args.posterior_mean_low_std_alpha),
        "posterior_mean_high_std_alpha": float(args.posterior_mean_high_std_alpha),
        "posterior_mean_line_search_alphas": parse_float_list(
            args.posterior_mean_line_search_alphas
        ),
        "posterior_mean_line_search_min_gain": float(
            args.posterior_mean_line_search_min_gain
        ),
        "posterior_std_prior_scale": posterior_std_prior_scale,
    }
    if config["neural_center"] == "single":
        basis_center_tag = ""
    elif config["neural_center"] == "ensemble_calibrated":
        basis_center_tag = f"_{config['neural_center']}_{file_sha256(ensemble_center_path)[:12]}"
    else:
        basis_center_tag = f"_{config['neural_center']}"
    basis_path = root / "basis" / (
        f"error_pca_mean{basis_center_tag}_latent{config['latent_dim']}_"
        f"train{config['pca_train_count']}_snrs0-5-10_seed{args.seed}.pt"
    )
    external_calibration_path = (
        args.external_calibration_json.resolve() if args.external_calibration_json else None
    )
    if external_calibration_path is not None and not external_calibration_path.is_file():
        raise FileNotFoundError(external_calibration_path)
    run_spec = {
        "schema_version": 1,
        "protocol_hash": protocol["protocol_hash"],
        "root": str(root),
        "methods": methods,
        "model_count": 1 if args.smoke else int(args.model_count),
        "case_limit": int(args.case_limit),
        "validation_local_indices": parse_local_indices(args.validation_local_indices),
        "test_local_indices": parse_local_indices(args.test_local_indices),
        "external_calibration_json": (
            str(external_calibration_path) if external_calibration_path else None
        ),
        "external_calibration_json_sha256": (
            file_sha256(external_calibration_path) if external_calibration_path else None
        ),
        "ensemble_center_json": str(ensemble_center_path) if ensemble_center_path else None,
        "ensemble_center_json_sha256": (
            file_sha256(ensemble_center_path) if ensemble_center_path else None
        ),
        "ensemble_center_bias": (
            str(ensemble_center_bias_path) if ensemble_center_bias_path else None
        ),
        "ensemble_center_bias_sha256": (
            file_sha256(ensemble_center_bias_path) if ensemble_center_bias_path else None
        ),
        "posterior_std_prior_json": (
            str(posterior_std_prior_json_path) if posterior_std_prior_json_path else None
        ),
        "posterior_std_prior_json_sha256": (
            file_sha256(posterior_std_prior_json_path)
            if posterior_std_prior_json_path
            else None
        ),
        "posterior_std_prior": str(posterior_std_prior_path) if posterior_std_prior_path else None,
        "posterior_std_prior_sha256": (
            file_sha256(posterior_std_prior_path) if posterior_std_prior_path else None
        ),
        "seed": int(args.seed),
        "config": config,
        "basis_path": str(basis_path.resolve()),
    }
    run_spec_path = out_dir / "run_spec.json"
    if args.resume and run_spec_path.exists():
        saved_run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
        if saved_run_spec != run_spec:
            raise ValueError(
                f"Resume configuration does not match {run_spec_path}; use a new output directory"
            )
    elif args.resume and metrics_path.exists():
        raise ValueError(f"Cannot safely resume legacy metrics without {run_spec_path}")
    else:
        run_spec_path.write_text(json.dumps(run_spec, indent=2), encoding="utf-8")
    basis_payload = build_error_pca_basis(
        protocol=protocol,
        artifacts=artifacts,
        model=bvi_center_model,
        latent_dim=int(config["latent_dim"]),
        include_mean=True,
        train_count=int(config["pca_train_count"]),
        snrs=[0.0, 5.0, 10.0],
        seed=int(args.seed),
        cache_path=basis_path,
        device=device,
    )
    basis = basis_payload["basis"]
    latent_prior_mean = basis_payload["latent_raw_mean"] / max(float(config["residual_scale"]), 1.0e-8)
    latent_prior_std = basis_payload["latent_raw_std"] / max(float(config["residual_scale"]), 1.0e-8)
    event_threshold = float(protocol["uq"]["event_threshold_normalized"])
    model_count = 1 if args.smoke else int(args.model_count)
    validation_cases = choose_cases(
        protocol,
        artifacts,
        "validation",
        model_count,
        parse_local_indices(args.validation_local_indices),
    )
    test_cases = choose_cases(
        protocol,
        artifacts,
        "test",
        model_count,
        parse_local_indices(args.test_local_indices),
    )
    validation_local_indices = parse_local_indices(args.validation_local_indices)
    test_local_indices = parse_local_indices(args.test_local_indices)
    if validation_local_indices is not None and len({int(case["global_index"]) for case in validation_cases}) != min(
        model_count, len(validation_local_indices)
    ):
        raise RuntimeError("Validation local-index selection was truncated or duplicated unexpectedly")
    if test_local_indices is not None and len({int(case["global_index"]) for case in test_cases}) != min(
        model_count, len(test_local_indices)
    ):
        raise RuntimeError("Test local-index selection was truncated or duplicated unexpectedly")
    if args.smoke:
        validation_cases = validation_cases[:1]
        test_cases = test_cases[:1]
    elif int(args.case_limit) > 0:
        validation_cases = validation_cases[: int(args.case_limit)]
        test_cases = test_cases[: int(args.case_limit)]

    records = read_csv(metrics_path) if args.resume else []
    experiment_started = time.time()
    if "validation" in splits:
        records = run_split(
            split="validation",
            cases=validation_cases,
            methods=methods,
            records=records,
            metrics_path=metrics_path,
            out_dir=out_dir,
            model=model,
            ensemble_models=ensemble_models,
            ensemble_center_weights=ensemble_center_weights,
            ensemble_center_bias=ensemble_center_bias,
            ensemble_center_bias_scale=ensemble_center_bias_scale,
            ensemble_center_mean_shift=ensemble_center_mean_shift,
            ensemble_center_contrast_correction=ensemble_center_contrast_correction,
            posterior_std_prior=posterior_std_prior,
            posterior_std_prior_scale=posterior_std_prior_scale,
            forward=forward,
            basis=basis,
            latent_prior_mean=latent_prior_mean,
            latent_prior_std=latent_prior_std,
            event_threshold=event_threshold,
            device=device,
            config=config,
            seed=int(args.seed),
        )
    if external_calibration_path:
        calibration_payload = json.loads(external_calibration_path.read_text(encoding="utf-8"))
        validation_scales = dict(calibration_payload["method_scales"])
    else:
        validation_scales = {
            method: calibration_scale_from_records(
                [
                    record
                    for record in records
                    if record.get("split") == "validation" and record.get("method") == method
                ]
            )
            for method in methods
            if method in UQ_METHODS
        }
    if "test" in splits:
        missing_calibration = [
            method
            for method in methods
            if method in UQ_METHODS
            and method not in validation_scales
            and not any(
                record.get("split") == "validation" and record.get("method") == method
                for record in records
            )
        ]
        if missing_calibration:
            raise ValueError(
                f"Test split requires saved validation records for UQ methods: {missing_calibration}"
            )
        records = run_split(
            split="test",
            cases=test_cases,
            methods=methods,
            records=records,
            metrics_path=metrics_path,
            out_dir=out_dir,
            model=model,
            ensemble_models=ensemble_models,
            ensemble_center_weights=ensemble_center_weights,
            ensemble_center_bias=ensemble_center_bias,
            ensemble_center_bias_scale=ensemble_center_bias_scale,
            ensemble_center_mean_shift=ensemble_center_mean_shift,
            ensemble_center_contrast_correction=ensemble_center_contrast_correction,
            posterior_std_prior=posterior_std_prior,
            posterior_std_prior_scale=posterior_std_prior_scale,
            forward=forward,
            basis=basis,
            latent_prior_mean=latent_prior_mean,
            latent_prior_std=latent_prior_std,
            event_threshold=event_threshold,
            device=device,
            config=config,
            seed=int(args.seed),
        )
    calibrated_records = [
        add_calibrated_metrics(record, scale_for_record(validation_scales, record))
        if record.get("split") == "test"
        else record
        for record in records
    ]
    write_csv(metrics_path, calibrated_records)
    test_records = [record for record in calibrated_records if record.get("split") == "test"]
    validation_records = [record for record in calibrated_records if record.get("split") == "validation"]
    validation_method_summary = aggregate(validation_records)
    method_summary = aggregate(test_records)
    statistics = build_statistics(
        test_records,
        resamples=1000 if args.smoke else int(args.bootstrap_resamples),
        seed=int(args.seed) + 20260719,
    )
    planned_statistics = build_planned_statistics(
        test_records,
        resamples=1000 if args.smoke else int(args.bootstrap_resamples),
        seed=int(args.seed) + 20260719,
    )
    write_comparison_table(out_dir / "comparison_table.csv", method_summary)
    write_csv(out_dir / "clustered_bootstrap.csv", statistics)
    write_csv(out_dir / "planned_comparisons.csv", planned_statistics)
    summary = {
        "status": "complete" if "test" in splits else "validation_complete",
        "protocol_hash": protocol["protocol_hash"],
        "device": str(device),
        "forward_backend": forward_payload.get("architecture"),
        "methods": methods,
        "splits": splits,
        "config": config,
        "basis_path": str(basis_path.resolve()),
        "basis_metadata": {
            "kind": "training_error_pca_basis_v2",
            "train_count": int(config["pca_train_count"]),
            "data_boundary": "training errors only; no validation or test labels",
        },
        "validation_model_count": len({int(case["global_index"]) for case in validation_cases}),
        "validation_case_count_selected": len(validation_cases),
        "validation_record_count": len(
            [record for record in calibrated_records if record.get("split") == "validation"]
        ),
        "test_model_count": len({int(case["global_index"]) for case in test_cases}),
        "test_case_count_selected": len(test_cases),
        "test_record_count": len(test_records),
        "validation_scales": validation_scales,
        "validation_method_summary": validation_method_summary,
        "method_summary": method_summary,
        "clustered_bootstrap": statistics,
        "planned_comparisons": planned_statistics,
        "metrics_path": str(metrics_path),
        "comparison_table": str(out_dir / "comparison_table.csv"),
        "planned_comparison_table": str(out_dir / "planned_comparisons.csv"),
        "run_spec": str(run_spec_path),
        "seconds": float(time.time() - experiment_started),
        "claim_boundary": (
            "Noise views are repeated measurements of each subsurface model. Confidence intervals cluster by model. "
            "The scalar-wave matched-physics benchmark does not establish electromagnetic model-mismatch robustness."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
