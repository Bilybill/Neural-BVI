"""Metrics and paired statistics for the LA010010 publication protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from skimage.metrics import structural_similarity


def _gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def deterministic_metrics(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    event_threshold: float = 0.5,
) -> list[dict[str, float]]:
    prediction = prediction.detach().cpu().float()
    truth = truth.detach().cpu().float()
    records = []
    for index in range(len(prediction)):
        pred = prediction[index : index + 1]
        target = truth[index : index + 1]
        diff = pred - target
        pgy, pgx = _gradients(pred)
        tgy, tgx = _gradients(target)
        pred_event = pred >= event_threshold
        true_event = target >= event_threshold
        intersection = float((pred_event & true_event).sum())
        union = float((pred_event | true_event).sum())
        tp = intersection
        fp = float((pred_event & ~true_event).sum())
        fn = float((~pred_event & true_event).sum())
        pred_np = pred.squeeze().numpy()
        target_np = target.squeeze().numpy()
        records.append(
            {
                "normalized_rmse": float(torch.sqrt(torch.mean(diff**2))),
                "epsilon_rmse": float(8.0 * torch.sqrt(torch.mean(diff**2))),
                "normalized_mae": float(torch.mean(torch.abs(diff))),
                "epsilon_mae": float(8.0 * torch.mean(torch.abs(diff))),
                "ssim": float(structural_similarity(target_np, pred_np, data_range=1.0)),
                "gradient_rmse": float(
                    torch.sqrt(0.5 * (F.mse_loss(pgy, tgy) + F.mse_loss(pgx, tgx)))
                ),
                "high_eps_iou": intersection / max(union, 1.0),
                "high_eps_f1": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
            }
        )
    return records


def pearson_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().flatten()
    y = b.detach().float().flatten()
    x = x - x.mean()
    y = y - y.mean()
    return float(torch.sum(x * y) / (torch.sqrt(torch.sum(x**2) * torch.sum(y**2)) + 1.0e-12))


def normalized_data_rmse(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.sqrt(torch.mean((prediction.float() - target.float()) ** 2))
    denominator = torch.sqrt(torch.mean(target.float() ** 2)).clamp_min(1.0e-8)
    return float(numerator / denominator)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    scores = scores.astype(np.float64).ravel()
    positive, negative = int(labels.sum()), int((~labels).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = labels.astype(np.float64).ravel()
    probabilities = np.clip(probabilities.astype(np.float64).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (probabilities < right if right < 1.0 else probabilities <= right)
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return error if total else float("nan")


def ause(std: np.ndarray, abs_error: np.ndarray) -> float:
    uncertainty = std.ravel().astype(np.float64)
    error = abs_error.ravel().astype(np.float64)
    fractions = np.linspace(0.0, 0.9, 20)
    predicted_order = np.argsort(-uncertainty)
    oracle_order = np.argsort(-error)
    predicted_curve, oracle_curve = [], []
    for fraction in fractions:
        remove = int(round(fraction * len(error)))
        keep_pred = predicted_order[remove:]
        keep_oracle = oracle_order[remove:]
        predicted_curve.append(float(error[keep_pred].mean()))
        oracle_curve.append(float(error[keep_oracle].mean()))
    return float(np.trapezoid(np.asarray(predicted_curve) - np.asarray(oracle_curve), fractions))


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> float:
    """Compute the exact empirical-ensemble CRPS without an S-by-S tensor."""

    samples = samples.astype(np.float64)
    truth = truth.astype(np.float64)
    if len(samples) == 0:
        return float("nan")
    first = np.mean(np.abs(samples - truth), axis=0).mean()
    ordered = np.sort(samples, axis=0)
    sample_count = len(ordered)
    coefficients = (2.0 * np.arange(sample_count) - sample_count + 1.0).reshape(
        (sample_count,) + (1,) * (ordered.ndim - 1)
    )
    half_pairwise_term = np.sum(coefficients * ordered, axis=0).mean() / float(sample_count**2)
    return float(first - half_pairwise_term)


def uq_metrics(
    samples: torch.Tensor,
    truth: torch.Tensor,
    event_threshold: float,
    std_scale: float = 1.0,
) -> dict[str, float]:
    samples_np = samples.detach().cpu().numpy().astype(np.float64)
    truth_np = truth.detach().cpu().numpy().astype(np.float64)
    mean = samples_np.mean(axis=0, keepdims=True)
    std = np.maximum(samples_np.std(axis=0, keepdims=True) * std_scale, 1.0e-6)
    low = mean - 1.96 * std
    high = mean + 1.96 * std
    abs_error = np.abs(mean - truth_np)
    event_probability = np.mean(samples_np >= event_threshold, axis=0, keepdims=True)
    event_truth = truth_np >= event_threshold
    standardized = abs_error / std
    if np.ptp(std) <= 1.0e-12 or np.ptp(abs_error) <= 1.0e-12:
        std_error_spearman = float("nan")
    else:
        std_error_spearman = float(stats.spearmanr(std.ravel(), abs_error.ravel()).statistic)
    return {
        "epsilon_rmse": float(8.0 * np.sqrt(np.mean((mean - truth_np) ** 2))),
        "epsilon_mae": float(8.0 * np.mean(abs_error)),
        "crps": crps_ensemble(samples_np, truth_np),
        "gaussian_nll": float(np.mean(0.5 * standardized**2 + np.log(std) + 0.5 * math.log(2.0 * math.pi))),
        "coverage_95": float(np.mean((truth_np >= low) & (truth_np <= high))),
        "mpiw_95": float(np.mean(high - low)),
        "std_error_spearman": std_error_spearman,
        "ause": ause(std, abs_error),
        "event_auroc": binary_auc(event_truth, event_probability),
        "event_brier": float(np.mean((event_probability - event_truth.astype(np.float64)) ** 2)),
        "event_ece": expected_calibration_error(event_truth, event_probability),
    }


def calibration_scale(validation_records: list[dict[str, Any]], target_coverage: float = 0.95) -> float:
    ratios = []
    for record in validation_records:
        truth = np.load(record["arrays"])["truth"].astype(np.float64)
        mean = np.load(record["arrays"])["mean"].astype(np.float64)
        std = np.maximum(np.load(record["arrays"])["std"].astype(np.float64), 1.0e-8)
        ratios.append((np.abs(truth - mean) / std).ravel())
    if not ratios:
        return 1.0
    quantile = float(np.quantile(np.concatenate(ratios), target_coverage))
    return max(quantile / 1.96, 1.0e-3)


def paired_bootstrap(
    baseline: np.ndarray,
    method: np.ndarray,
    resamples: int,
    confidence: float,
    seed: int = 20260706,
) -> dict[str, float]:
    baseline = np.asarray(baseline, dtype=np.float64)
    method = np.asarray(method, dtype=np.float64)
    if baseline.shape != method.shape or baseline.size == 0:
        raise ValueError("Paired bootstrap requires non-empty arrays with the same shape")
    differences = method - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    wilcoxon = stats.wilcoxon(differences, alternative="two-sided", zero_method="zsplit")
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "accuracy_success": bool(float(np.quantile(means, 1.0 - alpha / 2.0)) < 0.0),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        corrected = min(1.0, (total - rank) * float(value))
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted
