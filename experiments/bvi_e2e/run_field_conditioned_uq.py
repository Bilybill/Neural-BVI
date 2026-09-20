"""Build field-conditioned uncertainty candidates for the final LA010010 inversion."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch

from finalize_hyperbola_conditioned_field_bvi import AdaptiveMatchedForward
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, set_seed
from run_deepwave_map_bvi_synthetic import fixed_split_cases


HERE = Path(__file__).resolve().parent
ROOT = HERE / "publication_la010010" / "full"
FIELD_DIR = ROOT / "field_joint_early_layered_bvi"
FIELD_ARRAYS = FIELD_DIR / "field_joint_early_layered_bvi_arrays.npz"
FIELD_SUMMARY = FIELD_DIR / "summary.json"
SURROGATE = ROOT / "surrogate" / "best.pt"
TRAIN_DIR = ROOT / "train" / "unet"
PREPARED = ROOT / "prepared" / "artifacts.pt"
PROTOCOL = ROOT / "protocol.snapshot.json"
SYNTHETIC_UQ = ROOT / "physics_map_bvi_laplace_errorpca_mean_res0003_prior0005_steps8_test12_uqstd"
OUT_DIR = ROOT / "field_conditioned_uq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--rbf-nx", type=int, default=8)
    parser.add_argument("--rbf-nz", type=int, default=8)
    parser.add_argument("--fd-model-step", type=float, default=0.003)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--selected-local-scale", type=float, default=0.06)
    parser.add_argument("--posterior-samples", type=int, default=512)
    parser.add_argument("--predictive-samples", type=int, default=16)
    parser.add_argument("--ensemble-validation-nrmse-max", type=float, default=0.105)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--reuse-jacobian", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def save_all_formats(fig: mpl.figure.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel().copy()
    y = np.asarray(right, dtype=np.float64).ravel().copy()
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / max(denominator, 1.0e-12))


def make_l2_normalized_rbf_basis(
    height: int,
    width: int,
    depth_m: float,
    width_m: float,
    nx: int,
    nz: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.linspace(0.0, width_m, width)
    z = np.linspace(0.0, depth_m, height)
    x_centers = np.linspace(0.0, width_m, int(nx))
    if int(nz) == 8:
        z_centers = np.asarray([0.06, 0.20, 0.40, 0.65, 0.95, 1.30, 1.75, 2.25])
    else:
        z_centers = np.linspace(0.04, depth_m - 0.08, int(nz))
    sigma_x = 0.72 * width_m / max(int(nx) - 1, 1)
    local_spacing = np.gradient(z_centers)
    atoms = []
    centers = []
    for iz, center_z in enumerate(z_centers):
        sigma_z = max(0.08, 0.72 * float(local_spacing[iz]))
        for center_x in x_centers:
            atom = np.exp(
                -0.5 * ((z[:, None] - center_z) / sigma_z) ** 2
                -0.5 * ((x[None, :] - center_x) / sigma_x) ** 2
            )
            atoms.append(atom)
            centers.append([float(center_x), float(center_z), float(sigma_x), float(sigma_z)])
    basis = np.stack(atoms).astype(np.float64)
    pointwise_norm = np.sqrt(np.sum(np.square(basis), axis=0)).clip(1.0e-8)
    basis /= pointwise_norm[None]
    return basis.astype(np.float32), {
        "kind": "localized_l2_normalized_gaussian_rbf",
        "nx": int(nx),
        "nz": int(nz),
        "centers": centers,
        "pointwise_prior_variance_min": float(np.sum(np.square(basis), axis=0).min()),
        "pointwise_prior_variance_max": float(np.sum(np.square(basis), axis=0).max()),
    }


def predict_in_batches(model: torch.nn.Module, observations: torch.Tensor, device: torch.device) -> torch.Tensor:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(observations), 8):
            predictions.append(model(observations[start : start + 8].to(device)).cpu())
    return torch.cat(predictions, dim=0)


def build_network_ensemble(
    field_observation: torch.Tensor,
    device: torch.device,
    validation_nrmse_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    protocol = load_json(PROTOCOL)
    artifacts = torch.load(PREPARED, map_location="cpu", weights_only=False)
    validation_cases = fixed_split_cases(protocol, artifacts, "validation")
    validation_observation = torch.cat([case["observation"] for case in validation_cases], dim=0).float()
    validation_truth = torch.cat([case["truth"] for case in validation_cases], dim=0).float()

    field_members = []
    validation_members = []
    member_records = []
    checkpoints = []
    checkpoint_records = []
    for architecture in ("unet", "unetpp", "transunet", "tinynet"):
        for checkpoint in sorted((ROOT / "train" / architecture).glob("seed_*/best.pt")):
            training_summary = load_json(checkpoint.parent / "summary.json")
            validation_nrmse = float(training_summary["best_validation_normalized_rmse"])
            if validation_nrmse <= float(validation_nrmse_max):
                checkpoints.append(checkpoint)
                checkpoint_records.append((architecture, validation_nrmse))
    member_architectures = []
    for checkpoint, (architecture, validation_nrmse) in zip(checkpoints, checkpoint_records):
        model, payload = load_inversion_checkpoint(checkpoint, device)
        field_members.append(predict_in_batches(model, field_observation, device).numpy()[0, 0])
        validation_members.append(predict_in_batches(model, validation_observation, device).numpy()[:, 0])
        member_records.append(
            {
                "checkpoint": str(checkpoint),
                "architecture": architecture,
                "seed": int(payload.get("seed", checkpoint.parent.name.split("_")[-1])),
                "best_validation_normalized_rmse": validation_nrmse,
            }
        )
        member_architectures.append(architecture)
        del model
        torch.cuda.empty_cache()

    field_stack = np.stack(field_members).astype(np.float32)
    validation_stack = np.stack(validation_members).astype(np.float32)
    validation_mean = validation_stack.mean(axis=0)
    validation_std = validation_stack.std(axis=0, ddof=0)
    truth = validation_truth.numpy()[:, 0]
    floor = 0.005
    stabilized_std = np.sqrt(np.square(validation_std) + floor**2)
    standardized_error = np.abs(validation_mean - truth) / stabilized_std
    scale_95 = float(np.quantile(standardized_error, 0.95) / 1.96)
    scale_mle = float(np.sqrt(np.mean(np.square((validation_mean - truth) / stabilized_std))))
    coverage_raw = float(np.mean(np.abs(validation_mean - truth) <= 1.96 * stabilized_std))
    coverage_scaled = float(np.mean(np.abs(validation_mean - truth) <= 1.96 * scale_95 * stabilized_std))
    return field_stack, np.asarray(member_architectures), {
        "member_count": len(field_members),
        "validation_nrmse_max": float(validation_nrmse_max),
        "members": member_records,
        "validation_case_count": len(validation_cases),
        "normalized_std_floor": floor,
        "scale_95": scale_95,
        "scale_mle": scale_mle,
        "raw_95pct_coverage": coverage_raw,
        "scaled_95pct_coverage": coverage_scaled,
    }


def ensemble_leave_one_out_stability(
    centered_deviations: np.ndarray,
    weights: np.ndarray,
    full_std: np.ndarray,
) -> dict[str, Any]:
    correlations = []
    relative_l2 = []
    for omitted in range(len(centered_deviations)):
        keep = np.arange(len(centered_deviations)) != omitted
        subset = centered_deviations[keep]
        subset_weights = weights[keep].astype(np.float64)
        subset_weights /= subset_weights.sum()
        center = np.einsum("s,shw->hw", subset_weights, subset)
        variance = np.einsum("s,shw->hw", subset_weights, np.square(subset - center[None]))
        std = np.sqrt(np.maximum(scipy.ndimage.gaussian_filter(variance, sigma=1.25, mode="nearest"), 0.0))
        correlations.append(pearson(std, full_std))
        relative_l2.append(float(np.linalg.norm(std - full_std) / max(np.linalg.norm(full_std), 1.0e-12)))
    return {
        "leave_one_out_count": len(correlations),
        "map_pearson_min": float(np.min(correlations)),
        "map_pearson_mean": float(np.mean(correlations)),
        "relative_l2_max": float(np.max(relative_l2)),
        "relative_l2_mean": float(np.mean(relative_l2)),
    }


def compute_field_weighted_ensemble(
    members: np.ndarray,
    base: torch.Tensor,
    matched_forward: torch.nn.Module,
    observation: torch.Tensor,
    noise_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    member_mean = members.mean(axis=0)
    deviations = members - member_mean[None]
    shifted = np.clip(base.cpu().numpy()[0, 0][None] + deviations, 0.0, 1.0).astype(np.float32)
    losses = []
    rmses = []
    with torch.no_grad():
        for model in shifted:
            prediction = matched_forward(torch.from_numpy(model)[None, None].to(device))
            mse = torch.mean((prediction - observation.to(device)).square())
            losses.append(float(mse / (2.0 * float(noise_std) ** 2)))
            rmses.append(float(torch.sqrt(mse)))
    losses_np = np.asarray(losses, dtype=np.float64)
    weights = np.exp(-(losses_np - losses_np.min()))
    weights /= weights.sum()
    weighted_center = np.einsum("s,shw->hw", weights, deviations)
    centered = deviations - weighted_center[None]
    variance = np.einsum("s,shw->hw", weights, np.square(centered))
    variance = scipy.ndimage.gaussian_filter(variance, sigma=1.25, mode="nearest")
    std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
    return std, centered.astype(np.float32), weights.astype(np.float32), {
        "matched_rmse": rmses,
        "negative_log_likelihood": losses,
        "effective_member_count": float(1.0 / np.sum(np.square(weights))),
        "weights": weights.tolist(),
        "spatial_variance_smoothing_sigma_pixels": 1.25,
    }


def compute_unit_jacobian_gram(
    basis: np.ndarray,
    base: torch.Tensor,
    matched_forward: torch.nn.Module,
    device: torch.device,
    fd_model_step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    columns = []
    clipping = []
    started = time.time()
    with torch.no_grad():
        for index, atom in enumerate(basis):
            direction = torch.from_numpy(atom)[None, None].to(device)
            plus = torch.clamp(base.to(device) + float(fd_model_step) * direction, 0.0, 1.0)
            minus = torch.clamp(base.to(device) - float(fd_model_step) * direction, 0.0, 1.0)
            pred_plus = matched_forward(plus)
            pred_minus = matched_forward(minus)
            derivative = ((pred_plus - pred_minus) / (2.0 * float(fd_model_step))).reshape(-1)
            columns.append(derivative.cpu().double())
            clipping.append(
                float(
                    torch.mean(
                        ((plus <= 0.0) | (plus >= 1.0) | (minus <= 0.0) | (minus >= 1.0)).float()
                    )
                )
            )
            if (index + 1) % 8 == 0 or index + 1 == len(basis):
                print(f"Field Jacobian columns {index + 1:03d}/{len(basis):03d}", flush=True)
    jacobian = torch.stack(columns, dim=1)
    gram_mean = (jacobian.T @ jacobian) / float(jacobian.shape[0])
    return gram_mean.numpy(), {
        "column_count": int(jacobian.shape[1]),
        "data_dimension": int(jacobian.shape[0]),
        "fd_model_step_normalized": float(fd_model_step),
        "maximum_clipped_fraction": float(max(clipping)),
        "seconds": time.time() - started,
    }


def posterior_from_gram(
    basis: np.ndarray,
    gram_mean: np.ndarray,
    scale: float,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    latent_dim = basis.shape[0]
    precision = np.eye(latent_dim) + (float(scale) ** 2 / float(noise_std) ** 2) * gram_mean
    eigvals, eigvecs = np.linalg.eigh(0.5 * (precision + precision.T))
    eigvals = np.maximum(eigvals, 1.0e-10)
    covariance = (eigvecs / eigvals[None]) @ eigvecs.T
    flat_basis = basis.reshape(latent_dim, -1).astype(np.float64)
    transformed = np.linalg.cholesky(covariance).T @ flat_basis
    posterior_variance = float(scale) ** 2 * np.sum(np.square(transformed), axis=0)
    posterior_std = np.sqrt(np.maximum(posterior_variance, 0.0)).reshape(basis.shape[1:])
    prior_variance = float(scale) ** 2 * np.sum(np.square(basis.astype(np.float64)), axis=0)
    contraction = 1.0 - posterior_variance.reshape(basis.shape[1:]) / np.maximum(prior_variance, 1.0e-12)
    return posterior_std.astype(np.float32), contraction.astype(np.float32), covariance.astype(np.float32), {
        "scale_normalized": float(scale),
        "prior_std_epsilon": 8.0 * float(scale),
        "noise_std": float(noise_std),
        "precision_condition": float(eigvals.max() / eigvals.min()),
        "precision_data_trace": float(np.trace(precision - np.eye(latent_dim))),
        "mean_variance_contraction": float(contraction.mean()),
        "q95_variance_contraction": float(np.quantile(contraction, 0.95)),
    }


def load_synthetic_std_mean() -> np.ndarray:
    summary = load_json(SYNTHETIC_UQ / "summary.json")
    maps = []
    for record in summary["records"]:
        with np.load(record["arrays"]) as arrays:
            maps.append(8.0 * np.squeeze(arrays["std"]).astype(np.float64))
    return np.mean(maps, axis=0).astype(np.float32)


def map_metrics(
    std_epsilon: np.ndarray,
    old_std: np.ndarray,
    synthetic_std: np.ndarray,
    edge: np.ndarray,
    depth: np.ndarray,
) -> dict[str, float]:
    high_edge = edge >= np.quantile(edge, 0.90)
    low_edge = edge <= np.quantile(edge, 0.50)
    return {
        "mean_epsilon": float(std_epsilon.mean()),
        "q50_epsilon": float(np.quantile(std_epsilon, 0.50)),
        "q95_epsilon": float(np.quantile(std_epsilon, 0.95)),
        "q995_epsilon": float(np.quantile(std_epsilon, 0.995)),
        "max_epsilon": float(std_epsilon.max()),
        "pearson_with_old_pca_std": pearson(std_epsilon, old_std),
        "pearson_with_synthetic_std_mean": pearson(std_epsilon, synthetic_std),
        "pearson_with_inversion_edge": pearson(std_epsilon, edge),
        "edge_q90_to_low_q50_mean_ratio": float(std_epsilon[high_edge].mean() / max(std_epsilon[low_edge].mean(), 1.0e-8)),
        "top_0_0p5m_mean": float(std_epsilon[depth < 0.5].mean()),
        "target_0p9_1p8m_mean": float(std_epsilon[(depth >= 0.9) & (depth < 1.8)].mean()),
        "deep_1p8_2p5m_mean": float(std_epsilon[depth >= 1.8].mean()),
    }


def sample_total_models(
    base: np.ndarray,
    basis: np.ndarray,
    covariance: np.ndarray,
    local_scale: float,
    ensemble_deviations: np.ndarray,
    ensemble_weights: np.ndarray,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    latent = rng.multivariate_normal(np.zeros(len(basis)), covariance, size=int(sample_count)).astype(np.float32)
    member_indices = rng.choice(len(ensemble_deviations), size=int(sample_count), p=ensemble_weights)
    model_sum = np.zeros_like(base, dtype=np.float64)
    model_square_sum = np.zeros_like(base, dtype=np.float64)
    keep = []
    for start in range(0, int(sample_count), 16):
        stop = min(start + 16, int(sample_count))
        local = np.einsum("sd,dhw->shw", latent[start:stop], basis) * float(local_scale)
        models = np.clip(
            base[None] + local + ensemble_deviations[member_indices[start:stop]],
            0.0,
            1.0,
        ).astype(np.float32)
        model_sum += models.sum(axis=0)
        model_square_sum += np.square(models, dtype=np.float64).sum(axis=0)
        if len(keep) < 128:
            keep.append(models[: min(len(models), 128 - sum(len(item) for item in keep))])
    mean = model_sum / float(sample_count)
    variance = model_square_sum / float(sample_count) - np.square(mean)
    return (
        mean.astype(np.float32),
        np.sqrt(np.maximum(variance, 0.0)).astype(np.float32),
        np.concatenate(keep, axis=0).astype(np.float32),
    )


def predictive_audit(
    samples: np.ndarray,
    matched_forward: torch.nn.Module,
    observation: torch.Tensor,
    device: torch.device,
    sample_count: int,
) -> dict[str, float]:
    indices = np.linspace(0, len(samples) - 1, min(int(sample_count), len(samples)), dtype=int)
    predictions = []
    rmses = []
    observation_device = observation.to(device)
    with torch.no_grad():
        for index in indices:
            prediction = matched_forward(torch.from_numpy(samples[index])[None, None].to(device))
            predictions.append(prediction.cpu())
            rmses.append(float(torch.sqrt(torch.mean((prediction - observation_device) ** 2))))
    stack = torch.cat(predictions, dim=0)
    std = stack.std(dim=0, unbiased=False)
    return {
        "sample_count": len(indices),
        "matched_rmse_mean": float(np.mean(rmses)),
        "matched_rmse_min": float(np.min(rmses)),
        "matched_rmse_max": float(np.max(rmses)),
        "bscan_std_mean": float(std.mean()),
        "bscan_std_q95": float(torch.quantile(std, 0.95)),
        "bscan_std_max": float(std.max()),
    }


def add_image(
    axis: mpl.axes.Axes,
    image: np.ndarray,
    title: str,
    extent: tuple[float, float, float, float],
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
) -> None:
    handle = axis.imshow(image, extent=extent, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("Distance (m)")
    axis.set_ylabel("Depth (m)")
    cbar = axis.figure.colorbar(handle, ax=axis, fraction=0.046, pad=0.025)
    cbar.set_label(colorbar_label)


def render_diagnostic(
    out_dir: Path,
    final_epsilon: np.ndarray,
    old_std: np.ndarray,
    local_std: np.ndarray,
    contraction: np.ndarray,
    ensemble_std: np.ndarray,
    total_std: np.ndarray,
    calibrated_total_std: np.ndarray,
    edge: np.ndarray,
    width_m: float,
    depth_m: float,
) -> None:
    configure_matplotlib()
    extent = (0.0, float(width_m), float(depth_m), 0.0)
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.05), constrained_layout=True)
    add_image(axes[0, 0], final_epsilon, "Neural-BVI mean", extent, "viridis", 2.2, 6.2, r"$\epsilon_r$")
    for axis, image, title in zip(
        axes[0, 1:],
        (old_std, local_std, contraction),
        ("Synthetic-PCA std", "Field Jacobian std", "Variance contraction"),
    ):
        if title == "Variance contraction":
            add_image(axis, image, title, extent, "cividis", 0.0, max(float(np.quantile(image, 0.995)), 0.01), "fraction")
        else:
            add_image(axis, image, title, extent, "magma", 0.0, max(float(np.quantile(image, 0.995)), 0.02), r"$\sigma_{\epsilon_r}$")
    for axis, image, title in zip(
        axes[1],
        (ensemble_std, total_std, calibrated_total_std, edge),
        ("Field ensemble std", "Combined structural std", "Calibrated total std", "Mean-model edge"),
    ):
        label = r"$|\nabla\epsilon_r|$" if title == "Mean-model edge" else r"$\sigma_{\epsilon_r}$"
        cmap = "inferno" if title == "Mean-model edge" else "magma"
        add_image(axis, image, title, extent, cmap, 0.0, max(float(np.quantile(image, 0.995)), 0.02), label)
    labels = "abcdefgh"
    for label, axis in zip(labels, axes.ravel()):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontweight="bold", fontsize=8)
    save_all_formats(fig, out_dir / "fig_field_conditioned_uq_candidates")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Field-conditioned Deepwave UQ requires CUDA")
    set_seed(int(args.seed))

    field_summary = load_json(FIELD_SUMMARY)
    profile = load_json(ROOT / "field_laplace_nn" / "noise_0p080" / "summary.json")["forward_backend"]["acquisition_profile"]
    with np.load(FIELD_ARRAYS) as source:
        observation = torch.from_numpy(source["observation"]).float()
        base = torch.from_numpy(source["posterior_mean"]).float()
        final_epsilon = 2.0 + 8.0 * source["posterior_mean"].squeeze().astype(np.float64)
        old_std = 8.0 * source["posterior_std"].squeeze().astype(np.float64)
    height, width = final_epsilon.shape
    depth_m = float(profile["model_domain"]["target_depth_m"])
    width_m = float(profile["model_domain"]["width_m"])
    depth = np.linspace(0.0, depth_m, height)
    gradient_z, gradient_x = np.gradient(final_epsilon, depth_m / (height - 1), width_m / (width - 1))
    edge = np.hypot(gradient_x, gradient_z)
    synthetic_std = load_synthetic_std_mean()

    forward, _ = load_surrogate_checkpoint(SURROGATE, device)
    matched_forward = AdaptiveMatchedForward(forward, observation.to(device))
    field_members, member_architectures, ensemble_calibration = build_network_ensemble(
        observation,
        device,
        float(args.ensemble_validation_nrmse_max),
    )
    ensemble_std, ensemble_deviations, ensemble_weights, ensemble_field = compute_field_weighted_ensemble(
        field_members,
        base,
        matched_forward,
        observation,
        float(args.noise_std),
        device,
    )
    ensemble_stability = ensemble_leave_one_out_stability(
        ensemble_deviations,
        ensemble_weights,
        ensemble_std,
    )
    unet_mask = member_architectures == "unet"
    unet_std, _unet_deviations, _unet_weights, unet_field = compute_field_weighted_ensemble(
        field_members[unet_mask],
        base,
        matched_forward,
        observation,
        float(args.noise_std),
        device,
    )

    basis, basis_meta = make_l2_normalized_rbf_basis(
        height,
        width,
        depth_m,
        width_m,
        int(args.rbf_nx),
        int(args.rbf_nz),
    )
    jacobian_cache = out_dir / f"jacobian_gram_rbf{int(args.rbf_nx)}x{int(args.rbf_nz)}.npz"
    if args.reuse_jacobian and jacobian_cache.exists():
        with np.load(jacobian_cache) as cached:
            gram_mean = cached["gram_mean"].astype(np.float64)
        jacobian_meta = load_json(out_dir / "jacobian_summary.json")
    else:
        gram_mean, jacobian_meta = compute_unit_jacobian_gram(
            basis,
            base,
            matched_forward,
            device,
            float(args.fd_model_step),
        )
        np.savez_compressed(jacobian_cache, gram_mean=gram_mean)
        (out_dir / "jacobian_summary.json").write_text(json.dumps(jacobian_meta, indent=2), encoding="utf-8")

    candidate_records = []
    candidate_payloads = {}
    for scale in (0.02, 0.04, 0.06, 0.08):
        local_std_norm, contraction, covariance, record = posterior_from_gram(
            basis,
            gram_mean,
            scale,
            float(args.noise_std),
        )
        local_std_epsilon = 8.0 * local_std_norm
        record["map_metrics"] = map_metrics(local_std_epsilon, old_std, synthetic_std, edge, depth)
        candidate_records.append(record)
        candidate_payloads[float(scale)] = (local_std_norm, contraction, covariance)

    selected_scale = float(args.selected_local_scale)
    if selected_scale not in candidate_payloads:
        raise ValueError("selected-local-scale must be one of 0.02, 0.04, 0.06, or 0.08")
    local_std_norm, contraction, covariance = candidate_payloads[selected_scale]
    local_std_epsilon = 8.0 * local_std_norm
    ensemble_std_epsilon = 8.0 * ensemble_std
    total_structural_std = np.sqrt(np.square(local_std_epsilon) + np.square(ensemble_std_epsilon))
    field_floor = float(ensemble_calibration["normalized_std_floor"])
    ensemble_calibrated = 8.0 * float(ensemble_calibration["scale_95"]) * np.sqrt(np.square(ensemble_std) + field_floor**2)
    calibrated_total_std = np.sqrt(np.square(local_std_epsilon) + np.square(ensemble_calibrated))

    total_mean, total_sample_std, saved_samples = sample_total_models(
        base.numpy()[0, 0],
        basis,
        covariance,
        selected_scale,
        ensemble_deviations,
        ensemble_weights,
        int(args.posterior_samples),
        int(args.seed) + 31,
    )
    predictive = predictive_audit(
        saved_samples,
        matched_forward,
        observation,
        device,
        int(args.predictive_samples),
    )
    sampled_total_std_epsilon = 8.0 * total_sample_std

    metrics = {
        "current_synthetic_pca": map_metrics(old_std, old_std, synthetic_std, edge, depth),
        "field_jacobian": map_metrics(local_std_epsilon, old_std, synthetic_std, edge, depth),
        "field_ensemble": map_metrics(ensemble_std_epsilon, old_std, synthetic_std, edge, depth),
        "same_backbone_unet_ensemble": map_metrics(8.0 * unet_std, old_std, synthetic_std, edge, depth),
        "combined_structural": map_metrics(total_structural_std, old_std, synthetic_std, edge, depth),
        "combined_samples": map_metrics(sampled_total_std_epsilon, old_std, synthetic_std, edge, depth),
        "calibrated_total": map_metrics(calibrated_total_std, old_std, synthetic_std, edge, depth),
    }
    sample_consistency = {
        "analytic_vs_sampled_total_std_pearson": pearson(total_structural_std, sampled_total_std_epsilon),
        "analytic_vs_sampled_total_std_relative_l2": float(
            np.linalg.norm(total_structural_std - sampled_total_std_epsilon)
            / max(np.linalg.norm(total_structural_std), 1.0e-12)
        ),
        "saved_sample_count": int(len(saved_samples)),
        "generated_sample_count": int(args.posterior_samples),
    }

    np.savez_compressed(
        out_dir / "field_conditioned_uq_arrays.npz",
        posterior_mean=base.numpy(),
        final_epsilon=final_epsilon,
        old_pca_std_epsilon=old_std,
        synthetic_std_mean_epsilon=synthetic_std,
        localized_basis=basis,
        localized_covariance=covariance,
        localized_std_epsilon=local_std_epsilon,
        variance_contraction=contraction,
        ensemble_members=field_members,
        ensemble_member_architectures=member_architectures,
        ensemble_deviations=ensemble_deviations,
        ensemble_weights=ensemble_weights,
        ensemble_std_epsilon=ensemble_std_epsilon,
        combined_structural_std_epsilon=total_structural_std,
        calibrated_total_std_epsilon=calibrated_total_std,
        sampled_total_mean=total_mean,
        sampled_total_std_epsilon=sampled_total_std_epsilon,
        posterior_samples=saved_samples[:, None],
        inversion_edge=edge,
    )
    render_diagnostic(
        out_dir,
        final_epsilon,
        old_std,
        local_std_epsilon,
        contraction,
        ensemble_std_epsilon,
        total_structural_std,
        calibrated_total_std,
        edge,
        width_m,
        depth_m,
    )

    summary = {
        "status": "complete",
        "seconds": time.time() - started,
        "device": str(device),
        "field_result_source": str(FIELD_ARRAYS),
        "field_result_status": field_summary["status"],
        "method": (
            "law-of-total-variance field UQ: localized field-Deepwave Jacobian Laplace plus "
            "field-input five-checkpoint neural epistemic ensemble"
        ),
        "basis": basis_meta,
        "jacobian": jacobian_meta,
        "ensemble_calibration": ensemble_calibration,
        "ensemble_field_weighting": ensemble_field,
        "same_backbone_unet_field_weighting": unet_field,
        "ensemble_leave_one_out_stability": ensemble_stability,
        "local_candidates": candidate_records,
        "selected_local_scale": selected_scale,
        "metrics": metrics,
        "sample_consistency": sample_consistency,
        "posterior_predictive": predictive,
        "claim_boundary": (
            "The combined structural standard deviation is field-input- and field-Jacobian-conditioned, but it remains "
            "a local scalar-Deepwave approximation. The five-network component is calibrated on synthetic validation "
            "data and is reported separately from the uncalibrated structural map; field interval coverage is unknown."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "selected_local_scale": selected_scale,
                "metrics": metrics,
                "sample_consistency": sample_consistency,
                "posterior_predictive": predictive,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
