"""Matched Deepwave full-Laplace inference for the LA010010 measured record."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepwave_physics_surrogate import preprocess_bscan
from path_utils import resolve_data_path
from publication_data import resolve_protocol_path
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, set_seed
from run_deepwave_map_bvi_synthetic import build_error_pca_basis, run_map_bvi_case


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "publication_la010010" / "full"


def acquisition_profile(protocol: dict[str, Any]) -> dict[str, Any]:
    path = resolve_protocol_path(Path(protocol["protocol_path"]), protocol["acquisition_profile"])
    profiles = json.loads(path.read_text(encoding="utf-8"))
    return profiles["la010010_pipe_native"]


def read_rd3_pair(rad_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    for raw_line in rad_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        try:
            number = float(value)
            metadata[key] = int(number) if number.is_integer() else number
        except ValueError:
            metadata[key] = value
    rd3_path = rad_path.with_suffix(".RD3")
    if not rd3_path.exists():
        rd3_path = rad_path.with_suffix(".rd3")
    n_samples = int(metadata["samples"])
    raw = np.fromfile(rd3_path, dtype="<i2").astype(np.float32)
    n_traces = int(metadata.get("last_trace", 0)) or raw.size // n_samples
    if raw.size != n_samples * n_traces:
        raise ValueError(f"{rd3_path} has an unexpected number of int16 values")
    return raw.reshape(n_traces, n_samples).T, {
        "rad_file": str(rad_path),
        "n_samples": n_samples,
        "n_traces": n_traces,
        "timewindow_ns": float(metadata["timewindow"]),
        "distance_interval_m": float(metadata["distance_interval"]),
    }


def align_ramac_observation(
    raw: np.ndarray,
    header: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    obs = profile["observation"]
    reference = profile["field_reference"]
    if reference.get("crop_mode") != "fixed_native":
        raise ValueError("The matched field runner requires a fixed-native acquisition profile")
    if Path(header["rad_file"]).name.lower() != Path(reference["relative_rad_path"]).name.lower():
        raise ValueError("The measured record does not match the frozen acquisition profile")
    expected_shape = (int(reference["expected_raw_samples"]), int(reference["expected_raw_traces"]))
    if raw.shape != expected_shape:
        raise ValueError(f"Raw field shape {raw.shape} does not match {expected_shape}")
    source_dt = float(header["timewindow_ns"]) * 1.0e-9 / raw.shape[0]
    if not np.isclose(source_dt, float(obs["sample_interval_s"]), rtol=0.0, atol=1.0e-15):
        raise ValueError("Measured sample interval does not match the frozen acquisition profile")
    if not np.isclose(
        float(header["distance_interval_m"]),
        float(obs["trace_spacing_m"]),
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("Measured trace spacing does not match the frozen acquisition profile")
    sample_start = int(reference["sample_start"])
    sample_end = int(reference["sample_end"])
    trace_start = int(reference["trace_start"])
    trace_end = int(reference["trace_end"])
    aligned = raw[sample_start:sample_end, trace_start:trace_end]
    target_shape = (int(obs["n_time"]), int(obs["n_traces"]))
    if aligned.shape != target_shape:
        raise ValueError(f"Fixed-native crop {aligned.shape} does not match {target_shape}")
    aligned = preprocess_bscan(aligned, profile)
    return aligned, {
        "mode": "fixed_native_no_resampling",
        "sample_start": sample_start,
        "sample_end": sample_end,
        "trace_start": trace_start,
        "trace_end": trace_end,
        "start_distance_m": float(reference["start_distance_m"]),
        "target_time_window_ns": target_shape[0] * float(obs["sample_interval_s"]) * 1.0e9,
        "target_aperture_m": (target_shape[1] - 1) * float(obs["trace_spacing_m"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--posterior-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--center-mode", choices=["nn", "empirical"], default="nn")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Rebuild the field figure from saved arrays without rerunning inference.",
    )
    return parser.parse_args()


def save_field_figure(
    path: Path,
    *,
    observation: np.ndarray,
    nn_prediction: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_std: np.ndarray,
    event_probability: np.ndarray,
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
    width_m = float(profile["model_domain"]["width_m"])
    depth_m = float(profile["model_domain"]["target_depth_m"])
    start_m = float(crop["start_distance_m"])
    aperture_m = float(crop["target_aperture_m"])
    time_ns = float(crop["target_time_window_ns"])
    eps_nn = 2.0 + 8.0 * nn_prediction
    eps_mean = 2.0 + 8.0 * posterior_mean
    eps_std = 8.0 * posterior_std
    correction = np.abs(eps_mean - eps_nn)

    fig = plt.figure(figsize=(7.16, 3.30), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        3,
        left=0.060,
        right=0.990,
        bottom=0.175,
        top=0.950,
        wspace=0.29,
        hspace=0.78,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]

    obs_limit = float(np.quantile(np.abs(observation), 0.995))
    im_obs = axes[0].imshow(
        observation,
        cmap="gray",
        vmin=-obs_limit,
        vmax=obs_limit,
        aspect="auto",
        extent=(start_m, start_m + aperture_m, time_ns, 0.0),
    )
    del im_obs
    axes[0].set_title("Measured B-scan")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Time (ns)")

    model_extent = (0.0, width_m, depth_m, 0.0)
    im_nn = axes[1].imshow(eps_nn, cmap="viridis", vmin=2.0, vmax=10.0, extent=model_extent, aspect="auto")
    axes[1].set_title("Neural estimate")
    im_mean = axes[2].imshow(eps_mean, cmap="viridis", vmin=2.0, vmax=10.0, extent=model_extent, aspect="auto")
    axes[2].set_title("Posterior mean")
    std_max = max(float(np.quantile(eps_std, 0.995)), 1.0e-6)
    im_std = axes[3].imshow(eps_std, cmap="magma", vmin=0.0, vmax=std_max, extent=model_extent, aspect="auto")
    axes[3].set_title("Posterior std.")
    correction_max = max(float(np.quantile(correction, 0.995)), 1.0e-6)
    im_correction = axes[4].imshow(
        correction,
        cmap="inferno",
        vmin=0.0,
        vmax=correction_max,
        extent=model_extent,
        aspect="auto",
    )
    axes[4].set_title(r"$|\overline{\epsilon}_r-\epsilon_{r,\mathrm{NN}}|$")
    im_event = axes[5].imshow(
        event_probability,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        extent=model_extent,
        aspect="auto",
    )
    axes[5].set_title(r"$P(\epsilon_r>6)$")

    for axis in axes[:3]:
        axis.set_xlabel("")
    for axis in axes[3:]:
        axis.set_xlabel("Distance (m)", labelpad=2)
    axes[1].set_ylabel("Depth (m)")
    axes[3].set_ylabel("Depth (m)")
    for axis in (axes[2], axes[4], axes[5]):
        axis.set_ylabel("")
    fig.canvas.draw()
    top_y = axes[1].get_position().y0 - 0.058
    bottom_y = axes[3].get_position().y0 - 0.120
    shared_box = axes[1].get_position()
    mean_box = axes[2].get_position()
    std_box = axes[3].get_position()
    correction_box = axes[4].get_position()
    event_box = axes[5].get_position()
    colorbars = [
        fig.colorbar(
            im_nn,
            cax=fig.add_axes([shared_box.x0, top_y, mean_box.x1 - shared_box.x0, 0.013]),
            orientation="horizontal",
        ),
        fig.colorbar(
            im_std,
            cax=fig.add_axes([std_box.x0, bottom_y, std_box.width, 0.013]),
            orientation="horizontal",
        ),
        fig.colorbar(
            im_correction,
            cax=fig.add_axes([correction_box.x0, bottom_y, correction_box.width, 0.013]),
            orientation="horizontal",
        ),
        fig.colorbar(
            im_event,
            cax=fig.add_axes([event_box.x0, bottom_y, event_box.width, 0.013]),
            orientation="horizontal",
        ),
    ]
    for colorbar, label in zip(colorbars, [r"$\epsilon_r$", r"$\sigma_{\epsilon_r}$", r"$|\Delta\epsilon_r|$", "Probability"]):
        colorbar.set_label(label, labelpad=1)
        colorbar.ax.tick_params(labelsize=5.5, pad=1)
    for label, axis in zip("abcdef", axes[:6]):
        axis.text(
            -0.13,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_sensitivity_summary(parent: Path) -> Path:
    records = []
    for path in sorted(parent.glob("noise_*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "noise_std": float(payload["config"]["noise_std"]),
                "data_misfit_nn": float(payload["result"]["data_misfit_nn"]),
                "data_misfit_posterior_mean": float(payload["result"]["data_misfit_bvi_mean"]),
                "relative_misfit_reduction": float(payload["diagnostics"]["data_misfit_relative_reduction"]),
                "correction_rms_normalized": float(payload["diagnostics"]["correction_rms_normalized"]),
                "posterior_std_mean_normalized": float(payload["diagnostics"]["posterior_std_mean_normalized"]),
                "event_area_at_p50": float(payload["diagnostics"]["event_area_at_p50"]),
                "summary": str(path),
            }
        )
    expected = {0.05, 0.08, 0.12}
    available = {round(record["noise_std"], 2) for record in records}
    reductions = [record["relative_misfit_reduction"] for record in records]
    summary = {
        "status": "complete" if expected.issubset(available) else "partial",
        "center_mode": parent.name.removeprefix("field_laplace_"),
        "nominal_noise_std": 0.08,
        "records": records,
        "aggregate": {
            "record_count": len(records),
            "all_misfits_improved": bool(records) and all(value > 0.0 for value in reductions),
            "relative_misfit_reduction_min": min(reductions) if reductions else None,
            "relative_misfit_reduction_max": max(reductions) if reductions else None,
        },
        "claim_boundary": "Noise-scale sensitivity for one measured record; not a population-level field claim.",
    }
    out_path = parent / "sensitivity_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else root
        / f"field_laplace_{args.center_mode}"
        / f"noise_{float(args.noise_std):.3f}".replace(".", "p")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.snapshot.json").read_text(encoding="utf-8"))
    profile = acquisition_profile(protocol)
    if args.render_only:
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        arrays = np.load(out_dir / "la010010_full_laplace_arrays.npz")
        figure_path = out_dir / "fig_field_la010010_deepwave"
        save_field_figure(
            figure_path,
            observation=arrays["observation"].squeeze(),
            nn_prediction=arrays["nn_prediction"].squeeze(),
            posterior_mean=arrays["posterior_mean"].squeeze(),
            posterior_std=arrays["posterior_std"].squeeze(),
            event_probability=arrays["event_probability"].squeeze(),
            profile=profile,
            crop=summary["crop"],
        )
        print(json.dumps({"status": "rendered", "figure": str(figure_path)}, indent=2))
        return
    artifacts = torch.load(root / "prepared" / "artifacts.pt", map_location="cpu", weights_only=False)
    if artifacts["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Prepared artifacts do not match the protocol snapshot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_inversion_checkpoint(root / "train" / "unet" / f"seed_{args.seed}" / "best.pt", device)
    forward, forward_summary = load_surrogate_checkpoint(root / "surrogate" / "best.pt", device)
    field_root = resolve_data_path(
        resolve_protocol_path(Path(protocol["protocol_path"]), protocol["field_root"]),
        "Field_GPR_DATA.lnk",
        "Filed_GPR_Data.lnk",
    )
    rad_path = field_root / Path(profile["field_reference"]["relative_rad_path"])
    raw, header = read_rd3_pair(rad_path)
    bscan, crop = align_ramac_observation(raw, header, profile)
    observation = torch.from_numpy(bscan)[None, None].float()
    with torch.no_grad():
        nn_prediction = model(observation.to(device)).cpu()

    latent_dim = 16
    residual_scale = 0.003
    basis_path = root / "basis" / "error_pca_mean_latent16_train96_snrs0-5-10_seed7.pt"
    basis_payload = build_error_pca_basis(
        protocol=protocol,
        artifacts=artifacts,
        model=model,
        latent_dim=latent_dim,
        include_mean=True,
        train_count=96,
        snrs=[0.0, 5.0, 10.0],
        seed=int(args.seed),
        cache_path=basis_path,
        device=device,
    )
    basis = basis_payload["basis"]
    latent_prior_mean = (
        torch.zeros(latent_dim, device=device)
        if args.center_mode == "nn"
        else basis_payload["latent_raw_mean"] / residual_scale
    )
    latent_prior_std = basis_payload["latent_raw_std"] / residual_scale
    field_seed = int(protocol["noise"]["evaluation_seed"]) + int(args.seed) * 101 + 10010
    set_seed(field_seed)
    started = time.time()
    result, samples, _ = run_map_bvi_case(
        forward,
        observation,
        None,
        None,
        nn_prediction,
        latent_dim=latent_dim,
        residual_scale=residual_scale,
        model_prior_std=0.005,
        latent_prior_weight=0.5,
        steps=8,
        lr=0.03,
        basis_type="error_pca_mean",
        posterior_latent_std=0.03,
        posterior_sample_count=int(args.posterior_samples),
        event_threshold=0.5,
        device=device,
        noise_std_override=float(args.noise_std),
        basis_override=basis,
        latent_prior_mean=latent_prior_mean,
        latent_prior_std=latent_prior_std,
        latent_init="zero" if args.center_mode == "nn" else "prior_mean",
        model_prior_center="nn" if args.center_mode == "nn" else "latent_prior_mean",
        posterior_sampler="laplace_full",
    )
    posterior_mean = samples.mean(dim=0, keepdim=True)
    posterior_std = samples.std(dim=0, keepdim=True, unbiased=False)
    event_probability = (samples >= 0.5).float().mean(dim=0, keepdim=True)
    correction = posterior_mean - nn_prediction
    arrays_path = out_dir / "la010010_full_laplace_arrays.npz"
    np.savez_compressed(
        arrays_path,
        observation=observation.numpy(),
        nn_prediction=nn_prediction.numpy(),
        posterior_samples=samples.numpy(),
        posterior_mean=posterior_mean.numpy(),
        posterior_std=posterior_std.numpy(),
        event_probability=event_probability.numpy(),
        absolute_correction=correction.abs().numpy(),
    )
    figure_path = out_dir / "fig_field_la010010_deepwave"
    save_field_figure(
        figure_path,
        observation=observation.numpy().squeeze(),
        nn_prediction=nn_prediction.numpy().squeeze(),
        posterior_mean=posterior_mean.numpy().squeeze(),
        posterior_std=posterior_std.numpy().squeeze(),
        event_probability=event_probability.numpy().squeeze(),
        profile=profile,
        crop=crop,
    )
    summary = {
        "status": "complete",
        "source": str(rad_path),
        "crop": crop,
        "artificial_noise_added": False,
        "device": str(device),
        "forward_backend": forward_summary,
        "config": {
            "method": "trust_region_map_full_latent_laplace",
            "inversion_checkpoint": str(root / "train" / "unet" / f"seed_{args.seed}" / "best.pt"),
            "latent_dim": latent_dim,
            "basis": "error_pca_mean",
            "center_mode": args.center_mode,
            "residual_scale": residual_scale,
            "model_prior_std": 0.005,
            "latent_prior_weight": 0.5,
            "steps": 8,
            "learning_rate": 0.03,
            "noise_std": float(args.noise_std),
            "posterior_samples": int(args.posterior_samples),
            "event_threshold_normalized": 0.5,
            "epsilon_mapping": "epsilon_r = 2 + 8m",
            "physics_evaluations_approx": 42,
            "seed": field_seed,
        },
        "result": asdict(result),
        "diagnostics": {
            "data_misfit_reduction": float(result.data_misfit_nn - result.data_misfit_bvi_mean),
            "data_misfit_relative_reduction": float(
                (result.data_misfit_nn - result.data_misfit_bvi_mean) / result.data_misfit_nn
            ),
            "correction_rms_normalized": float(torch.sqrt(torch.mean(correction.square()))),
            "posterior_std_mean_normalized": float(posterior_std.mean()),
            "posterior_std_p95_normalized": float(torch.quantile(posterior_std, 0.95)),
            "event_probability_mean": float(event_probability.mean()),
            "event_probability_max": float(event_probability.max()),
            "event_area_at_p50": float((event_probability >= 0.5).float().mean()),
        },
        "seconds": time.time() - started,
        "arrays": str(arrays_path),
        "figure_pdf": str(figure_path.with_suffix(".pdf")),
        "figure_svg": str(figure_path.with_suffix(".svg")),
        "figure_png": str(figure_path.with_suffix(".png")),
        "claim_boundary": (
            "Measured-data transfer with no pixelwise permittivity truth. "
            "Only exact-Deepwave data consistency, residual-subspace dispersion, "
            "correction magnitude, and threshold-event interrogation are reported."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["sensitivity_summary"] = str(write_sensitivity_summary(out_dir.parent))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
