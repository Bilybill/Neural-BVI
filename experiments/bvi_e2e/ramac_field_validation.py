"""Run Neural-BVI on a RAMAC-style RAD/RD3 field radargram.

The nested ``Filed_GPR_Data/Data Set`` tree includes pipe examples stored as
paired text RAD metadata and binary RD3 samples. This script parses one or more
pairs, applies the synthetic-trained inversion network, and runs the same
trust-region BVI refinement used for the ZRY and HMJ field checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from deepwave_dataset import DeepwaveForwardSurrogate, load_manifest
from build_deepwave_dataset import preprocess_bscan
from e2e_bvi_gpr import TinyInversionNet, json_safe, run_bvi, set_seed
from field_bvi_gpr import save_field_figure
from path_utils import resolve_data_path


DEFAULT_RAD = Path(
    "data/field/Data Set/pipe/0701/ASCII_707-01/ASCII/LA010010.RAD"
)


def has_rd3_pair(rad_path: Path) -> bool:
    return rad_path.with_suffix(".RD3").exists() or rad_path.with_suffix(".rd3").exists()


def discover_rad_files(root: Path, pattern: str, max_files: int | None) -> list[Path]:
    paths = sorted(path for path in root.rglob(pattern) if path.is_file() and has_rd3_pair(path))
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def parse_rad(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not value:
            metadata[key] = value
            continue
        try:
            number = float(value)
            metadata[key] = int(number) if number.is_integer() else number
        except ValueError:
            metadata[key] = value
    return metadata


def read_rd3_pair(rad_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = parse_rad(rad_path)
    rd3_path = rad_path.with_suffix(".RD3")
    if not rd3_path.exists():
        rd3_path = rad_path.with_suffix(".rd3")
    if not rd3_path.exists():
        raise FileNotFoundError(f"Missing RD3 pair for {rad_path}")

    n_samples = int(metadata["samples"])
    n_traces = int(metadata.get("last_trace", 0))
    raw = np.fromfile(rd3_path, dtype="<i2").astype(np.float32)
    if n_traces <= 0:
        n_traces = raw.size // n_samples
    expected = n_samples * n_traces
    if raw.size != expected:
        raise ValueError(f"{rd3_path} has {raw.size} int16 values, expected {expected}")

    radargram = raw.reshape(n_traces, n_samples).T
    header = {
        "rad_file": str(rad_path),
        "rd3_file": str(rd3_path),
        "n_samples": n_samples,
        "n_traces": n_traces,
        "timewindow_ns": float(metadata.get("timewindow", 0.0)),
        "distance_interval_m": float(metadata.get("distance_interval", 0.0)),
        "frequency_mhz": float(metadata.get("frequency", 0.0)),
        "stop_position_m": float(metadata.get("stop_position", 0.0)),
        "raw_int16_values": int(raw.size),
        "rd3_bytes": int(rd3_path.stat().st_size),
        "metadata": metadata,
    }
    return radargram, header


def safe_label(rad_path: Path, index: int) -> str:
    parts = [part for part in rad_path.parts if part.lower() in {"pipe", "rebar", "tunnel"}]
    category = parts[0].lower() if parts else "ramac"
    stem = "".join(ch if ch.isalnum() else "_" for ch in rad_path.stem).strip("_")
    return f"ramac_{index:02d}_{category}_{stem}"


def aggregate_profiles(profiles: list[dict], failures: list[dict]) -> dict[str, Any]:
    bvi_records = [profile["bvi"] for profile in profiles]

    def values(key: str) -> list[float]:
        return [float(rec[key]) for rec in bvi_records if rec.get(key) is not None]

    nn_misfits = values("data_misfit_nn")
    bvi_misfits = values("data_misfit_bvi_mean")
    paired = [(nn, bvi) for nn, bvi in zip(nn_misfits, bvi_misfits) if nn > 0.0]
    reductions = [(nn - bvi) / nn for nn, bvi in paired]
    mean_stds = values("mean_std")
    central_probs = values("central_high_eps_prob")
    high_area = values("high_eps_area_mean")

    def mean_or_none(items: list[float]) -> float | None:
        return float(np.mean(items)) if items else None

    def std_or_none(items: list[float]) -> float | None:
        return float(np.std(items)) if items else None

    return {
        "profile_count": len(profiles),
        "failure_count": len(failures),
        "improved_count": int(sum(bvi < nn for nn, bvi in paired)),
        "improvement_fraction": float(np.mean([bvi < nn for nn, bvi in paired])) if paired else None,
        "data_misfit_nn_mean": mean_or_none(nn_misfits),
        "data_misfit_nn_std": std_or_none(nn_misfits),
        "data_misfit_bvi_mean_mean": mean_or_none(bvi_misfits),
        "data_misfit_bvi_mean_std": std_or_none(bvi_misfits),
        "relative_misfit_reduction_mean": mean_or_none(reductions),
        "relative_misfit_reduction_std": std_or_none(reductions),
        "mean_std_mean": mean_or_none(mean_stds),
        "mean_std_std": std_or_none(mean_stds),
        "central_high_eps_prob_mean": mean_or_none(central_probs),
        "high_eps_area_mean": mean_or_none(high_area),
    }


def source_relpath(profile: dict[str, Any], root_hint: str | None) -> str:
    path = Path(profile["file"])
    if root_hint:
        try:
            return path.relative_to(Path(root_hint)).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def acquisition_group(relpath: str) -> str:
    parts = Path(relpath).parts
    if "pipe" in parts:
        idx = parts.index("pipe")
        return "/".join(parts[idx + 1 : -1]) or "pipe"
    return "/".join(parts[:-1]) or "unknown"


def write_batch_figure(summary: dict[str, Any], out_path: Path) -> None:
    profiles = summary["profiles"]
    labels = np.arange(1, len(profiles) + 1)
    nn = np.array([profile["bvi"]["data_misfit_nn"] for profile in profiles], dtype=float)
    bvi = np.array([profile["bvi"]["data_misfit_bvi_mean"] for profile in profiles], dtype=float)
    reduction = (nn - bvi) / np.maximum(nn, 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    axes[0].plot(labels, nn, color="#4c78a8", linewidth=1.2, label="Neural inverse")
    axes[0].plot(labels, bvi, color="#f58518", linewidth=1.2, label="Neural-BVI")
    axes[0].set_xlabel("RAD/RD3 profile index")
    axes[0].set_ylabel("Proxy forward misfit")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].hist(reduction, bins=12, color="#54a24b", edgecolor="white")
    axes[1].axvline(reduction.mean(), color="#222222", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Relative misfit reduction")
    axes[1].set_ylabel("Profile count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("RAMAC/RD3 batch screening", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_paper_summary(
    summary: dict[str, Any],
    out_json: Path | None,
    out_md: Path | None,
    out_figure: Path | None = None,
) -> None:
    aggregate = summary["aggregate"]
    root_hint = summary.get("config", {}).get("discover_root")
    paper_summary = {
        "source_summary": str(summary["summary_path"]),
        "aggregate": aggregate,
        "profiles": [
            {
                "profile_label": profile["profile_label"],
                "source_file_name": profile["source_file_name"],
                "source_relpath": source_relpath(profile, root_hint),
                "acquisition_group": acquisition_group(source_relpath(profile, root_hint)),
                "n_samples": profile["header"]["n_samples"],
                "n_traces": profile["header"]["n_traces"],
                "data_misfit_nn": profile["bvi"]["data_misfit_nn"],
                "data_misfit_bvi_mean": profile["bvi"]["data_misfit_bvi_mean"],
                "mean_std": profile["bvi"]["mean_std"],
                "central_high_eps_prob": profile["bvi"]["central_high_eps_prob"],
            }
            for profile in summary["profiles"]
        ],
        "failures": summary["failures"],
        "notes": summary["notes"],
    }
    if out_figure is not None:
        write_batch_figure(summary, out_figure)
        paper_summary["figure"] = str(out_figure)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(json_safe(paper_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# RAMAC Batch Screening",
            "",
            "This report summarizes a qualitative batch screening of RAD/RD3 pipe profiles. The records have no matched permittivity labels, so the evidence is limited to proxy forward-data consistency and posterior interrogation statistics.",
            "",
            "## Aggregate",
            "",
            f"- Profiles processed: `{aggregate['profile_count']}`",
            f"- Failed profiles: `{aggregate['failure_count']}`",
            f"- Profiles with lower BVI proxy misfit: `{aggregate['improved_count']}`",
            f"- Improvement fraction: `{aggregate['improvement_fraction']:.3f}`" if aggregate["improvement_fraction"] is not None else "- Improvement fraction: `n/a`",
            f"- Mean NN proxy misfit: `{aggregate['data_misfit_nn_mean']:.4f}`" if aggregate["data_misfit_nn_mean"] is not None else "- Mean NN proxy misfit: `n/a`",
            f"- Mean BVI proxy misfit: `{aggregate['data_misfit_bvi_mean_mean']:.4f}`" if aggregate["data_misfit_bvi_mean_mean"] is not None else "- Mean BVI proxy misfit: `n/a`",
            f"- Mean relative misfit reduction: `{aggregate['relative_misfit_reduction_mean']:.3f}`" if aggregate["relative_misfit_reduction_mean"] is not None else "- Mean relative misfit reduction: `n/a`",
            f"- Mean posterior std.: `{aggregate['mean_std_mean']:.4f}`" if aggregate["mean_std_mean"] is not None else "- Mean posterior std.: `n/a`",
            "",
            "## Profiles",
            "",
            "| Source path | Group | Samples x traces | NN misfit | BVI misfit | Mean std. |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for profile in paper_summary["profiles"]:
            lines.append(
                f"| `{profile['source_relpath']}` | `{profile['acquisition_group']}` | "
                f"{profile['n_samples']} x {profile['n_traces']} | "
                f"{profile['data_misfit_nn']:.4f} | {profile['data_misfit_bvi_mean']:.4f} | {profile['mean_std']:.4f} |"
            )
        if summary["failures"]:
            lines.extend(["", "## Failures", ""])
            for failure in summary["failures"]:
                lines.append(f"- `{failure['file']}`: {failure['error']}")
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def align_ramac_observation(
    raw: np.ndarray,
    header: dict[str, Any],
    acquisition: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Preprocess and resample a RAD/RD3 record on physical time/distance axes."""

    obs_cfg = acquisition["observation"]
    prep_cfg = acquisition["preprocessing"]
    target_time = int(obs_cfg["n_time"])
    target_traces = int(obs_cfg["n_traces"])
    target_dt = float(obs_cfg["sample_interval_s"])
    target_dx = float(obs_cfg["trace_spacing_m"])
    source_window_s = float(header["timewindow_ns"]) * 1.0e-9
    source_dx = float(header["distance_interval_m"])
    if source_window_s <= 0.0 or source_dx <= 0.0:
        raise ValueError(f"Invalid RAMAC physical axes: {header}")

    data = raw.astype(np.float32)
    reference = acquisition.get("field_reference", {})
    if reference.get("crop_mode") == "fixed_native":
        expected_name = Path(reference["relative_rad_path"]).name.lower()
        if Path(header["rad_file"]).name.lower() != expected_name:
            raise ValueError(
                f"Profile is locked to {expected_name}, received {Path(header['rad_file']).name}"
            )
        checks = {
            "raw samples": (int(header["n_samples"]), int(reference["expected_raw_samples"])),
            "raw traces": (int(header["n_traces"]), int(reference["expected_raw_traces"])),
        }
        for label, (actual, expected) in checks.items():
            if actual != expected:
                raise ValueError(f"LA010010 {label} mismatch: {actual} != {expected}")
        if not np.isclose(source_dx, float(reference["expected_trace_spacing_m"]), rtol=0.0, atol=1.0e-9):
            raise ValueError(f"LA010010 trace spacing mismatch: {source_dx}")
        raw_dt = source_window_s / int(header["n_samples"])
        if not np.isclose(raw_dt, target_dt, rtol=0.0, atol=1.0e-15):
            raise ValueError(f"LA010010 sample interval mismatch: {raw_dt} != {target_dt}")
        sample_start = int(reference["sample_start"])
        sample_end = int(reference["sample_end"])
        trace_start = int(reference["trace_start"])
        trace_end = int(reference["trace_end"])
        aligned = data[sample_start:sample_end, trace_start:trace_end]
        if aligned.shape != (target_time, target_traces):
            raise ValueError(f"LA010010 fixed crop has shape {aligned.shape}")
        aligned = preprocess_bscan(aligned, acquisition)
        return aligned, {
            "mode": "fixed_native_no_resampling",
            "sample_start": sample_start,
            "sample_end": sample_end,
            "trace_start": trace_start,
            "trace_end": trace_end,
            "start_distance_m": float(reference["start_distance_m"]),
            "target_time_window_ns": target_time * target_dt * 1.0e9,
            "target_aperture_m": (target_traces - 1) * target_dx,
        }

    physical_width = (target_traces - 1) * target_dx
    required_source_traces = int(np.ceil(physical_width / source_dx)) + 1
    if data.shape[1] < required_source_traces:
        raise ValueError(
            f"Profile has {data.shape[1]} traces but {required_source_traces} are needed "
            "for the acquisition-aligned physical window"
        )
    energy = np.mean(np.abs(data), axis=0)
    smooth = scipy.ndimage.uniform_filter1d(energy, size=min(101, max(5, data.shape[1] // 10)))
    center = int(np.argmax(smooth))
    start = max(0, min(center - required_source_traces // 2, data.shape[1] - required_source_traces))
    crop = data[:, start : start + required_source_traces]

    source_t = np.linspace(0.0, source_window_s, crop.shape[0], endpoint=False)
    source_x = np.arange(crop.shape[1], dtype=np.float64) * source_dx
    target_t = np.arange(target_time, dtype=np.float64) * target_dt
    target_x = np.arange(target_traces, dtype=np.float64) * target_dx
    tt, xx = np.meshgrid(target_t, target_x, indexing="ij")
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (source_t, source_x), crop, bounds_error=False, fill_value=0.0
    )
    aligned = interpolator(np.stack((tt.ravel(), xx.ravel()), axis=-1)).reshape(target_time, target_traces)
    # Apply exactly the same profile preprocessing used for synthetic records.
    aligned = preprocess_bscan(aligned, acquisition)
    return aligned, {
        "trace_start": start,
        "trace_end": start + required_source_traces,
        "source_time_window_ns": float(header["timewindow_ns"]),
        "target_time_window_ns": target_time * target_dt * 1.0e9,
        "source_trace_spacing_m": source_dx,
        "target_trace_spacing_m": target_dx,
        "preprocessing": prep_cfg,
    }


def run_case(
    args: argparse.Namespace,
    rad_path: Path,
    index: int,
    net: TinyInversionNet,
    forward: DeepwaveForwardSurrogate,
    acquisition: dict[str, Any],
) -> dict:
    raw, header = read_rd3_pair(rad_path)
    bscan, crop_meta = align_ramac_observation(raw, header, acquisition)
    obs = torch.tensor(bscan[None, None], dtype=torch.float32)
    device = next(net.parameters()).device

    with torch.no_grad():
        nn_pred = net(obs.to(device)).cpu()

    bvi_result, bvi_mean, bvi_std, event_prob = run_bvi(
        forward=forward,
        observation=obs,
        true_model=None,
        nn_mean=nn_pred,
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

    label = safe_label(rad_path, index)
    arrays_path = args.out_dir / f"{label}_arrays.npz"
    np.savez_compressed(
        arrays_path,
        raw_header=np.array(json.dumps(header, ensure_ascii=False)),
        crop_meta=np.array(json.dumps(crop_meta, ensure_ascii=False)),
        bscan=bscan,
        nn_prediction=nn_pred.numpy(),
        bvi_mean=bvi_mean.numpy(),
        bvi_std=bvi_std.numpy(),
        high_eps_probability=event_prob.numpy(),
    )
    fig_path = args.out_dir / f"{label}_result.png"
    save_field_figure(
        fig_path,
        label,
        bscan,
        obs,
        nn_pred,
        bvi_mean,
        bvi_std,
        event_prob,
        args.interrogation_threshold,
    )
    if index == 1 and args.paper_figure:
        args.paper_figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fig_path, args.paper_figure)

    return {
        "profile_label": label,
        "source_file_name": rad_path.name,
        "file": str(rad_path),
        "header": header,
        "crop": crop_meta,
        "figure": str(fig_path),
        "paper_figure": str(args.paper_figure) if index == 1 and args.paper_figure else None,
        "arrays": str(arrays_path),
        "bvi": json_safe(bvi_result.__dict__),
        "nn_prediction_mean": float(nn_pred.mean()),
        "nn_prediction_std": float(nn_pred.std()),
        "bvi_std_mean": float(bvi_std.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rad-files", type=Path, nargs="+", default=[DEFAULT_RAD])
    parser.add_argument("--discover-root", type=Path, default=None)
    parser.add_argument("--rad-pattern", default="*.RAD")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/bvi_e2e/results_grsl_trust/tiny_inversion_net.pt"))
    parser.add_argument(
        "--forward-checkpoint",
        type=Path,
        default=Path("experiments/bvi_e2e/results_grsl_trust/deepwave_forward_surrogate.pt"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("experiments/bvi_e2e/datasets/deepwave_ramac_256/manifest.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bvi_e2e/ramac_field_validation"))
    parser.add_argument("--paper-figure", type=Path, default=Path("paper_grsl/figures/fig_ramac_pipe_result.png"))
    parser.add_argument("--paper-summary-json", type=Path, default=None)
    parser.add_argument("--paper-summary-md", type=Path, default=None)
    parser.add_argument("--paper-summary-figure", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--bvi-steps", type=int, default=180)
    parser.add_argument("--samples-per-component", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.04)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--model-prior-std", type=float, default=0.03)
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--interrogation-threshold", type=float, default=0.35)
    parser.add_argument("--basis-type", choices=["cosine", "mixed"], default="cosine")
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.discover_root is not None:
        args.discover_root = resolve_data_path(args.discover_root, "Filed_GPR_Data.lnk")
    else:
        args.rad_files = [resolve_data_path(path) for path in args.rad_files]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.dataset_manifest.exists():
        raise FileNotFoundError(f"Missing Deepwave dataset manifest {args.dataset_manifest}")
    dataset_manifest = load_manifest(args.dataset_manifest)
    if dataset_manifest["profile_name"] not in {"ramac_200ns", "la010010_pipe_native"}:
        raise ValueError(
            "RAMAC validation requires ramac_200ns or la010010_pipe_native, "
            f"got {dataset_manifest['profile_name']}"
        )
    acquisition = dataset_manifest["acquisition"]
    args.n_time = int(acquisition["observation"]["n_time"])
    args.n_traces = int(acquisition["observation"]["n_traces"])
    forward = DeepwaveForwardSurrogate(args.n_time, args.n_traces).to(device)
    net = TinyInversionNet(args.image_size).to(device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {args.checkpoint}. Run e2e_bvi_gpr.py first.")
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    if not args.forward_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing Deepwave forward-surrogate checkpoint {args.forward_checkpoint}. Run e2e_bvi_gpr.py first."
        )
    forward.load_state_dict(torch.load(args.forward_checkpoint, map_location=device))
    net.eval()
    forward.eval()
    for parameter in forward.parameters():
        parameter.requires_grad_(False)

    if args.discover_root is not None:
        args.rad_files = discover_rad_files(args.discover_root, args.rad_pattern, args.max_files)
        if not args.rad_files:
            raise FileNotFoundError(f"No paired RAD/RD3 files found under {args.discover_root}")

    profiles = []
    failures = []
    for index, rad_path in enumerate(args.rad_files, start=1):
        try:
            rec = run_case(args, rad_path, index, net, forward, acquisition)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failures.append({"file": str(rad_path), "error": repr(exc)})
            if args.quiet:
                print(f"failed {index}/{len(args.rad_files)}: {rad_path.name}: {exc!r}")
            else:
                print(json.dumps(json_safe(failures[-1]), ensure_ascii=False, indent=2))
            continue
        profiles.append(json_safe(rec))
        if args.quiet:
            bvi = rec["bvi"]
            print(
                f"processed {index}/{len(args.rad_files)}: {rad_path.name} "
                f"misfit {bvi['data_misfit_nn']:.4f}->{bvi['data_misfit_bvi_mean']:.4f}"
            )
        else:
            print(json.dumps(json_safe(rec), ensure_ascii=False, indent=2))

    summary = {
        "config": json_safe(vars(args) | {"device": str(device)}),
        "acquisition": acquisition,
        "profiles": profiles,
        "failures": failures,
        "aggregate": aggregate_profiles(profiles, failures),
        "notes": (
            "RAD/RD3 field data provide an additional qualitative measured-data check. "
            "The physical time and distance axes are resampled to the same RAMAC acquisition "
            "contract used by the Deepwave synthetic dataset. No matched permittivity ground "
            "truth is available, so metrics remain forward-surrogate consistency and posterior "
            "interrogation statistics."
        ),
    }
    summary_path = args.out_dir / "ramac_field_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_paper_summary(summary, args.paper_summary_json, args.paper_summary_md, args.paper_summary_figure)
    print(f"processed {len(profiles)} RAMAC field files -> {summary_path}")


if __name__ == "__main__":
    main()
