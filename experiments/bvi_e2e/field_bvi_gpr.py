"""Field-data Neural-BVI inference for ZRY GPR files.

The script reads ZRY files using the header offsets in data_read_zry.m,
preprocesses the radargram, applies the synthetic-trained inversion network,
and performs BVI refinement without using ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent))
from e2e_bvi_gpr import HyperbolaForward, TinyInversionNet, run_bvi, set_seed
from path_utils import resolve_data_path


def read_zry(path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    with path.open("rb") as f:
        f.seek(3218)
        n_samples = struct.unpack("<h", f.read(2))[0]
        f.seek(3214)
        dt_ps = struct.unpack("<f", f.read(4))[0]
        f.seek(3222)
        eps_header = struct.unpack("<f", f.read(4))[0]
        f.seek(3230)
        dx_header = struct.unpack("<f", f.read(4))[0]
        f.seek(0, 2)
        file_size = f.tell()
        n_traces = int((file_size - 3320) // (n_samples * 2 + 80))

        data = np.zeros((n_samples, n_traces), dtype=np.float32)
        for j in range(n_traces):
            f.seek(3320 + j * (80 + n_samples * 2) + 80)
            raw = f.read(n_samples * 2)
            if len(raw) != n_samples * 2:
                data = data[:, :j]
                break
            data[:, j] = np.frombuffer(raw, dtype="<i2").astype(np.float32)

    header = {
        "n_samples": n_samples,
        "n_traces": int(data.shape[1]),
        "dt_ps": float(dt_ps),
        "eps_header": float(eps_header),
        "dx_header": float(dx_header),
        "file_size": int(file_size),
    }
    return data, header


def preprocess_bscan(
    raw: np.ndarray,
    n_time: int,
    n_traces: int,
    crop_samples: int,
    crop_traces: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    data = raw.astype(np.float32)
    data = data - scipy.ndimage.uniform_filter1d(data, size=81, axis=0, mode="nearest")
    data = data - np.mean(data, axis=1, keepdims=True)
    gain = np.sqrt(np.arange(data.shape[0], dtype=np.float32) + 1.0)[:, None]
    data = data * gain

    start_t = 0
    end_t = min(crop_samples, data.shape[0])
    focus = data[start_t:end_t]

    if focus.shape[1] > crop_traces:
        energy = np.mean(np.abs(focus), axis=0)
        smooth_energy = scipy.ndimage.uniform_filter1d(energy, size=min(101, max(5, focus.shape[1] // 10)))
        center = int(np.argmax(smooth_energy))
        start_x = max(0, min(center - crop_traces // 2, focus.shape[1] - crop_traces))
        end_x = start_x + crop_traces
    else:
        start_x = 0
        end_x = focus.shape[1]

    crop = focus[:, start_x:end_x]
    clip = np.percentile(np.abs(crop), 99.0)
    if clip <= 0:
        clip = 1.0
    crop = np.clip(crop / clip, -1.0, 1.0)
    resized = scipy.ndimage.zoom(crop, (n_time / crop.shape[0], n_traces / crop.shape[1]), order=1)
    resized = resized.astype(np.float32)
    meta = {
        "time_start": int(start_t),
        "time_end": int(end_t),
        "trace_start": int(start_x),
        "trace_end": int(end_x),
        "clip_abs_p99": float(clip),
    }
    return resized, meta


def save_field_figure(
    out_path: Path,
    name: str,
    raw_view: np.ndarray,
    obs: torch.Tensor,
    nn_pred: torch.Tensor,
    bvi_mean: torch.Tensor,
    bvi_std: torch.Tensor,
    event_prob: torch.Tensor,
    interrogation_threshold: float,
) -> None:
    raw_np = raw_view
    obs_np = obs.squeeze().cpu().numpy()
    nn_np = nn_pred.squeeze().cpu().numpy()
    mean_np = bvi_mean.squeeze().cpu().numpy()
    std_np = bvi_std.squeeze().cpu().numpy()
    prob_np = event_prob.squeeze().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    panels = [
        (raw_np, "Preprocessed field radargram", "seismic"),
        (obs_np, "Network input crop", "seismic"),
        (nn_np, "E2E CNN permittivity", "viridis"),
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
    fig.suptitle(name, fontsize=10)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def field_label(path: Path, index: int, header: Dict[str, float] | None = None) -> str:
    stem = path.stem.lower()
    if "pillar" in stem or "柱子" in path.stem:
        return f"field_{index:02d}_pillar"
    if "grass" in stem or "草地" in path.stem:
        return f"field_{index:02d}_grass"
    if header is not None:
        n_traces = int(header.get("n_traces", 0))
        if n_traces > 1000:
            return f"field_{index:02d}_pillar"
        if n_traces > 0:
            return f"field_{index:02d}_grass"
    safe_new = "".join(ch if ch.isalnum() else "_" for ch in path.stem.encode("ascii", "ignore").decode("ascii"))
    safe_new = safe_new.strip("_") or "unknown"
    return f"field_{index:02d}_{safe_new}"


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-dir", type=Path, default=Path("data/field"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/bvi_e2e/results_grsl_trust/tiny_inversion_net.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bvi_e2e/field_results"))
    parser.add_argument("--image-size", type=int, default=48)
    parser.add_argument("--n-time", type=int, default=64)
    parser.add_argument("--n-traces", type=int, default=48)
    parser.add_argument("--crop-samples", type=int, default=768)
    parser.add_argument("--crop-traces", type=int, default=320)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--bvi-steps", type=int, default=160)
    parser.add_argument("--samples-per-component", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--model-prior-std", type=float, default=0.0)
    parser.add_argument("--interrogation-threshold", type=float, default=0.35)
    parser.add_argument("--basis-type", choices=["cosine", "mixed"], default="cosine")
    parser.add_argument("--noise-std", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.field_dir = resolve_data_path(args.field_dir, "Filed_GPR_Data.lnk")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forward = HyperbolaForward(args.image_size, args.n_traces, args.n_time).to(device)
    net = TinyInversionNet(args.image_size).to(device)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint {args.checkpoint}. Run e2e_bvi_gpr.py first.")
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net.eval()

    summaries = []
    for index, zry_path in enumerate(sorted(args.field_dir.glob("*.zry")), start=1):
        raw, header = read_zry(zry_path)
        bscan, crop_meta = preprocess_bscan(raw, args.n_time, args.n_traces, args.crop_samples, args.crop_traces)
        obs = torch.tensor(bscan[None, None], dtype=torch.float32)
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

        safe_stem = field_label(zry_path, index, header)
        npz_path = args.out_dir / f"{safe_stem}_field_arrays.npz"
        np.savez_compressed(
            npz_path,
            raw_header=np.array(json.dumps(header, ensure_ascii=False)),
            crop_meta=np.array(json.dumps(crop_meta, ensure_ascii=False)),
            bscan=bscan,
            nn_prediction=nn_pred.numpy(),
            bvi_mean=bvi_mean.numpy(),
            bvi_std=bvi_std.numpy(),
            high_eps_probability=event_prob.numpy(),
        )
        fig_path = args.out_dir / f"{safe_stem}_field_result.png"
        save_field_figure(
            fig_path,
            safe_stem,
            bscan,
            obs,
            nn_pred,
            bvi_mean,
            bvi_std,
            event_prob,
            args.interrogation_threshold,
        )
        rec = {
            "profile_label": safe_stem,
            "source_file_name": zry_path.name,
            "file": str(zry_path),
            "checkpoint": str(args.checkpoint),
            "header": header,
            "crop": crop_meta,
            "figure": str(fig_path),
            "arrays": str(npz_path),
            "bvi": json_safe(bvi_result.__dict__),
            "nn_prediction_mean": float(nn_pred.mean()),
            "nn_prediction_std": float(nn_pred.std()),
            "bvi_std_mean": float(bvi_std.mean()),
        }
        safe_rec = json_safe(rec)
        summaries.append(safe_rec)
        print(json.dumps(safe_rec, ensure_ascii=False, indent=2))

    summary = {
        "config": json_safe(vars(args) | {"device": str(device)}),
        "profiles": summaries,
    }
    (args.out_dir / "field_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"processed {len(summaries)} field files -> {args.out_dir}")


if __name__ == "__main__":
    main()
