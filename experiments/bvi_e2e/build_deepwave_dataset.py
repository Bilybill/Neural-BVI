"""Build an explicit 256x256 permittivity -> Deepwave B-scan dataset.

The acquisition contract is read from ``acquisition_profiles.json``. The
default LJYSH/YGJ profile reproduces the reported 600-MHz, 512-sample/30-ns,
1-cm tunnel acquisition and its air/lining layers.

Deepwave's scalar propagator is used as the repository's current GPR wave
approximation. It is not a Maxwell TM solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.ndimage
import scipy.signal
import torch

from path_utils import resolve_data_path


C0 = 299_792_458.0
UNIT_SCALE = 1.0e6


def numeric_model_key(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def blackman_harris_derivative(frequency: float, time: np.ndarray) -> np.ndarray:
    coefficients = [0.35322222, -0.488, 0.145, -0.010222222]
    duration = 1.14 / frequency
    window = np.zeros_like(time)
    active = time < duration
    for order, coefficient in enumerate(coefficients):
        window[active] += coefficient * np.cos(2.0 * order * np.pi * time[active] / duration)
    pulse = np.concatenate((window[1:], np.zeros(1))) - window
    return (pulse / max(float(np.max(np.abs(pulse))), 1.0e-12)).astype(np.float32)


def load_profile(path: Path, name: str, source_frequency_hz: float | None) -> dict[str, Any]:
    profiles = json.loads(path.read_text(encoding="utf-8"))
    if name not in profiles:
        raise KeyError(f"Unknown acquisition profile {name!r}; choices={sorted(profiles)}")
    profile = profiles[name]
    if not profile.get("ready", False):
        raise ValueError(f"Profile {name!r} is not simulation-ready; missing={profile.get('missing', [])}")
    frequency = source_frequency_hz or profile["source"].get("center_frequency_hz")
    if frequency is None:
        raise ValueError(
            "Antenna centre frequency is absent from the measured metadata. "
            "Pass --source-frequency-hz after confirming the acquisition system."
        )
    profile = json.loads(json.dumps(profile))
    profile["source"]["center_frequency_hz"] = float(frequency)
    if source_frequency_hz is not None:
        profile["source"]["frequency_status"] = "explicit_user_value"
    return profile


def load_model(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)
    key = "model" if "model" in mat else "ep" if "ep" in mat else None
    if key is None:
        raise KeyError(f"{path} contains neither 'model' nor 'ep'")
    permittivity = np.asarray(mat[key], dtype=np.float32)
    height, width = permittivity.shape
    side = min(height, width)
    y0, x0 = (height - side) // 2, (width - side) // 2
    permittivity = permittivity[y0 : y0 + side, x0 : x0 + side]
    if permittivity.shape != (size, size):
        zoom = (size / permittivity.shape[0], size / permittivity.shape[1])
        permittivity = scipy.ndimage.zoom(permittivity, zoom, order=1)
    permittivity = np.clip(permittivity, 2.0, 10.0).astype(np.float32)
    normalized = ((permittivity - 2.0) / 8.0).astype(np.float32)
    return permittivity, normalized


def parse_rad(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
    return metadata


def read_rd3_pair(rad_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    metadata = parse_rad(rad_path)
    rd3_path = rad_path.with_suffix(".RD3")
    if not rd3_path.exists():
        rd3_path = rad_path.with_suffix(".rd3")
    if not rd3_path.exists():
        raise FileNotFoundError(f"Missing RD3 pair for {rad_path}")
    n_samples = int(metadata["samples"])
    raw = np.fromfile(rd3_path, dtype="<i2").astype(np.float32)
    n_traces = int(metadata.get("last_trace", 0)) or raw.size // n_samples
    if raw.size != n_samples * n_traces:
        raise ValueError(
            f"{rd3_path} contains {raw.size} int16 values, expected {n_samples * n_traces}"
        )
    return raw.reshape(n_traces, n_samples).T, {
        "rad_path": str(rad_path),
        "rd3_path": str(rd3_path),
        "n_samples": n_samples,
        "n_traces": n_traces,
        "time_window_ns": float(metadata.get("timewindow", 0.0)),
        "sample_interval_s": float(metadata.get("timewindow", 0.0)) * 1.0e-9 / n_samples,
        "trace_spacing_m": float(metadata.get("distance_interval", 0.0)),
    }


def crop_native(data: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    sample_start = int(cfg.get("sample_start", 0))
    sample_end = int(cfg["sample_end"])
    trace_start = int(cfg.get("trace_start", 0))
    trace_end = int(cfg["trace_end"])
    if sample_end > data.shape[0] or trace_end > data.shape[1]:
        raise ValueError(
            f"Configured crop {(sample_start, sample_end, trace_start, trace_end)} exceeds {data.shape}"
        )
    return np.asarray(data[sample_start:sample_end, trace_start:trace_end], dtype=np.float32)


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
    bandpass = cfg.get("bandpass_trapezoid_fc")
    if bandpass:
        dt = float(profile["observation"]["sample_interval_s"])
        fc = float(profile["source"]["center_frequency_hz"])
        f1, f2, f3, f4 = [float(value) * fc for value in bandpass]
        frequencies = np.fft.rfftfreq(result.shape[0], d=dt)
        response = np.zeros_like(frequencies, dtype=np.float64)
        rising = (frequencies >= f1) & (frequencies < f2)
        passband = (frequencies >= f2) & (frequencies <= f3)
        falling = (frequencies > f3) & (frequencies <= f4)
        response[rising] = (frequencies[rising] - f1) / max(f2 - f1, 1.0)
        response[passband] = 1.0
        response[falling] = (f4 - frequencies[falling]) / max(f4 - f3, 1.0)
        result = np.fft.irfft(np.fft.rfft(result, axis=0) * response[:, None], n=result.shape[0], axis=0).astype(np.float32)
    if cfg.get("time_zero_correction", False):
        search = min(int(cfg.get("time_zero_search_samples", 96)), result.shape[0])
        corrected = np.zeros_like(result)
        positive_peak = np.argmax(result[:search], axis=0)
        for trace_index, shift in enumerate(positive_peak.tolist()):
            available = result.shape[0] - shift
            corrected[:available, trace_index] = result[shift:, trace_index]
        result = corrected
    percentile = float(cfg.get("clip_percentile", 99.0))
    scale = float(np.percentile(np.abs(result), percentile))
    result = np.clip(result / max(scale, 1.0e-8), -1.0, 1.0)
    return result.astype(np.float32)


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float32) ** 2)) + 1.0e-12)


def standardize_noise(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = data - float(data.mean())
    return (data / rms(data)).astype(np.float32)


def extract_noise_residual(data: np.ndarray, noise_cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    import bm3d
    from skimage.restoration import denoise_nl_means, estimate_sigma

    scale = float(np.percentile(np.abs(data), 99.5) + 1.0e-8)
    normalized = np.clip(data / scale, -1.0, 1.0)
    normalized01 = (normalized + 1.0) * 0.5
    sigma = float(np.mean(estimate_sigma(normalized01, channel_axis=None)))
    nlm = denoise_nl_means(
        normalized01,
        h=float(noise_cfg.get("nlm_h_factor", 0.85)) * sigma,
        sigma=sigma,
        patch_size=int(noise_cfg.get("nlm_patch_size", 5)),
        patch_distance=int(noise_cfg.get("nlm_patch_distance", 7)),
        fast_mode=True,
        preserve_range=True,
        channel_axis=None,
    ).astype(np.float32)
    bm3d_result = bm3d.bm3d(
        normalized01,
        sigma_psd=float(noise_cfg.get("bm3d_sigma_factor", 0.85)) * sigma,
        profile="np",
    ).astype(np.float32)
    denoised = np.clip(0.5 * (nlm + bm3d_result), 0.0, 1.0)
    denoised_amplitude = ((denoised * 2.0) - 1.0) * scale
    residual = np.asarray(data, dtype=np.float32) - denoised_amplitude.astype(np.float32)
    return standardize_noise(residual), {
        "sigma_estimate_normalized": sigma,
        "residual_rms_before_standardization": rms(residual),
    }


def build_noise_bank(
    profile: dict[str, Any],
    field_root: Path | None,
) -> tuple[list[tuple[str, np.ndarray]], list[dict[str, Any]]]:
    noise_cfg = profile.get("noise", {})
    if not noise_cfg.get("enabled", False):
        return [], []
    if field_root is None:
        raise ValueError("The selected profile requires --field-root for its measured residual bank")
    target_shape = (
        int(profile["observation"]["n_time"]),
        int(profile["observation"]["n_traces"]),
    )
    residuals: list[tuple[str, np.ndarray]] = []
    reports: list[dict[str, Any]] = []
    for source in noise_cfg.get("sources", []):
        rad_path = field_root / Path(source["relative_rad_path"])
        raw, header = read_rd3_pair(rad_path)
        cropped = crop_native(raw, source)
        if cropped.shape != target_shape:
            raise ValueError(f"Noise crop {source['key']} has shape {cropped.shape}, expected {target_shape}")
        processed = preprocess_bscan(cropped, profile)
        residual, stats = extract_noise_residual(processed, noise_cfg)
        residuals.append((str(source["key"]), residual))
        reports.append({"key": source["key"], "header": header, "crop": source, **stats})
    if not residuals:
        raise ValueError("Noise is enabled but the residual bank is empty")
    return residuals, reports


def add_mixed_noise(
    clean: np.ndarray,
    noise_bank: list[tuple[str, np.ndarray]],
    noise_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_key, field_residual = noise_bank[int(rng.integers(0, len(noise_bank)))]
    field_residual = np.roll(field_residual, int(rng.integers(0, field_residual.shape[1])), axis=1)
    sign = -1.0 if bool(rng.integers(0, 2)) else 1.0
    gaussian = standardize_noise(rng.normal(size=clean.shape).astype(np.float32))
    field_weight = float(noise_cfg.get("field_weight", 0.7))
    gaussian_weight = float(noise_cfg.get("gaussian_weight", 0.3))
    mixed = standardize_noise(sign * field_weight * field_residual + gaussian_weight * gaussian)
    snr_low, snr_high = map(float, noise_cfg.get("snr_db_range", [0.0, 10.0]))
    snr_db = float(rng.uniform(snr_low, snr_high))
    scaled = mixed * (rms(clean) / (rms(mixed) * 10.0 ** (snr_db / 20.0)))
    return (clean + scaled).astype(np.float32), scaled.astype(np.float32), {
        "source_key": source_key,
        "field_sign": sign,
        "field_weight": field_weight,
        "gaussian_weight": gaussian_weight,
        "snr_db": snr_db,
    }


def simulate_one(
    permittivity: np.ndarray,
    profile: dict[str, Any],
    device: torch.device,
    shot_batch_size: int,
    pml_width: int,
) -> np.ndarray:
    from deepwave import scalar

    model_cfg = profile["model_domain"]
    obs = profile["observation"]
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
        fixed_arrays.append(
            np.full((layer_cells, nx), float(layer["relative_permittivity"]), dtype=np.float32)
        )
    simulation_permittivity = np.concatenate([*fixed_arrays, permittivity], axis=0) if fixed_arrays else permittivity
    source_z_index = int(round(antenna_depth / dz))
    source_x_index = np.rint(source_x / dx).astype(np.int64)
    receiver_x_index = np.rint(receiver_x / dx).astype(np.int64)

    velocity = torch.as_tensor(C0 / np.sqrt(simulation_permittivity) / UNIT_SCALE, dtype=torch.float32, device=device)
    dt_scaled = dt_real * UNIT_SCALE
    frequency_scaled = frequency_real / UNIT_SCALE
    time_scaled = np.arange(n_time, dtype=np.float64) * dt_scaled
    wavelet = torch.as_tensor(
        blackman_harris_derivative(frequency_scaled, time_scaled), dtype=torch.float32, device=device
    )
    records: list[torch.Tensor] = []
    for start in range(0, n_traces, shot_batch_size):
        stop = min(start + shot_batch_size, n_traces)
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
            pml_width=pml_width,
            pml_freq=frequency_scaled,
            accuracy=4,
        )[-1]
        records.append(receiver[:, 0].detach().cpu())
    bscan = torch.cat(records, dim=0).numpy().T
    return preprocess_bscan(bscan, profile)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_preview(path: Path, model: torch.Tensor, bscan: torch.Tensor, profile: dict[str, Any]) -> None:
    obs = profile["observation"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(model.squeeze().numpy() * 8.0 + 2.0, cmap="viridis", aspect="auto")
    axes[0].set_title("Permittivity (256 x 256)")
    axes[1].imshow(
        bscan.squeeze().numpy(),
        cmap="seismic",
        aspect="auto",
        extent=[
            0.0,
            (int(obs["n_traces"]) - 1) * float(obs["trace_spacing_m"]),
            float(obs["time_window_s"]) * 1e9,
            0.0,
        ],
    )
    axes[1].set_title("Deepwave training B-scan")
    axes[1].set_xlabel("Profile distance (m)")
    axes[1].set_ylabel("Time (ns)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("data/models"))
    parser.add_argument(
        "--field-root",
        type=Path,
        default=Path("data/field"),
        help="Measured-data root used only to construct a leakage-safe residual noise bank.",
    )
    parser.add_argument("--profiles", type=Path, default=here / "acquisition_profiles.json")
    parser.add_argument("--profile", default="ljysh_ygj_600mhz_30ns")
    parser.add_argument("--source-frequency-hz", type=float, default=None)
    parser.add_argument("--snr-db-range", type=float, nargs=2, default=None)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--disable-noise", action="store_true")
    parser.add_argument("--model-count", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--shot-batch-size", type=int, default=32)
    parser.add_argument("--pml-width", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/bvi_e2e/datasets/deepwave_ljysh_ygj_256"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir = resolve_data_path(args.model_dir)
    profile = load_profile(args.profiles, args.profile, args.source_frequency_hz)
    if args.disable_noise:
        profile.setdefault("noise", {})["enabled"] = False
    if args.snr_db_range is not None:
        profile.setdefault("noise", {})["snr_db_range"] = [float(v) for v in args.snr_db_range]
    if args.noise_seed is not None:
        profile.setdefault("noise", {})["random_seed"] = int(args.noise_seed)
    field_root = None
    if profile.get("noise", {}).get("enabled", False):
        args.field_root = resolve_data_path(args.field_root)
        field_root = args.field_root
    model_size = int(profile["model_shape"][0])
    if profile["model_shape"] != [256, 256]:
        raise ValueError("This experiment contract requires 256 x 256 models")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    paths = sorted(args.model_dir.glob("model_*.mat"), key=numeric_model_key)[: args.model_count]
    if not paths:
        raise FileNotFoundError(f"No model_*.mat files found in {args.model_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")

    noise_cfg = profile.get("noise", {})
    noise_rng = np.random.default_rng(int(noise_cfg.get("random_seed", 0)))
    noise_bank, noise_bank_report = build_noise_bank(profile, field_root)
    shards: list[dict[str, Any]] = []
    preview_payload: tuple[torch.Tensor, torch.Tensor] | None = None
    for shard_start in range(0, len(paths), args.shard_size):
        shard_paths = paths[shard_start : shard_start + args.shard_size]
        models: list[torch.Tensor] = []
        bscans: list[torch.Tensor] = []
        clean_bscans: list[torch.Tensor] = []
        noise_arrays: list[torch.Tensor] = []
        noise_metadata: list[dict[str, Any]] = []
        for index, model_path in enumerate(shard_paths, start=shard_start + 1):
            print(f"[{index}/{len(paths)}] Deepwave forward: {model_path.name}", flush=True)
            permittivity, normalized = load_model(model_path, model_size)
            clean_bscan = simulate_one(permittivity, profile, device, args.shot_batch_size, args.pml_width)
            if noise_bank:
                bscan, noise_array, noise_meta = add_mixed_noise(clean_bscan, noise_bank, noise_cfg, noise_rng)
            else:
                bscan = clean_bscan
                noise_array = np.zeros_like(clean_bscan)
                noise_meta = {"source_key": None, "snr_db": None}
            models.append(torch.from_numpy(normalized)[None])
            bscans.append(torch.from_numpy(bscan)[None])
            clean_bscans.append(torch.from_numpy(clean_bscan)[None])
            noise_arrays.append(torch.from_numpy(noise_array)[None])
            noise_metadata.append(noise_meta)
        model_tensor = torch.stack(models)
        bscan_tensor = torch.stack(bscans)
        clean_bscan_tensor = torch.stack(clean_bscans)
        noise_tensor = torch.stack(noise_arrays)
        shard_name = f"shard_{shard_start // args.shard_size:04d}.pt"
        shard_path = args.out_dir / shard_name
        torch.save(
            {
                "models": model_tensor,
                "bscans": bscan_tensor,
                "clean_bscans": clean_bscan_tensor,
                "noise": noise_tensor,
                "noise_metadata": noise_metadata,
                "model_names": [path.name for path in shard_paths],
            },
            shard_path,
        )
        shards.append(
            {
                "file": shard_name,
                "count": len(shard_paths),
                "model_names": [path.name for path in shard_paths],
                "sha256": sha256(shard_path),
            }
        )
        if preview_payload is None:
            preview_payload = model_tensor[0], bscan_tensor[0]

    manifest = {
        "format": "deepwave-gpr-dataset-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "physics": "Deepwave scalar wave approximation; not Maxwell TM",
        "profile_name": args.profile,
        "acquisition": profile,
        "model_normalization": {"epsilon_min": 2.0, "epsilon_max": 10.0, "formula": "(epsilon_r - 2) / 8"},
        "sample_count": len(paths),
        "model_shape": [1, model_size, model_size],
        "bscan_shape": [1, int(profile["observation"]["n_time"]), int(profile["observation"]["n_traces"])],
        "observation_channel": "bscans (field-residual/Gaussian noisy when acquisition.noise.enabled; otherwise clean)",
        "clean_channel": "clean_bscans (used to train the differentiable forward surrogate)",
        "noise_bank": noise_bank_report,
        "target_leakage_guard": (
            "LA010010 is the held-out field target and is not a residual-noise source"
            if args.profile == "la010010_pipe_native"
            else None
        ),
        "source_model_dir": str(args.model_dir),
        "field_root": str(field_root) if field_root is not None else None,
        "device": str(device),
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if preview_payload is not None:
        save_preview(args.out_dir / "preview.png", *preview_payload, profile)
    print(f"Saved {len(paths)} Deepwave samples to {manifest_path}")


if __name__ == "__main__":
    main()
