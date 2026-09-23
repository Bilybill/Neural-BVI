"""Deterministic splits, dynamic noise, fixed views, and OOD models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from deepwave_dataset import load_deepwave_dataset
from path_utils import resolve_data_path


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float32) ** 2)) + 1.0e-12)


def standardize_noise(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = data - float(data.mean())
    return (data / rms(data)).astype(np.float32)


def protocol_hash(protocol: dict[str, Any]) -> str:
    payload = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    protocol["protocol_path"] = str(path.resolve())
    protocol["protocol_hash"] = protocol_hash({k: v for k, v in protocol.items() if not k.startswith("protocol_")})
    return protocol


def resolve_protocol_path(protocol_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[2]
    return (root / path).resolve()


def make_split(count: int, cfg: dict[str, Any]) -> dict[str, list[int]]:
    required = int(cfg["train"]) + int(cfg["validation"]) + int(cfg["test"])
    if count < required:
        raise ValueError(f"Dataset has {count} models, split requires {required}")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(int(cfg["seed"]))).tolist()[:required]
    a = int(cfg["train"])
    b = a + int(cfg["validation"])
    return {"train": order[:a], "validation": order[a:b], "test": order[b:]}


def model_features(models: torch.Tensor) -> np.ndarray:
    data = models[:, 0].numpy().astype(np.float64)
    dy = np.diff(data, axis=1)
    dx = np.diff(data, axis=2)
    return np.stack(
        [
            data.mean((1, 2)),
            data.std((1, 2)),
            (data >= 0.5).mean((1, 2)),
            np.mean(np.abs(dy), axis=(1, 2)) + np.mean(np.abs(dx), axis=(1, 2)),
        ],
        axis=1,
    )


def farthest_point_subset(models: torch.Tensor, count: int) -> list[int]:
    features = model_features(models)
    features = (features - features.mean(0)) / np.maximum(features.std(0), 1.0e-8)
    selected = [int(np.argmin(np.linalg.norm(features, axis=1)))]
    distances = np.linalg.norm(features - features[selected[0]], axis=1)
    while len(selected) < min(count, len(features)):
        candidate = int(np.argmax(distances))
        selected.append(candidate)
        distances = np.minimum(distances, np.linalg.norm(features - features[candidate], axis=1))
    return selected


def synthesize_noise_view(
    clean: torch.Tensor,
    residual_bank: torch.Tensor,
    seed: int,
    snr_db: float | None,
    mode: str,
    field_weight: float,
    gaussian_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "clean" or snr_db is None:
        return clean.clone(), {"mode": "clean", "snr_db": None, "source_index": None}
    rng = np.random.default_rng(int(seed))
    source_index = int(rng.integers(0, len(residual_bank)))
    residual = residual_bank[source_index, 0].numpy().copy()
    residual = np.roll(residual, int(rng.integers(0, residual.shape[1])), axis=1)
    sign = -1.0 if bool(rng.integers(0, 2)) else 1.0
    gaussian = standardize_noise(rng.normal(size=residual.shape).astype(np.float32))
    if mode == "gaussian":
        mixed = gaussian
        effective_field_weight, effective_gaussian_weight = 0.0, 1.0
    elif mode == "field":
        mixed = standardize_noise(sign * residual)
        effective_field_weight, effective_gaussian_weight = 1.0, 0.0
    elif mode == "mixed":
        mixed = standardize_noise(sign * field_weight * residual + gaussian_weight * gaussian)
        effective_field_weight, effective_gaussian_weight = field_weight, gaussian_weight
    else:
        raise ValueError(f"Unknown noise mode {mode!r}")
    clean_np = clean[0].numpy().astype(np.float32)
    scaled = mixed * (rms(clean_np) / (rms(mixed) * 10.0 ** (float(snr_db) / 20.0)))
    noisy = clean_np + scaled
    return torch.from_numpy(noisy[None]), {
        "mode": mode,
        "snr_db": float(snr_db),
        "source_index": source_index,
        "field_sign": sign,
        "field_weight": effective_field_weight,
        "gaussian_weight": effective_gaussian_weight,
    }


class DynamicNoiseDataset(Dataset):
    def __init__(
        self,
        clean: torch.Tensor,
        models: torch.Tensor,
        global_indices: list[int],
        residual_bank: torch.Tensor,
        base_seed: int,
        snr_range: tuple[float, float],
        mode: str,
        field_weight: float,
        gaussian_weight: float,
    ):
        self.clean = clean
        self.models = models
        self.global_indices = global_indices
        self.residual_bank = residual_bank
        self.base_seed = int(base_seed)
        self.snr_range = snr_range
        self.mode = mode
        self.field_weight = field_weight
        self.gaussian_weight = gaussian_weight
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.models)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seed = self.base_seed + self.epoch * 1_000_003 + int(self.global_indices[index]) * 10_007
        rng = np.random.default_rng(seed)
        snr = None if self.mode == "clean" else float(rng.uniform(*self.snr_range))
        view, _ = synthesize_noise_view(
            self.clean[index],
            self.residual_bank,
            seed,
            snr,
            self.mode,
            self.field_weight,
            self.gaussian_weight,
        )
        return view, self.models[index]


def build_fixed_views(
    clean: torch.Tensor,
    global_indices: list[int],
    residual_bank: torch.Tensor,
    noise_cfg: dict[str, Any],
    split_tag: int,
) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, Any]]]]:
    views = {"clean": clean.clone()}
    metadata: dict[str, list[dict[str, Any]]] = {
        "clean": [{"mode": "clean", "global_index": idx} for idx in global_indices]
    }
    for snr in noise_cfg["evaluation_snrs_db"]:
        label = f"mixed_{float(snr):g}db"
        records, metas = [], []
        for local, global_index in enumerate(global_indices):
            seed = int(noise_cfg["evaluation_seed"]) + split_tag * 10_000_000 + global_index * 1009 + int(snr * 10)
            view, meta = synthesize_noise_view(
                clean[local],
                residual_bank,
                seed,
                float(snr),
                "mixed",
                float(noise_cfg["field_weight"]),
                float(noise_cfg["gaussian_weight"]),
            )
            records.append(view)
            metas.append(meta | {"global_index": global_index, "seed": seed})
        views[label] = torch.stack(records)
        metadata[label] = metas
    return views, metadata


def prepare_protocol_artifacts(protocol: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    from build_deepwave_dataset import build_noise_bank, load_profile

    manifest_path = resolve_protocol_path(Path(protocol["protocol_path"]), protocol["dataset_manifest"])
    models, _, clean, names, manifest = load_deepwave_dataset(manifest_path, return_clean=True)
    if smoke:
        models, clean, names = models[:8], clean[:8], names[:8]
    split_cfg = dict(protocol["split"])
    if smoke:
        split_cfg.update({"train": 4, "validation": 2, "test": 2})
    splits = make_split(len(models), split_cfg)

    profiles_path = resolve_protocol_path(Path(protocol["protocol_path"]), protocol["acquisition_profile"])
    profile = load_profile(profiles_path, manifest["profile_name"], None)
    field_root = resolve_data_path(Path(protocol["field_root"]))
    residuals, residual_report = build_noise_bank(profile, field_root)
    residual_bank = torch.from_numpy(np.stack([residual for _, residual in residuals])[:, None])
    residual_keys = [key for key, _ in residuals]
    if any("la010010" in key.lower() for key in residual_keys):
        raise ValueError("Target leakage: LA010010 appears in residual keys")

    split_payload: dict[str, Any] = {}
    for tag, split_name in enumerate(("validation", "test"), start=1):
        indices = splits[split_name]
        split_clean = clean[indices]
        views, view_meta = build_fixed_views(
            split_clean, indices, residual_bank, protocol["noise"], split_tag=tag
        )
        subset_count = min(int(protocol["uq"][f"{split_name}_models"]), len(indices))
        uq_local = farthest_point_subset(models[indices], subset_count)
        split_payload[split_name] = {
            "indices": indices,
            "model_names": [names[index] for index in indices],
            "views": views,
            "view_metadata": view_meta,
            "uq_local_indices": uq_local,
            "uq_global_indices": [indices[index] for index in uq_local],
        }
    return {
        "format": "la010010-publication-artifacts-v1",
        "protocol_hash": protocol["protocol_hash"],
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_profile": manifest["profile_name"],
        "model_names": names,
        "models": models,
        "clean_bscans": clean,
        "splits": splits,
        "split_model_names": {key: [names[index] for index in value] for key, value in splits.items()},
        "residual_bank": residual_bank,
        "residual_keys": residual_keys,
        "residual_report": residual_report,
        "fixed": split_payload,
        "smoke": smoke,
    }


def generate_ood_models(cfg: dict[str, Any], size: int = 256) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    rng = np.random.default_rng(int(cfg["seed"]))
    per_class = int(cfg["per_class"])
    zz, xx = np.meshgrid(np.linspace(0.0, 2.5, size), np.linspace(0.0, 3.5328, size), indexing="ij")
    models, metadata = [], []
    for class_name in cfg["classes"]:
        for case in range(per_class):
            top = rng.uniform(3.0, 5.0)
            bottom = rng.uniform(4.5, 7.0)
            boundary = rng.uniform(0.8, 1.5)
            model = np.where(zz < boundary, top, bottom).astype(np.float32)
            params: dict[str, Any] = {"boundary_m": float(boundary), "top_eps": float(top), "bottom_eps": float(bottom)}
            if class_name in {"single_pipe", "double_pipe"}:
                centers = [(rng.uniform(0.7, 2.8), rng.uniform(0.35, 1.7))]
                if class_name == "double_pipe":
                    first_x, first_z = centers[0]
                    centers.append((np.clip(first_x + rng.uniform(0.35, 0.8), 0.5, 3.0), np.clip(first_z + rng.uniform(-0.15, 0.15), 0.3, 1.8)))
                radius = rng.uniform(0.08, 0.28)
                pipe_eps = rng.uniform(7.0, 10.0)
                for cx, cz in centers:
                    model[(xx - cx) ** 2 + (zz - cz) ** 2 <= radius**2] = pipe_eps
                params.update({"centers_m": centers, "radius_m": float(radius), "target_eps": float(pipe_eps)})
            elif class_name == "low_eps_void":
                cx, cz = rng.uniform(0.7, 2.8), rng.uniform(0.4, 1.8)
                rx, rz = rng.uniform(0.12, 0.4), rng.uniform(0.08, 0.3)
                model[((xx - cx) / rx) ** 2 + ((zz - cz) / rz) ** 2 <= 1.0] = 2.0
                params.update({"center_m": [float(cx), float(cz)], "radii_m": [float(rx), float(rz)]})
            elif class_name == "dipping_interface":
                slope = rng.uniform(-0.35, 0.35)
                interface = boundary + slope * (xx - 1.7664)
                model = np.where(zz < interface, top, bottom).astype(np.float32)
                params["slope"] = float(slope)
            else:
                raise ValueError(f"Unknown OOD class {class_name}")
            normalized = np.clip((model - 2.0) / 8.0, 0.0, 1.0)
            models.append(torch.from_numpy(normalized[None]))
            metadata.append({"class": class_name, "case": case, "parameters": params})
    return torch.stack(models), metadata
