"""Loading utilities and a differentiable surrogate for Deepwave datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "deepwave-gpr-dataset-v1":
        raise ValueError(f"Unsupported dataset format in {path}")
    return manifest


def load_deepwave_dataset(
    manifest_path: Path,
    count: int | None = None,
    return_clean: bool = False,
) -> tuple:
    """Load saved shards as ``models, bscans, [clean_bscans,] names, manifest``.

    Models are normalized permittivity tensors [N, 1, 256, 256]. B-scans are
    acquisition-aligned, preprocessed tensors [N, 1, T, X].
    """

    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    models: list[torch.Tensor] = []
    bscans: list[torch.Tensor] = []
    clean_bscans: list[torch.Tensor] = []
    names: list[str] = []
    remaining = count
    for shard in manifest["shards"]:
        payload = torch.load(manifest_path.parent / shard["file"], map_location="cpu", weights_only=False)
        shard_models = payload["models"].float()
        shard_bscans = payload["bscans"].float()
        shard_clean = payload.get("clean_bscans", shard_bscans).float()
        shard_names = list(payload["model_names"])
        take = len(shard_names) if remaining is None else min(remaining, len(shard_names))
        if take <= 0:
            break
        models.append(shard_models[:take])
        bscans.append(shard_bscans[:take])
        clean_bscans.append(shard_clean[:take])
        names.extend(shard_names[:take])
        if remaining is not None:
            remaining -= take
    if not models:
        raise ValueError(f"Dataset {manifest_path} contains no samples")
    if return_clean:
        return torch.cat(models), torch.cat(bscans), torch.cat(clean_bscans), names, manifest
    return torch.cat(models), torch.cat(bscans), names, manifest


class DeepwaveForwardSurrogate(nn.Module):
    """Differentiable map from normalized permittivity to Deepwave B-scans.

    Deepwave itself creates the saved dataset. This network is the tractable
    likelihood operator used inside multi-sample BVI; its validation error is
    recorded alongside each experiment and it must not be described as a full
    electromagnetic solver.
    """

    def __init__(self, n_time: int, n_traces: int):
        super().__init__()
        self.n_time = n_time
        self.n_traces = n_traces
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 1, 1),
            nn.Tanh(),
        )

    def forward(self, model: torch.Tensor) -> torch.Tensor:
        features = self.encoder(model)
        features = F.interpolate(
            features,
            size=(self.n_time, self.n_traces),
            mode="bilinear",
            align_corners=False,
        )
        return self.head(features)
