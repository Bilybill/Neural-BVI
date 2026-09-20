"""Unified, resumable publication experiment runner for LA010010."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from publication_data import load_protocol, prepare_protocol_artifacts, resolve_protocol_path
from publication_report import run_publication_report
from publication_training import train_forward_surrogate, train_inversion
from publication_transfer import (
    run_deepwave_reforward,
    run_fdtd_transfer,
    run_field_transfer,
    run_frequency_sensitivity,
    run_ood_evaluation,
)
from publication_uq import run_publication_uq


STAGES = ("prepare", "train", "uq", "ood", "fdtd", "field", "report")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_root(protocol: dict[str, Any], smoke: bool) -> Path:
    base = resolve_protocol_path(Path(protocol["protocol_path"]), protocol["output_root"])
    return base / ("smoke" if smoke else "full")


def artifacts_path(root: Path) -> Path:
    return root / "prepared" / "artifacts.pt"


def load_artifacts(root: Path) -> dict[str, Any]:
    path = artifacts_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared artifacts {path}; run --stage prepare first")
    return torch.load(path, map_location="cpu", weights_only=False)


def run_prepare(protocol: dict[str, Any], root: Path, smoke: bool, resume: bool) -> dict[str, Any]:
    path = artifacts_path(root)
    manifest_path = root / "prepared" / "manifest.json"
    if resume and path.exists() and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = prepare_protocol_artifacts(protocol, smoke=smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        "format": payload["format"],
        "protocol_hash": protocol["protocol_hash"],
        "smoke": smoke,
        "artifacts": str(path),
        "artifacts_sha256": sha256(path),
        "dataset_manifest": payload["dataset_manifest"],
        "dataset_profile": payload["dataset_manifest_profile"],
        "tensor_shapes": {
            "models": list(payload["models"].shape),
            "clean_bscans": list(payload["clean_bscans"].shape),
            "residual_bank": list(payload["residual_bank"].shape),
        },
        "split_counts": {key: len(value) for key, value in payload["splits"].items()},
        "split_model_names": payload["split_model_names"],
        "residual_keys": payload["residual_keys"],
        "leakage_guard": all("la010010" not in key.lower() for key in payload["residual_keys"]),
        "fixed_views": {
            split: {name: list(tensor.shape) for name, tensor in payload["fixed"][split]["views"].items()}
            for split in ("validation", "test")
        },
        "uq_global_indices": {
            split: payload["fixed"][split]["uq_global_indices"] for split in ("validation", "test")
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_train(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    root: Path,
    smoke: bool,
    resume: bool,
    backbone_filter: str | None,
    seed_filter: int | None,
    surrogate_only: bool = False,
) -> dict[str, Any]:
    surrogate_summary_path = root / "surrogate" / "summary.json"
    if not (resume and surrogate_summary_path.exists()):
        train_forward_surrogate(protocol, artifacts, root / "surrogate", smoke=smoke)
    if surrogate_only:
        return {"status": "complete", "runs": [], "surrogate": str(surrogate_summary_path)}
    backbones = [backbone_filter] if backbone_filter else (protocol["backbones"][:1] if smoke else protocol["backbones"])
    seeds = [seed_filter] if seed_filter is not None else (protocol["training_seeds"][:1] if smoke else protocol["training_seeds"])
    completed = []
    for backbone in backbones:
        for seed in seeds:
            out_dir = root / "train" / backbone / f"seed_{seed}"
            if resume and (out_dir / "summary.json").exists():
                completed.append(str(out_dir))
                continue
            train_inversion(protocol, artifacts, backbone, int(seed), out_dir, "mixed", smoke=smoke)
            completed.append(str(out_dir))
    # Complete the five-member U-Net ensemble without repeating the three registered seeds.
    if backbone_filter in {None, "unet"} and seed_filter is None:
        ensemble_seeds = protocol["ensemble_seeds"][:1] if smoke else protocol["ensemble_seeds"]
        for seed in ensemble_seeds:
            out_dir = root / "train" / "unet" / f"seed_{seed}"
            if resume and (out_dir / "summary.json").exists():
                continue
            train_inversion(protocol, artifacts, "unet", int(seed), out_dir, "mixed", smoke=smoke)
    # Data-domain ablations use the primary seed only.
    if backbone_filter is None and seed_filter is None:
        primary_seed = int(protocol["training_seeds"][0])
        modes = ["clean"] if smoke else ["clean", "gaussian"]
        for mode in modes:
            out_dir = root / "data_ablation" / mode / f"seed_{primary_seed}"
            if resume and (out_dir / "summary.json").exists():
                continue
            train_inversion(protocol, artifacts, "unet", primary_seed, out_dir, mode, smoke=smoke)
    summary = {"status": "complete", "runs": completed, "surrogate": str(surrogate_summary_path)}
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "train" / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=here / "la010010_protocol.json")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backbone", choices=["unet", "unetpp", "transunet", "tinynet"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--surrogate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol.resolve())
    root = output_root(protocol, args.smoke)
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.snapshot.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    requested = STAGES if args.stage == "all" else (args.stage,)
    results = {}
    artifacts = None
    for stage in requested:
        print(f"=== publication stage: {stage} ===", flush=True)
        if stage == "prepare":
            results[stage] = run_prepare(protocol, root, args.smoke, args.resume)
            continue
        if artifacts is None:
            artifacts = load_artifacts(root)
        if artifacts["protocol_hash"] != protocol["protocol_hash"]:
            raise ValueError("Prepared artifacts do not match the current protocol hash")
        if stage == "train":
            results[stage] = run_train(
                protocol,
                artifacts,
                root,
                args.smoke,
                args.resume,
                args.backbone,
                args.seed,
                args.surrogate_only,
            )
        elif stage == "uq":
            results[stage] = run_publication_uq(protocol, artifacts, root, smoke=args.smoke)
        elif stage == "ood":
            results[stage] = {
                "ood": run_ood_evaluation(protocol, artifacts, root, args.smoke),
                "frequency": run_frequency_sensitivity(protocol, artifacts, root, args.smoke),
                "deepwave_reforward": run_deepwave_reforward(protocol, artifacts, root, args.smoke),
            }
        elif stage == "fdtd":
            results[stage] = run_fdtd_transfer(protocol, root, args.smoke)
        elif stage == "field":
            results[stage] = run_field_transfer(protocol, artifacts, root, args.smoke)
        elif stage == "report":
            results[stage] = run_publication_report(protocol, root, args.smoke)
    print(json.dumps({key: value.get("status", "complete") for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
