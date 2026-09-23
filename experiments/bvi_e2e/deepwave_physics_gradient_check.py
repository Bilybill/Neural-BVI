"""Gradient and consistency smoke test for the LA010010 Deepwave physics backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from publication_metrics import normalized_data_rmse, pearson_flat
from publication_training import load_surrogate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "publication_la010010" / "full")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    checkpoint = args.checkpoint or (root / "surrogate" / "best.pt")
    out_path = args.out or (root / "surrogate" / "gradient_check.json")
    artifacts = torch.load(root / "prepared" / "artifacts.pt", map_location="cpu", weights_only=False)
    split_indices = list(artifacts["splits"][args.split])
    if not split_indices:
        raise ValueError(f"Split {args.split} is empty")
    global_index = int(split_indices[int(args.case) % len(split_indices)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate, payload = load_surrogate_checkpoint(checkpoint, device)

    model = artifacts["models"][global_index : global_index + 1].to(device).detach().clone()
    target = artifacts["clean_bscans"][global_index : global_index + 1].to(device)
    model.requires_grad_(True)

    started = time.time()
    prediction = surrogate(model)
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()
    elapsed = time.time() - started

    grad = model.grad.detach()
    result: dict[str, Any] = {
        "status": "pass",
        "checkpoint": str(checkpoint),
        "architecture": payload.get("architecture"),
        "surrogate_kind": payload.get("surrogate_kind"),
        "split": args.split,
        "global_index": global_index,
        "model_name": artifacts["model_names"][global_index],
        "device": str(device),
        "seconds": elapsed,
        "loss": float(loss.detach().cpu()),
        "prediction_nrmse": normalized_data_rmse(prediction.detach().cpu(), target.detach().cpu()),
        "prediction_pearson": pearson_flat(prediction.detach().cpu(), target.detach().cpu()),
        "grad_abs_sum": float(grad.abs().sum().cpu()),
        "grad_abs_mean": float(grad.abs().mean().cpu()),
        "grad_abs_max": float(grad.abs().max().cpu()),
        "grad_finite": bool(torch.isfinite(grad).all().item()),
        "prediction_finite": bool(torch.isfinite(prediction).all().item()),
        "input_requires_grad": bool(model.requires_grad),
        "output_requires_grad": bool(prediction.requires_grad),
    }
    if (
        not result["grad_finite"]
        or not result["prediction_finite"]
        or result["grad_abs_sum"] <= 0.0
        or not result["output_requires_grad"]
    ):
        result["status"] = "fail"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
