"""Publication UQ matrix: BVI tuning, baselines, calibration, and saved posterior products."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from e2e_bvi_gpr import run_bvi
from inversion_models import enable_mc_dropout
from publication_metrics import calibration_scale, uq_metrics
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, set_seed, write_csv


def uq_cases(protocol: dict[str, Any], artifacts: dict[str, Any], split: str) -> Iterator[dict[str, Any]]:
    fixed = artifacts["fixed"][split]
    clean = fixed["views"]["clean"]
    for local_index in fixed["uq_local_indices"]:
        for snr in protocol["uq"]["snrs_db"]:
            view = f"mixed_{float(snr):g}db"
            yield {
                "split": split,
                "local_index": int(local_index),
                "global_index": int(fixed["indices"][local_index]),
                "model_name": fixed["model_names"][local_index],
                "view": view,
                "snr_db": float(snr),
                "observation": fixed["views"][view][local_index : local_index + 1],
                "clean_observation": clean[local_index : local_index + 1],
                "truth": artifacts["models"][fixed["indices"][local_index] : fixed["indices"][local_index] + 1],
            }


def resolved_bvi_config(protocol: dict[str, Any], override: dict[str, Any], smoke: bool) -> dict[str, Any]:
    config = dict(protocol["uq"]["base"])
    config.update({key: value for key, value in override.items() if key != "name"})
    config["name"] = override.get("name", "base")
    if smoke:
        config.update({"steps": 1, "samples_per_component": 1, "components": min(int(config["components"]), 1)})
    return config


def save_posterior_arrays(
    path: Path,
    case: dict[str, Any],
    nn_prediction: torch.Tensor,
    samples: torch.Tensor,
    event_threshold: float,
    std_override: torch.Tensor | None = None,
) -> None:
    sample_np = samples.numpy().astype(np.float32)
    sample_std = sample_np.std(axis=0, keepdims=True)
    reported_std = (
        std_override.detach().cpu().numpy().astype(np.float32)
        if std_override is not None
        else sample_std
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observation=case["observation"].numpy(),
        truth=case["truth"].numpy(),
        nn_prediction=nn_prediction.numpy(),
        mean=sample_np.mean(axis=0, keepdims=True),
        std=reported_std,
        sample_std=sample_std,
        q025=np.quantile(sample_np, 0.025, axis=0, keepdims=True).astype(np.float32),
        q975=np.quantile(sample_np, 0.975, axis=0, keepdims=True).astype(np.float32),
        event_probability=np.mean(sample_np >= event_threshold, axis=0, keepdims=True).astype(np.float32),
    )


def run_bvi_case(
    protocol: dict[str, Any],
    case: dict[str, Any],
    model: torch.nn.Module,
    surrogate: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    arrays_path: Path | None = None,
    std_scale: float = 1.0,
    smoke: bool = False,
) -> dict[str, Any]:
    set_seed(seed + int(case["global_index"]) * 101 + int(case["snr_db"] * 10))
    with torch.no_grad():
        nn_prediction = model(case["observation"].to(device)).cpu()
    noise = case["observation"] - case["clean_observation"]
    noise_std = max(float(torch.sqrt(torch.mean(noise**2))), 0.01)
    posterior_count = 32 if smoke else int(protocol["uq"]["posterior_samples"])
    result, mean, std, event, samples = run_bvi(
        forward=surrogate,
        observation=case["observation"],
        true_model=case["truth"],
        nn_mean=nn_prediction,
        latent_dim=int(config["latent_dim"]),
        components=int(config["components"]),
        steps=int(config["steps"]),
        samples_per_component=int(config["samples_per_component"]),
        residual_scale=float(config["residual_scale"]),
        noise_std=noise_std,
        kl_weight=float(config["kl_weight"]),
        model_prior_std=float(config["model_prior_std"]),
        interrogation_threshold=float(protocol["uq"]["event_threshold_normalized"]),
        basis_type=str(config["basis_type"]),
        device=device,
        posterior_sample_count=posterior_count,
        return_samples=True,
    )
    metrics = uq_metrics(
        samples,
        case["truth"],
        float(protocol["uq"]["event_threshold_normalized"]),
        std_scale=std_scale,
    )
    if arrays_path is not None:
        save_posterior_arrays(
            arrays_path,
            case,
            nn_prediction,
            samples,
            float(protocol["uq"]["event_threshold_normalized"]),
        )
    return metrics | {
        "global_index": case["global_index"],
        "model_name": case["model_name"],
        "view": case["view"],
        "snr_db": case["snr_db"],
        "noise_std": noise_std,
        "arrays": str(arrays_path) if arrays_path else None,
        "bvi": asdict(result),
    }


def select_candidate(candidate_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries = []
    for name in sorted({record["candidate"] for record in candidate_records}):
        subset = [record for record in candidate_records if record["candidate"] == name]
        summaries.append(
            {
                "candidate": name,
                "epsilon_rmse": float(np.mean([record["epsilon_rmse"] for record in subset])),
                "crps": float(np.mean([record["crps"] for record in subset])),
                "gaussian_nll": float(np.mean([record["gaussian_nll"] for record in subset])),
            }
        )
    minimum = min(record["epsilon_rmse"] for record in summaries)
    eligible = [record for record in summaries if record["epsilon_rmse"] <= 1.01 * minimum]
    selected = min(eligible, key=lambda record: (record["crps"], record["gaussian_nll"]))
    return selected, summaries


def samples_from_mc_dropout(
    model: torch.nn.Module, observation: torch.Tensor, count: int, device: torch.device
) -> torch.Tensor:
    enable_mc_dropout(model)
    with torch.no_grad():
        samples = torch.cat([model(observation.to(device)).cpu() for _ in range(count)], dim=0)
    model.eval()
    return samples


def run_publication_uq(
    protocol: dict[str, Any],
    artifacts: dict[str, Any],
    output_root: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uq_root = output_root / "uq"
    uq_root.mkdir(parents=True, exist_ok=True)
    surrogate_summary_path = output_root / "surrogate" / "summary.json"
    surrogate_summary = json.loads(surrogate_summary_path.read_text(encoding="utf-8"))
    if surrogate_summary["status"] != "pass" and not smoke:
        summary = {
            "status": "blocked_surrogate_gate",
            "protocol_hash": protocol["protocol_hash"],
            "surrogate_gate": surrogate_summary,
            "reason": (
                "Forward surrogate failed the pre-registered gate; formal BVI/UQ is not run "
                "until validation NRMSE and Pearson satisfy the protocol."
            ),
        }
        (uq_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    surrogate, _ = load_surrogate_checkpoint(output_root / "surrogate" / "best.pt", device)
    primary_seed = int(protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(
        output_root / "train" / "unet" / f"seed_{primary_seed}" / "best.pt", device
    )

    candidates = protocol["uq"]["candidates"][:2] if smoke else protocol["uq"]["candidates"]
    validation_cases = list(uq_cases(protocol, artifacts, "validation"))
    if smoke:
        validation_cases = validation_cases[:1]
    candidate_records: list[dict[str, Any]] = []
    candidate_configs: dict[str, dict[str, Any]] = {}
    for override in candidates:
        config = resolved_bvi_config(protocol, override, smoke)
        candidate_configs[config["name"]] = config
        for case in validation_cases:
            record = run_bvi_case(protocol, case, model, surrogate, config, device, primary_seed, smoke=smoke)
            candidate_records.append({key: value for key, value in record.items() if key != "bvi"} | {"candidate": config["name"]})
    selected_summary, candidate_summaries = select_candidate(candidate_records)
    selected_config = candidate_configs[selected_summary["candidate"]]
    write_csv(uq_root / "validation_candidate_metrics.csv", candidate_records)

    calibration_records = []
    for case in validation_cases:
        arrays_path = uq_root / "calibration" / f"model_{case['global_index']}_{case['view']}.npz"
        record = run_bvi_case(
            protocol, case, model, surrogate, selected_config, device, primary_seed, arrays_path, smoke=smoke
        )
        calibration_records.append(record)
    scale = calibration_scale(calibration_records)

    test_cases = list(uq_cases(protocol, artifacts, "test"))
    if smoke:
        test_cases = test_cases[:1]
    test_records: list[dict[str, Any]] = []
    # Full Neural-BVI for all registered main seeds.
    for seed in protocol["training_seeds"]:
        checkpoint_path = output_root / "train" / "unet" / f"seed_{seed}" / "best.pt"
        if not checkpoint_path.exists():
            if smoke:
                continue
            raise FileNotFoundError(checkpoint_path)
        seed_model, _ = load_inversion_checkpoint(checkpoint_path, device)
        for case in test_cases:
            arrays_path = uq_root / "test" / "neural_bvi" / f"seed_{seed}_model_{case['global_index']}_{case['view']}.npz"
            record = run_bvi_case(
                protocol,
                case,
                seed_model,
                surrogate,
                selected_config,
                device,
                int(seed),
                arrays_path,
                std_scale=scale,
                smoke=smoke,
            )
            test_records.append({key: value for key, value in record.items() if key != "bvi"} | {"method": "neural_bvi", "seed": seed})

    # K=1 and no-trust BVI baselines on the primary seed.
    for method, override in (
        ("single_gaussian_vi", {"name": "k1", "components": 1}),
        ("bvi_no_trust", {"name": "no_trust", "components": 3, "model_prior_std": 0.0}),
    ):
        config = resolved_bvi_config(protocol, override, smoke)
        for case in test_cases:
            arrays_path = uq_root / "test" / method / f"model_{case['global_index']}_{case['view']}.npz"
            record = run_bvi_case(
                protocol, case, model, surrogate, config, device, primary_seed, arrays_path, smoke=smoke
            )
            test_records.append({key: value for key, value in record.items() if key != "bvi"} | {"method": method, "seed": primary_seed})

    # MC-dropout baseline.
    for case in test_cases:
        samples = samples_from_mc_dropout(
            model,
            case["observation"],
            3 if smoke else int(protocol["uq"]["mc_dropout_samples"]),
            device,
        )
        metrics = uq_metrics(samples, case["truth"], float(protocol["uq"]["event_threshold_normalized"]))
        arrays_path = uq_root / "test" / "mc_dropout" / f"model_{case['global_index']}_{case['view']}.npz"
        with torch.no_grad():
            nn_prediction = model(case["observation"].to(device)).cpu()
        save_posterior_arrays(arrays_path, case, nn_prediction, samples, float(protocol["uq"]["event_threshold_normalized"]))
        test_records.append(metrics | {"method": "mc_dropout", "seed": primary_seed, "global_index": case["global_index"], "model_name": case["model_name"], "view": case["view"], "snr_db": case["snr_db"], "arrays": str(arrays_path)})

    # Five-member ensemble (or every available member in smoke mode).
    ensemble_models = []
    for seed in protocol["ensemble_seeds"]:
        path = output_root / "train" / "unet" / f"seed_{seed}" / "best.pt"
        if path.exists():
            ensemble_models.append(load_inversion_checkpoint(path, device)[0])
        elif not smoke:
            raise FileNotFoundError(path)
    if ensemble_models:
        for case in test_cases:
            with torch.no_grad():
                samples = torch.cat([member(case["observation"].to(device)).cpu() for member in ensemble_models])
            metrics = uq_metrics(samples, case["truth"], float(protocol["uq"]["event_threshold_normalized"]))
            arrays_path = uq_root / "test" / "deep_ensemble" / f"model_{case['global_index']}_{case['view']}.npz"
            save_posterior_arrays(arrays_path, case, samples.mean(0, keepdim=True), samples, float(protocol["uq"]["event_threshold_normalized"]))
            test_records.append(metrics | {"method": "deep_ensemble", "seed": -1, "global_index": case["global_index"], "model_name": case["model_name"], "view": case["view"], "snr_db": case["snr_db"], "arrays": str(arrays_path)})

    write_csv(uq_root / "test_metrics.csv", test_records)
    summary = {
        "status": "complete",
        "protocol_hash": protocol["protocol_hash"],
        "selected_candidate": selected_summary,
        "selected_config": selected_config,
        "candidate_summaries": candidate_summaries,
        "posterior_std_scale": scale,
        "validation_case_count": len(validation_cases),
        "test_case_count": len(test_cases),
        "test_record_count": len(test_records),
        "test_metrics": str(uq_root / "test_metrics.csv"),
    }
    (uq_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
