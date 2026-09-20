"""OOD, frequency, Deepwave re-forward, FDTD, and LA010010 transfer stages."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch

from build_deepwave_dataset import load_profile, simulate_one
from e2e_bvi_gpr import json_safe, run_bvi
from fdtd_validation import FDTD_CASES, load_normalized_permittivity, preprocess_fdtd_bscan
from publication_data import build_fixed_views, generate_ood_models, resolve_protocol_path, synthesize_noise_view
from publication_metrics import deterministic_metrics, normalized_data_rmse
from publication_training import load_inversion_checkpoint, load_surrogate_checkpoint, predict_batches, set_seed, write_csv
from ramac_field_validation import align_ramac_observation, read_rd3_pair


def acquisition_profile(protocol: dict[str, Any]) -> dict[str, Any]:
    path = resolve_protocol_path(Path(protocol["protocol_path"]), protocol["acquisition_profile"])
    return load_profile(path, "la010010_pipe_native", None)


def load_surrogate_gate(output_root: Path) -> dict[str, Any]:
    path = output_root / "surrogate" / "summary.json"
    if not path.exists():
        return {"status": "missing", "summary": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_ood(protocol: dict[str, Any], artifacts: dict[str, Any], output_root: Path, smoke: bool) -> dict[str, Any]:
    out_dir = output_root / "ood"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "ood_dataset.pt"
    if data_path.exists():
        return torch.load(data_path, map_location="cpu", weights_only=False)
    cfg = dict(protocol["ood"])
    if smoke:
        cfg["per_class"] = 1
    models, metadata = generate_ood_models(cfg)
    profile = acquisition_profile(protocol)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean = []
    for index, model in enumerate(models):
        print(f"OOD Deepwave {index + 1}/{len(models)}")
        permittivity = model.squeeze().numpy() * 8.0 + 2.0
        clean.append(torch.from_numpy(simulate_one(permittivity, profile, device, 32, 20))[None])
    clean_tensor = torch.stack(clean)
    indices = list(range(len(models)))
    views, view_metadata = build_fixed_views(
        clean_tensor, indices, artifacts["residual_bank"], protocol["noise"], split_tag=3
    )
    payload = {
        "format": "la010010-ood-v1",
        "protocol_hash": protocol["protocol_hash"],
        "models": models,
        "clean_bscans": clean_tensor,
        "metadata": metadata,
        "views": views,
        "view_metadata": view_metadata,
    }
    torch.save(payload, data_path)
    return payload


def run_ood_evaluation(
    protocol: dict[str, Any], artifacts: dict[str, Any], output_root: Path, smoke: bool
) -> dict[str, Any]:
    payload = prepare_ood(protocol, artifacts, output_root, smoke)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = protocol["training_seeds"][:1] if smoke else protocol["training_seeds"]
    backbones = protocol["backbones"][:1] if smoke else protocol["backbones"]
    records = []
    for backbone in backbones:
        for seed in seeds:
            path = output_root / "train" / backbone / f"seed_{seed}" / "best.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            model, _ = load_inversion_checkpoint(path, device)
            for view_name, observation in payload["views"].items():
                predictions = predict_batches(model, observation, int(protocol["training"]["batch_size"]), device)
                metrics = deterministic_metrics(
                    predictions, payload["models"], float(protocol["uq"]["event_threshold_normalized"])
                )
                for index, metric in enumerate(metrics):
                    records.append(
                        metric
                        | {
                            "backbone": backbone,
                            "seed": seed,
                            "view": view_name,
                            "ood_class": payload["metadata"][index]["class"],
                            "ood_case": payload["metadata"][index]["case"],
                            "protocol_hash": protocol["protocol_hash"],
                        }
                    )
    path = output_root / "ood" / "metrics.csv"
    write_csv(path, records)
    summary = {"status": "complete", "case_count": len(payload["models"]), "record_count": len(records), "metrics": str(path)}
    (output_root / "ood" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_frequency_sensitivity(
    protocol: dict[str, Any], artifacts: dict[str, Any], output_root: Path, smoke: bool
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(output_root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)
    profile = acquisition_profile(protocol)
    fixed = artifacts["fixed"]["validation"]
    locals_ = fixed["uq_local_indices"][:1] if smoke else fixed["uq_local_indices"]
    records = []
    out_dir = output_root / "frequency"
    out_dir.mkdir(parents=True, exist_ok=True)
    for frequency in protocol["ood"]["frequency_hz"]:
        frequency_profile = copy.deepcopy(profile)
        frequency_profile["source"]["center_frequency_hz"] = float(frequency)
        for local in locals_:
            global_index = fixed["indices"][local]
            truth = artifacts["models"][global_index : global_index + 1]
            eps = truth.squeeze().numpy() * 8.0 + 2.0
            clean = torch.from_numpy(simulate_one(eps, frequency_profile, device, 32, 20))[None, None]
            observation, meta = synthesize_noise_view(
                clean[0],
                artifacts["residual_bank"],
                int(protocol["noise"]["evaluation_seed"]) + global_index + int(frequency),
                5.0,
                "mixed",
                float(protocol["noise"]["field_weight"]),
                float(protocol["noise"]["gaussian_weight"]),
            )
            observation = observation[None]
            prediction = model(observation.to(device)).cpu()
            metric = deterministic_metrics(prediction, truth, float(protocol["uq"]["event_threshold_normalized"]))[0]
            records.append(metric | {"frequency_hz": frequency, "global_index": global_index, "noise_source_index": meta["source_index"]})
    path = out_dir / "metrics.csv"
    write_csv(path, records)
    summary = {"status": "complete", "record_count": len(records), "metrics": str(path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_deepwave_reforward(
    protocol: dict[str, Any], artifacts: dict[str, Any], output_root: Path, smoke: bool
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(output_root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)
    profile = acquisition_profile(protocol)
    fixed = artifacts["fixed"]["test"]
    locals_ = fixed["uq_local_indices"][:1] if smoke else fixed["uq_local_indices"]
    records = []
    for local in locals_:
        for snr in protocol["uq"]["snrs_db"]:
            view = f"mixed_{float(snr):g}db"
            observation = fixed["views"][view][local : local + 1]
            with torch.no_grad():
                prediction = model(observation.to(device)).cpu()
            eps = prediction.squeeze().numpy() * 8.0 + 2.0
            predicted_clean = torch.from_numpy(simulate_one(eps, profile, device, 32, 20))[None, None]
            target_clean = fixed["views"]["clean"][local : local + 1]
            records.append(
                {
                    "global_index": fixed["indices"][local],
                    "model_name": fixed["model_names"][local],
                    "view": view,
                    "snr_db": snr,
                    "deepwave_clean_nrmse": normalized_data_rmse(predicted_clean, target_clean),
                }
            )
    out_dir = output_root / "deepwave_reforward"
    path = out_dir / "metrics.csv"
    write_csv(path, records)
    summary = {"status": "complete", "record_count": len(records), "metrics": str(path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_fdtd_transfer(protocol: dict[str, Any], output_root: Path, smoke: bool) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(output_root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)
    surrogate_summary = load_surrogate_gate(output_root)
    surrogate = None
    if smoke or surrogate_summary.get("status") == "pass":
        surrogate, _ = load_surrogate_checkpoint(output_root / "surrogate" / "best.pt", device)
    simulation_dir = Path(__file__).resolve().parents[2] / "GPR_simulation"
    cases = FDTD_CASES[:1] if smoke else FDTD_CASES
    records = []
    out_dir = output_root / "fdtd"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, record_file, model_file in cases:
        record = sio.loadmat(simulation_dir / record_file)
        observation_np, obs_meta = preprocess_fdtd_bscan(record["co_data"], 179, 137)
        truth_np, model_meta = load_normalized_permittivity(simulation_dir / model_file, 256, np.asarray(record.get("pos", [])))
        observation = torch.from_numpy(observation_np)[None, None]
        truth = torch.from_numpy(truth_np)[None, None]
        with torch.no_grad():
            prediction = model(observation.to(device)).cpu()
            proxy_prediction = surrogate(prediction.to(device)).cpu() if surrogate is not None else None
        metric = deterministic_metrics(prediction, truth, float(protocol["uq"]["event_threshold_normalized"]))[0]
        if proxy_prediction is not None:
            proxy_status = "computed"
            proxy_nrmse = normalized_data_rmse(proxy_prediction, observation)
        else:
            proxy_status = "omitted_surrogate_gate_failed"
            proxy_nrmse = float("nan")
        records.append(
            metric
            | {
                "case": name,
                "proxy_observation_nrmse": proxy_nrmse,
                "proxy_observation_nrmse_status": proxy_status,
                "record_metadata": json.dumps(obs_meta),
                "model_metadata": json.dumps(model_meta),
                "boundary": "external TM-Maxwell OOD; frequency and sampling are not LA010010-aligned",
            }
        )
    path = out_dir / "metrics.csv"
    write_csv(path, records)
    summary = {
        "status": "complete",
        "case_count": len(records),
        "metrics": str(path),
        "surrogate_gate": surrogate_summary,
        "claim_boundary": "Per-case OOD stress test only; no significance claim.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_field_transfer(protocol: dict[str, Any], artifacts: dict[str, Any], output_root: Path, smoke: bool) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(protocol["training_seeds"][0])
    model, _ = load_inversion_checkpoint(output_root / "train" / "unet" / f"seed_{seed}" / "best.pt", device)
    surrogate_summary = load_surrogate_gate(output_root)
    profile = acquisition_profile(protocol)
    from path_utils import resolve_data_path

    field_root = resolve_data_path(
        resolve_protocol_path(Path(protocol["protocol_path"]), protocol["field_root"]),
        "Field_GPR_DATA.lnk",
        "Filed_GPR_Data.lnk",
    )
    rad_path = field_root / Path(profile["field_reference"]["relative_rad_path"])
    raw, header = read_rd3_pair(rad_path)
    bscan, crop = align_ramac_observation(raw, header, profile)
    observation = torch.from_numpy(bscan)[None, None]
    with torch.no_grad():
        prediction = model(observation.to(device)).cpu()
    out_dir = output_root / "field"
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "la010010_arrays.npz"
    if surrogate_summary.get("status") != "pass" and not smoke:
        np.savez_compressed(
            arrays_path,
            observation=observation.numpy(),
            nn_prediction=prediction.numpy(),
        )
        summary = {
            "status": "blocked_surrogate_gate",
            "source": str(rad_path),
            "crop": crop,
            "artificial_noise_added": False,
            "surrogate_gate": surrogate_summary,
            "arrays": str(arrays_path),
            "claim_boundary": "No pixelwise ground truth; deterministic qualitative transfer only until the BVI surrogate gate passes.",
            "reason": "Forward surrogate failed the pre-registered BVI gate; posterior/event statistics are not generated.",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    surrogate, _ = load_surrogate_checkpoint(output_root / "surrogate" / "best.pt", device)
    with torch.no_grad():
        proxy_nn = surrogate(prediction.to(device)).cpu()
    uq_summary_path = output_root / "uq" / "summary.json"
    config = (
        json.loads(uq_summary_path.read_text(encoding="utf-8"))["selected_config"]
        if uq_summary_path.exists()
        else dict(protocol["uq"]["base"])
    )
    if smoke:
        config.update({"components": 1, "steps": 1, "samples_per_component": 1})
    field_bvi_seed = int(protocol["noise"]["evaluation_seed"]) + int(seed) * 101 + 10010
    set_seed(field_bvi_seed)
    result, mean, std, event, samples = run_bvi(
        forward=surrogate,
        observation=observation,
        true_model=None,
        nn_mean=prediction,
        latent_dim=int(config["latent_dim"]),
        components=int(config["components"]),
        steps=int(config["steps"]),
        samples_per_component=int(config["samples_per_component"]),
        residual_scale=float(config["residual_scale"]),
        noise_std=0.08,
        kl_weight=float(config["kl_weight"]),
        model_prior_std=float(config["model_prior_std"]),
        interrogation_threshold=float(protocol["uq"]["event_threshold_normalized"]),
        basis_type=str(config["basis_type"]),
        device=device,
        posterior_sample_count=32 if smoke else int(protocol["uq"]["posterior_samples"]),
        return_samples=True,
    )
    std_scale = (
        float(json.loads(uq_summary_path.read_text(encoding="utf-8"))["posterior_std_scale"])
        if uq_summary_path.exists()
        else 1.0
    )
    np.savez_compressed(
        arrays_path,
        observation=observation.numpy(),
        nn_prediction=prediction.numpy(),
        bvi_mean=mean.numpy(),
        bvi_std=std.numpy(),
        bvi_std_calibrated=(std * std_scale).numpy(),
        event_probability=event.numpy(),
    )
    bvi_summary = json_safe(result.__dict__)
    data_misfit_delta = None
    if result.data_misfit_nn is not None and result.data_misfit_bvi_mean is not None:
        data_misfit_delta = float(result.data_misfit_bvi_mean) - float(result.data_misfit_nn)
    summary = {
        "status": "complete",
        "source": str(rad_path),
        "crop": crop,
        "artificial_noise_added": False,
        "surrogate_gate": surrogate_summary,
        "nn_proxy_nrmse": normalized_data_rmse(proxy_nn, observation),
        "field_bvi_seed": field_bvi_seed,
        "bvi": bvi_summary,
        "data_misfit_delta_bvi_minus_nn": data_misfit_delta,
        "posterior_std_scale_from_validation": std_scale,
        "arrays": str(arrays_path),
        "claim_boundary": "No pixelwise ground truth; qualitative transfer and proxy consistency only.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
