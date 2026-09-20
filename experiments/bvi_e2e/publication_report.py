"""Aggregate publication outputs and enforce statistical/claim gates."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from publication_metrics import holm_adjust, paired_bootstrap
from publication_training import write_csv


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if record.get(key) not in {"", None, "nan"}]
    return float(np.mean(values)) if values else float("nan")


def stage_status(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    if summary is None:
        return {"status": "missing", "summary": str(path)}
    return {"status": summary.get("status", "unknown"), "summary": str(path)}


def deterministic_run_paths(protocol: dict[str, Any], output_root: Path, smoke: bool) -> list[Path]:
    backbones = protocol["backbones"][:1] if smoke else protocol["backbones"]
    seeds = protocol["training_seeds"][:1] if smoke else protocol["training_seeds"]
    return [output_root / "train" / backbone / f"seed_{seed}" for backbone in backbones for seed in seeds]


def load_deterministic_records(protocol: dict[str, Any], output_root: Path, smoke: bool) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for run_dir in deterministic_run_paths(protocol, output_root, smoke):
        records.extend(read_csv(run_dir / "test_metrics.csv"))
    return records


def summarize_deterministic(records: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    metrics = ("epsilon_rmse", "epsilon_mae", "ssim", "gradient_rmse", "high_eps_iou", "high_eps_f1")
    backbones = sorted({record["backbone"] for record in records})
    views = sorted({record["view"] for record in records})
    for backbone in backbones:
        summary[backbone] = {}
        for view in views:
            subset = [record for record in records if record["backbone"] == backbone and record["view"] == view]
            if not subset:
                continue
            metric_summary: dict[str, Any] = {"n": len(subset)}
            for metric in metrics:
                all_values = np.asarray([float(record[metric]) for record in subset], dtype=np.float64)
                seed_means = []
                for seed in sorted({record["seed"] for record in subset}, key=lambda value: int(value)):
                    seed_subset = [record for record in subset if record["seed"] == seed]
                    seed_means.append(float(np.mean([float(record[metric]) for record in seed_subset])))
                seed_values = np.asarray(seed_means, dtype=np.float64)
                metric_summary[metric] = {
                    "mean": float(seed_values.mean()),
                    "seed_std": float(seed_values.std()),
                    "case_std": float(all_values.std()),
                }
            summary[backbone][view] = metric_summary
            rows.append(
                {
                    "backbone": backbone,
                    "view": view,
                    "n": metric_summary["n"],
                    "epsilon_rmse_mean": metric_summary["epsilon_rmse"]["mean"],
                    "epsilon_rmse_seed_std": metric_summary["epsilon_rmse"]["seed_std"],
                    "epsilon_mae_mean": metric_summary["epsilon_mae"]["mean"],
                    "ssim_mean": metric_summary["ssim"]["mean"],
                    "gradient_rmse_mean": metric_summary["gradient_rmse"]["mean"],
                    "high_eps_iou_mean": metric_summary["high_eps_iou"]["mean"],
                    "high_eps_f1_mean": metric_summary["high_eps_f1"]["mean"],
                }
            )
    return summary, rows


def load_data_ablation_records(protocol: dict[str, Any], output_root: Path, smoke: bool) -> list[dict[str, str]]:
    primary_seed = int(protocol["training_seeds"][0])
    sources = [
        ("field_residual_plus_gaussian", output_root / "train" / "unet" / f"seed_{primary_seed}" / "test_metrics.csv"),
        ("clean_only", output_root / "data_ablation" / "clean" / f"seed_{primary_seed}" / "test_metrics.csv"),
    ]
    if not smoke:
        sources.append(
            ("gaussian_only", output_root / "data_ablation" / "gaussian" / f"seed_{primary_seed}" / "test_metrics.csv")
        )
    records: list[dict[str, str]] = []
    for mode, path in sources:
        for record in read_csv(path):
            copied = dict(record)
            copied["data_mode"] = mode
            records.append(copied)
    return records


def summarize_data_ablation(records: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    metrics = ("epsilon_rmse", "epsilon_mae", "ssim", "gradient_rmse", "high_eps_iou", "high_eps_f1")
    modes = sorted({record["data_mode"] for record in records})
    views = sorted({record["view"] for record in records})
    for mode in modes:
        summary[mode] = {}
        for view in views:
            subset = [record for record in records if record["data_mode"] == mode and record["view"] == view]
            if not subset:
                continue
            metric_summary: dict[str, Any] = {"n": len(subset)}
            for metric in metrics:
                values = np.asarray([float(record[metric]) for record in subset], dtype=np.float64)
                metric_summary[metric] = {"mean": float(values.mean()), "case_std": float(values.std())}
            summary[mode][view] = metric_summary
            rows.append(
                {
                    "data_mode": mode,
                    "view": view,
                    "n": metric_summary["n"],
                    "epsilon_rmse_mean": metric_summary["epsilon_rmse"]["mean"],
                    "epsilon_mae_mean": metric_summary["epsilon_mae"]["mean"],
                    "ssim_mean": metric_summary["ssim"]["mean"],
                    "gradient_rmse_mean": metric_summary["gradient_rmse"]["mean"],
                    "high_eps_iou_mean": metric_summary["high_eps_iou"]["mean"],
                    "high_eps_f1_mean": metric_summary["high_eps_f1"]["mean"],
                }
            )
    return summary, rows


def compute_uq_statistics(
    protocol: dict[str, Any],
    output_root: Path,
    deterministic: list[dict[str, str]],
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], list[dict[str, str]]]:
    uq = read_csv(output_root / "uq" / "test_metrics.csv")
    primary_uq = [record for record in uq if record["method"] == "neural_bvi"]
    baseline_map = {
        (record["global_index"], record["view"], record["seed"]): float(record["epsilon_rmse"])
        for record in deterministic
        if record["backbone"] == "unet" and record["noise_mode"] == "mixed"
    }
    paired_baseline, paired_method = [], []
    for record in primary_uq:
        key = (record["global_index"], record["view"], record["seed"])
        if key in baseline_map:
            paired_baseline.append(baseline_map[key])
            paired_method.append(float(record["epsilon_rmse"]))

    primary_statistics = paired_bootstrap(
        np.asarray(paired_baseline),
        np.asarray(paired_method),
        int(protocol["statistics"]["bootstrap_resamples"] if not smoke else 100),
        float(protocol["statistics"]["confidence"]),
    )
    primary_statistics["paired_case_count"] = len(paired_method)

    method_means: dict[str, Any] = {}
    p_values = {}
    for method in sorted({record["method"] for record in uq}):
        subset = [record for record in uq if record["method"] == method]
        method_means[method] = {
            "epsilon_rmse": mean_metric(subset, "epsilon_rmse"),
            "crps": mean_metric(subset, "crps"),
            "coverage_95": mean_metric(subset, "coverage_95"),
            "event_auroc": mean_metric(
                [record for record in subset if record.get("event_auroc") not in {"", "nan"}], "event_auroc"
            ),
        }
        if method != "neural_bvi":
            common = min(len(subset), len(primary_uq))
            if common:
                _, p = stats.wilcoxon(
                    [float(record["epsilon_rmse"]) for record in subset[:common]],
                    [float(record["epsilon_rmse"]) for record in primary_uq[:common]],
                    zero_method="zsplit",
                )
                p_values[method] = float(p)
    return primary_statistics, method_means, holm_adjust(p_values) if p_values else {}, uq


def metric_value(summary: dict[str, Any], backbone: str, view: str, metric: str) -> float:
    try:
        return float(summary[backbone][view][metric]["mean"])
    except KeyError:
        return float("nan")


def run_publication_report(protocol: dict[str, Any], output_root: Path, smoke: bool) -> dict[str, Any]:
    required = [output_root / "surrogate" / "summary.json"]
    required.extend(run_dir / "summary.json" for run_dir in deterministic_run_paths(protocol, output_root, smoke))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Publication report is blocked by missing required outputs: {missing}")

    deterministic = load_deterministic_records(protocol, output_root, smoke)
    deterministic_summary, deterministic_rows = summarize_deterministic(deterministic)
    data_ablation_records = load_data_ablation_records(protocol, output_root, smoke)
    data_ablation_summary, data_ablation_rows = summarize_data_ablation(data_ablation_records)
    surrogate = json.loads((output_root / "surrogate" / "summary.json").read_text(encoding="utf-8"))
    uq_summary = read_json(output_root / "uq" / "summary.json")
    uq_complete = bool(
        uq_summary
        and uq_summary.get("status") == "complete"
        and (output_root / "uq" / "test_metrics.csv").exists()
        and (surrogate.get("status") == "pass" or smoke)
    )

    ancillary = {
        "ood": stage_status(output_root / "ood" / "summary.json"),
        "frequency": stage_status(output_root / "frequency" / "summary.json"),
        "deepwave_reforward": stage_status(output_root / "deepwave_reforward" / "summary.json"),
        "fdtd": stage_status(output_root / "fdtd" / "summary.json"),
        "field": stage_status(output_root / "field" / "summary.json"),
    }
    ancillary_complete = all(value["status"] == "complete" for value in ancillary.values())

    if uq_complete:
        primary_statistics, method_means, adjusted, _ = compute_uq_statistics(
            protocol, output_root, deterministic, smoke
        )
        status = "complete"
        blocked_reason = None
        posterior_std_scale = float(uq_summary["posterior_std_scale"])
    else:
        primary_statistics = {
            "status": "not_run",
            "paired_case_count": 0,
            "mean_difference": None,
            "ci_low": None,
            "ci_high": None,
            "accuracy_success": False,
        }
        method_means: dict[str, Any] = {}
        adjusted: dict[str, float] = {}
        posterior_std_scale = None
        if surrogate.get("status") != "pass" and not smoke:
            status = "deterministic_complete_bvi_blocked"
            blocked_reason = (
                "Forward surrogate failed the pre-registered BVI gate; formal UQ/field posterior "
                "experiments are intentionally not used for publication claims."
            )
        else:
            status = "deterministic_complete_uq_missing"
            blocked_reason = "UQ outputs are not available yet."

    paper_replacement_ready = bool(
        uq_complete
        and primary_statistics["accuracy_success"]
        and surrogate["status"] == "pass"
        and ancillary_complete
        and not smoke
    )
    experiment_workflow_complete = bool(
        deterministic
        and uq_complete
        and surrogate["status"] == "pass"
        and ancillary_complete
        and not smoke
    )
    uq_artifact_kind = uq_summary.get("artifact_kind") if uq_summary else None
    uq_claim_boundary = uq_summary.get("claim_boundary") if uq_summary else None
    report = {
        "status": status,
        "protocol_hash": protocol["protocol_hash"],
        "smoke": smoke,
        "experiment_workflow_complete": experiment_workflow_complete,
        "surrogate_gate": surrogate,
        "blocked_reason": blocked_reason,
        "deterministic_summary": deterministic_summary,
        "data_ablation_summary": data_ablation_summary,
        "ancillary_outputs": ancillary,
        "primary_endpoint": {
            "metric": "epsilon_rmse",
            "comparison": "Neural-BVI posterior mean minus deterministic U-Net",
            **primary_statistics,
        },
        "method_means": method_means,
        "holm_adjusted_p": adjusted,
        "posterior_std_scale": posterior_std_scale,
        "uq_artifact_kind": uq_artifact_kind,
        "uq_claim_boundary": uq_claim_boundary,
        "paper_replacement_ready": paper_replacement_ready,
        "claim_rule": (
            "Claim improved inversion accuracy only when the paired bootstrap upper CI is below zero; "
            "otherwise limit the claim to data consistency and uncertainty diagnostics."
        ),
    }

    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "deterministic_summary.csv", deterministic_rows)
    write_csv(report_dir / "data_ablation_summary.csv", data_ablation_rows)
    (report_dir / "publication_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# LA010010 Publication Experiment Report",
        "",
        f"- Protocol hash: `{protocol['protocol_hash']}`",
        f"- Report status: `{status}`",
        f"- Surrogate gate: `{surrogate['status']}` "
        f"(NRMSE `{float(surrogate['validation_nrmse']):.4f}`, Pearson `{float(surrogate['validation_pearson']):.4f}`)",
        f"- Experiment workflow complete: `{experiment_workflow_complete}`",
        f"- Paper replacement ready: `{paper_replacement_ready}`",
    ]
    if uq_artifact_kind:
        lines.append(f"- UQ artifact kind: `{uq_artifact_kind}`")
    if uq_claim_boundary:
        lines.append(f"- UQ claim boundary: {uq_claim_boundary}")
    if blocked_reason:
        lines.append(f"- Blocked reason: {blocked_reason}")
    lines.extend(
        [
            "",
            "## Deterministic IID test means",
            "",
            "| Backbone | View | epsilon RMSE | epsilon MAE | SSIM | high-eps F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deterministic_rows:
        lines.append(
            f"| {row['backbone']} | {row['view']} | "
            f"{float(row['epsilon_rmse_mean']):.5f} +/- {float(row['epsilon_rmse_seed_std']):.5f} | "
            f"{float(row['epsilon_mae_mean']):.5f} | {float(row['ssim_mean']):.5f} | "
            f"{float(row['high_eps_f1_mean']):.5f} |"
        )
    if data_ablation_rows:
        lines.extend(
            [
                "",
                "## Data ablation IID test means",
                "",
                "| Data mode | View | epsilon RMSE | epsilon MAE | SSIM | high-eps F1 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in data_ablation_rows:
            lines.append(
                f"| {row['data_mode']} | {row['view']} | {float(row['epsilon_rmse_mean']):.5f} | "
                f"{float(row['epsilon_mae_mean']):.5f} | {float(row['ssim_mean']):.5f} | "
                f"{float(row['high_eps_f1_mean']):.5f} |"
            )
    lines.extend(["", "## BVI/UQ status", ""])
    if uq_complete:
        lines.extend(
            [
                f"- Paired primary cases: `{primary_statistics['paired_case_count']}`",
                f"- Neural-BVI minus U-Net epsilon RMSE: `{primary_statistics['mean_difference']:.6f}`",
                f"- Bootstrap 95% CI: `[{primary_statistics['ci_low']:.6f}, {primary_statistics['ci_high']:.6f}]`",
                f"- Accuracy-success gate: `{primary_statistics['accuracy_success']}`",
                "",
                "| Method | epsilon RMSE | CRPS | 95% coverage | Event AUROC |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, values in method_means.items():
            lines.append(
                f"| {method} | {values['epsilon_rmse']:.5f} | {values['crps']:.5f} | "
                f"{values['coverage_95']:.5f} | {values['event_auroc']:.5f} |"
            )
    else:
        lines.append("Formal BVI/UQ is not included because the forward surrogate gate has not passed.")
    lines.extend(["", "## Ancillary stage status", "", "| Stage | Status |", "| --- | --- |"])
    for name, value in ancillary.items():
        lines.append(f"| {name} | `{value['status']}` |")
    (report_dir / "PUBLICATION_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex_lines = [
        "% Generated by publication_report.py; do not edit manually.",
        f"\\newcommand{{\\UNetCleanRMSE}}{{{metric_value(deterministic_summary, 'unet', 'clean', 'epsilon_rmse'):.4f}}}",
        f"\\newcommand{{\\UNetZeroDbRMSE}}{{{metric_value(deterministic_summary, 'unet', 'mixed_0db', 'epsilon_rmse'):.4f}}}",
        f"\\newcommand{{\\SurrogateNRMSE}}{{{float(surrogate['validation_nrmse']):.4f}}}",
        f"\\newcommand{{\\BVIStatus}}{{{status.replace('_', '-')}}}",
    ]
    if uq_complete:
        tex_lines.extend(
            [
                f"\\newcommand{{\\PrimaryDeltaRMSE}}{{{primary_statistics['mean_difference']:.4f}}}",
                f"\\newcommand{{\\PrimaryCILow}}{{{primary_statistics['ci_low']:.4f}}}",
                f"\\newcommand{{\\PrimaryCIHigh}}{{{primary_statistics['ci_high']:.4f}}}",
            ]
        )
    (report_dir / "paper_results.generated.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    return report
