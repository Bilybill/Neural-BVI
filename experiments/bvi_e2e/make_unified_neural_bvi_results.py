"""Create manuscript artifacts from the frozen unified Neural-BVI benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_RESULTS = (
    HERE
    / "publication_la010010"
    / "full"
    / "neural_bvi_optimization_20260721_phase5"
    / "final_holdout"
)
DEFAULT_PAPER_FIGURES = REPO / "paper_grsl" / "figures"
METHOD_LABELS = {
    "nn": "Neural inverse",
    "mc_dropout": "MC dropout",
    "deep_ensemble": "Deep ensemble",
    "residual_map": "Residual MAP",
    "residual_laplace": "Residual Laplace",
    "neural_bvi": "Neural-BVI",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)


def to_epsilon(array: np.ndarray) -> np.ndarray:
    return 2.0 + 8.0 * np.squeeze(array).astype(np.float64)


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.04,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def image_panel(
    ax: mpl.axes.Axes,
    array: np.ndarray,
    title: str,
    cmap: str,
    *,
    vmin: float,
    vmax: float,
    aspect: str = "equal",
) -> mpl.image.AxesImage:
    handle = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect=aspect)
    ax.set_title(title, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#4b5563")
    return handle


def horizontal_colorbar(
    fig: mpl.figure.Figure,
    axes: list[mpl.axes.Axes],
    handle: mpl.image.AxesImage,
    ticks: list[float],
    label: str,
) -> None:
    left = axes[0].get_position()
    right = axes[-1].get_position()
    cax = fig.add_axes([left.x0, left.y0 - 0.040, right.x1 - left.x0, 0.010])
    colorbar = fig.colorbar(handle, cax=cax, orientation="horizontal", ticks=ticks)
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(length=1.8, width=0.4, labelsize=5.5, pad=1)
    colorbar.set_label(label, fontsize=5.8, labelpad=1)


def fixed_case_row(records: list[dict[str, str]]) -> dict[str, str]:
    candidates = [
        row
        for row in records
        if row.get("split") == "test"
        and row.get("method") == "neural_bvi"
        and row.get("view") == "mixed_5db"
    ]
    if not candidates:
        raise ValueError("No Neural-BVI test record with the fixed mixed_5db view")
    return candidates[0]


def make_case_figure(
    records: list[dict[str, str]],
    summary: dict[str, Any],
    output_stems: list[Path],
) -> dict[str, Any]:
    row = fixed_case_row(records)
    arrays_path = Path(row["arrays"])
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing visualization arrays: {arrays_path}")
    arrays = np.load(arrays_path)
    observation = np.squeeze(arrays["observation"]).astype(np.float64)
    truth = to_epsilon(arrays["truth"])
    center = to_epsilon(arrays["nn_prediction"])
    mean = to_epsilon(arrays["mean"])
    calibration_scale = float(summary["validation_scales"]["neural_bvi"])
    posterior_std = 8.0 * calibration_scale * np.squeeze(arrays["std"]).astype(np.float64)
    absolute_error = np.abs(mean - truth)
    event_probability = np.squeeze(arrays["event_probability"]).astype(np.float64)
    obs_limit = max(float(np.quantile(np.abs(observation), 0.995)), 1.0e-4)
    std_limit = max(float(np.quantile(posterior_std, 0.995)), 0.02)
    error_limit = max(float(np.quantile(absolute_error, 0.995)), 0.2)

    fig = plt.figure(figsize=(7.15, 3.05), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        4,
        left=0.025,
        right=0.98,
        bottom=0.11,
        top=0.96,
        width_ratios=[0.92, 1.0, 1.0, 1.0],
        wspace=0.12,
        hspace=0.43,
    )
    observed_axis = fig.add_subplot(grid[:, 0])
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(1, 4)]
    bottom_axes = [fig.add_subplot(grid[1, index]) for index in range(1, 4)]

    observed_axis.imshow(
        observation,
        cmap="gray",
        vmin=-obs_limit,
        vmax=obs_limit,
        origin="upper",
        aspect="auto",
    )
    observed_axis.set_title("Observed B-scan", pad=2)
    observed_axis.set_xticks([])
    observed_axis.set_yticks([])

    map_handles = [
        image_panel(top_axes[0], truth, "Ground truth", "cividis", vmin=2.0, vmax=10.0),
        image_panel(top_axes[1], center, "Ensemble center", "cividis", vmin=2.0, vmax=10.0),
        image_panel(top_axes[2], mean, "Neural-BVI mean", "cividis", vmin=2.0, vmax=10.0),
    ]
    std_handle = image_panel(
        bottom_axes[0], posterior_std, "Uncertainty (std.)", "viridis", vmin=0.0, vmax=std_limit
    )
    error_handle = image_panel(
        bottom_axes[1], absolute_error, "Absolute error", "inferno", vmin=0.0, vmax=error_limit
    )
    event_handle = image_panel(
        bottom_axes[2], event_probability, r"$P(\epsilon_r>6)$", "magma", vmin=0.0, vmax=1.0
    )

    for label, ax in zip("abcdefg", [observed_axis, *top_axes, *bottom_axes]):
        add_panel_label(ax, label)

    fig.canvas.draw()
    horizontal_colorbar(fig, top_axes, map_handles[-1], [2.0, 6.0, 10.0], r"relative permittivity $\epsilon_r$")
    horizontal_colorbar(fig, [bottom_axes[0]], std_handle, [0.0, std_limit], r"$\Delta\epsilon_r$")
    horizontal_colorbar(fig, [bottom_axes[1]], error_handle, [0.0, error_limit], r"$|\Delta\epsilon_r|$")
    horizontal_colorbar(fig, [bottom_axes[2]], event_handle, [0.0, 0.5, 1.0], "probability")

    save_figure(fig, output_stems[0])
    for stem in output_stems[1:]:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".svg", ".png"):
            shutil.copy2(output_stems[0].with_suffix(suffix), stem.with_suffix(suffix))
    plt.close(fig)
    return {
        "selection_rule": "First model in the precommitted Phase-5 holdout order at the fixed 5 dB noise view.",
        "global_index": int(row["global_index"]),
        "model_name": row["model_name"],
        "view": row["view"],
        "arrays": str(Path(row["arrays"]).resolve()),
        "arrays_sha256": sha256(arrays_path),
        "calibration_scale": calibration_scale,
        "normalized_rmse": finite(row.get("normalized_rmse")),
        "data_misfit": finite(row.get("data_misfit")),
        "crps": finite(row.get("crps")),
        "std_error_spearman": finite(row.get("std_error_spearman")),
        "ause": finite(row.get("ause")),
        "panel_order": [
            "observed_bscan",
            "ground_truth",
            "ensemble_center",
            "neural_bvi_mean",
            "validation_scaled_posterior_std",
            "absolute_error",
            "high_permittivity_probability",
        ],
        "posterior_std_display_max_epsilon_r": std_limit,
        "absolute_error_display_max_epsilon_r": error_limit,
    }


def publication_table(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, method_summary in summary["method_summary"].items():
        metrics = method_summary["metrics"]
        row: dict[str, Any] = {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "model_count": method_summary["model_count"],
            "record_count": method_summary["record_count"],
        }
        for metric in (
            "normalized_rmse",
            "data_misfit",
            "crps",
            "calibrated_coverage_95",
            "calibrated_coverage_error",
            "calibrated_gaussian_crps",
            "std_error_spearman",
            "ause",
            "physics_evaluations",
            "wall_seconds",
        ):
            row[f"{metric}_mean"] = metrics[metric]["mean"]
            row[f"{metric}_std"] = metrics[metric]["std"]
        rows.append(row)
    return rows


def mean_record_metric(records: list[dict[str, str]], method: str, metric: str) -> float:
    values = [
        value
        for row in records
        if row.get("split") == "test"
        and row.get("method") == method
        and (value := finite(row.get(metric))) is not None
    ]
    if not values:
        raise ValueError(f"No finite {metric} values for {method}")
    return float(np.mean(values))


def mechanism_summary(records: list[dict[str, str]], method: str) -> dict[str, Any]:
    rows = [row for row in records if row.get("split") == "test" and row.get("method") == method]
    effective = np.asarray([float(row["mixture_effective_components"]) for row in rows])
    stage_weights = [json.loads(row["stage_new_weights"]) for row in rows]
    result = {
        "record_count": len(rows),
        "effective_components_mean": float(effective.mean()),
        "effective_components_min": float(effective.min()),
        "effective_components_max": float(effective.max()),
    }
    max_stages = max((len(weights) for weights in stage_weights), default=0)
    for stage_index in range(1, max_stages):
        result[f"accepted_stage_{stage_index + 1}_count"] = int(
            sum(len(weights) > stage_index and float(weights[stage_index]) > 0.0 for weights in stage_weights)
        )
    return result


def comparison_status(row: dict[str, str]) -> str:
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    direction = row["favorable_direction"]
    if direction == "negative":
        if high < 0.0:
            return "favorable_ci"
        if low > 0.0:
            return "unfavorable_ci"
    else:
        if low > 0.0:
            return "favorable_ci"
        if high < 0.0:
            return "unfavorable_ci"
    return "inconclusive_ci"


def write_results_markdown(
    path: Path,
    table_rows: list[dict[str, Any]],
    planned: list[dict[str, str]],
    case: dict[str, Any],
) -> None:
    primary = [row for row in planned if row.get("role") == "primary"]
    lines = [
        "# Unified Neural-BVI Synthetic Results",
        "",
        "## Planned primary comparisons",
        "",
        "| Question | Baseline | Metric | Neural-BVI minus baseline | 95% model-clustered CI | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in primary:
        lines.append(
            "| {} | {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {} |".format(
                row["research_question"],
                METHOD_LABELS.get(row["baseline"], row["baseline"]),
                row["metric"],
                float(row["difference_proposed_minus_baseline"]),
                float(row["ci_low"]),
                float(row["ci_high"]),
                comparison_status(row),
            )
        )

    lines.extend(
        [
            "",
            "## Method means",
            "",
            "| Method | Image NRMSE | Data RMSE | CRPS | Cov.95 | Spearman | AUSE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in table_rows:
        def value(name: str) -> str:
            number = finite(row.get(name))
            return "--" if number is None else f"{number:.4f}"

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["label"],
                value("normalized_rmse_mean"),
                value("data_misfit_mean"),
                value("crps_mean"),
                value("calibrated_coverage_95_mean"),
                value("std_error_spearman_mean"),
                value("ause_mean"),
            )
        )

    lines.extend(
        [
            "",
            "## Fixed visualization case",
            "",
            f"The figure uses `{case['model_name']}` / `{case['view']}` by a fixed first-model rule.",
            f"Its Neural-BVI normalized RMSE is `{case['normalized_rmse']:.6f}` and data RMSE is `{case['data_misfit']:.6f}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper-figures", type=Path, default=DEFAULT_PAPER_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    paper_figures = args.paper_figures.resolve()
    summary_path = results_dir / "summary.json"
    metrics_path = results_dir / "metrics.csv"
    planned_path = results_dir / "planned_comparisons.csv"
    if not summary_path.exists() or not metrics_path.exists() or not planned_path.exists():
        raise FileNotFoundError("The completed unified benchmark outputs are required")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("Refusing to generate manuscript artifacts from an incomplete benchmark")
    records = read_csv(metrics_path)
    planned = read_csv(planned_path)
    table_rows = publication_table(summary)
    calibration = {
        method: {
            "validation_scale": float(scale),
            "raw_test_coverage_95": mean_record_metric(records, method, "coverage_95"),
            "scaled_test_coverage_95": mean_record_metric(records, method, "calibrated_coverage_95"),
        }
        for method, scale in summary["validation_scales"].items()
    }
    mechanism = {"neural_bvi": mechanism_summary(records, "neural_bvi")}

    figure_name = "fig_method_neural_bvi_synthetic_results"
    result_figure_stem = results_dir / "figures" / figure_name
    paper_figure_stem = paper_figures / figure_name
    case = make_case_figure(
        records,
        summary,
        [result_figure_stem, paper_figure_stem],
    )

    table_path = results_dir / "publication_table.csv"
    markdown_path = results_dir / "UNIFIED_RESULTS.md"
    write_csv(table_path, table_rows)
    write_results_markdown(markdown_path, table_rows, planned, case)
    write_csv(
        paper_figures / "fig_method_neural_bvi_synthetic_results_metrics.csv",
        [case],
    )

    provenance = {
        "status": "complete",
        "generator": str(Path(__file__).resolve()),
        "results_summary": str(summary_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "planned_comparisons": str(planned_path.resolve()),
        "publication_table": str(table_path.resolve()),
        "case": case,
        "calibration": calibration,
        "mechanism": mechanism,
        "figures": {
            suffix: {
                "path": str(paper_figure_stem.with_suffix(suffix).resolve()),
                "sha256": sha256(paper_figure_stem.with_suffix(suffix)),
            }
            for suffix in (".pdf", ".svg", ".png")
        },
        "display": {
            "permittivity_range": [2.0, 10.0],
            "posterior_std": "validation-scale calibrated; scale fixed before test",
            "event_threshold_epsilon_r": 6.0,
        },
    }
    provenance_path = paper_figures / "fig_method_neural_bvi_synthetic_results_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "figure": str(paper_figure_stem.with_suffix('.pdf'))}, indent=2))


if __name__ == "__main__":
    configure_matplotlib()
    main()
