"""Verify archived aggregates without requiring the original data or PyTorch."""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"nn", "mc_dropout", "deep_ensemble", "residual_map", "residual_laplace", "neural_bvi"}
UQ_METHODS = METHODS - {"nn", "residual_map"}
METRICS = ("normalized_rmse", "data_misfit", "crps", "calibrated_coverage_95",
           "calibrated_coverage_error", "std_error_spearman")


def verify(directory):
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "publication_table.csv").open(encoding="utf-8", newline="") as stream:
        table_rows = list(csv.DictReader(stream))
    table = {row["method"]: row for row in table_rows}
    if len(table_rows) != 6 or set(table) != METHODS:
        raise ValueError("Publication table must contain the six distinct methods")
    keys = {(row["split"], row["global_index"], row["view"], row["method"]) for row in rows}
    counts = Counter(row["method"] for row in rows)
    if len(rows) != 360 or len(keys) != 360 or counts != Counter(dict.fromkeys(METHODS, 60)):
        raise ValueError("Expected 360 unique records and 60 per method")
    models = {row["global_index"] for row in rows}
    expected = {("test", model, view, method) for model in models
                for view in ("mixed_0db", "mixed_5db", "mixed_10db") for method in METHODS}
    if len(models) != 20 or keys != expected:
        raise ValueError("Expected the full 20-model x 3-view x 6-method test matrix")
    report = {}
    checks = 0
    for method in sorted(METHODS):
        records = [row for row in rows if row["method"] == method]
        report[method] = {}
        for metric in METRICS:
            if metric not in METRICS[:2] and method not in UQ_METHODS:
                if any(row[metric] for row in records):
                    raise ValueError(f"Unexpected posterior metric for {method}")
                continue
            values = [float(row[metric]) for row in records]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite {method}/{metric}")
            stats = {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
            for stat, value in stats.items():
                targets = [summary["method_summary"][method]["metrics"][metric][stat],
                           float(table[method][f"{metric}_{stat}"])]
                if not all(math.isclose(value, target, rel_tol=1e-10, abs_tol=1e-12)
                           for target in targets):
                    raise ValueError(f"Aggregate mismatch: {method}/{metric}/{stat}")
                checks += len(targets)
            report[method][metric] = stats["mean"]
    return {"status": "pass", "records": len(rows), "models": len(models),
            "aggregate_checks": checks, "means": report,
            "scope": "archived aggregate verification, not an experiment rerun"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/paper")
    print(json.dumps(verify(parser.parse_args().results), indent=2, allow_nan=False))
