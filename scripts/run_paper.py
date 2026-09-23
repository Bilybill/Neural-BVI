"""Replay the frozen final evaluation when trusted original artifacts are present."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTIVE_KEYS = {"basis_type", "basis_source", "latent_prior_source", "posterior_std_prior_scale"}


def required_artifacts(root):
    return [root / "protocol.snapshot.json", root / "prepared/artifacts.pt",
            root / "surrogate/best.pt", *[root / f"train/unet/seed_{seed}/best.pt"
                                           for seed in (7, 71, 711, 1701, 2701)]]


def build_command(artifact_root, output):
    spec = json.loads((ROOT / "configs/paper/run_spec.json").read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / "experiments/bvi_e2e/run_unified_neural_bvi_comparison.py"),
               "--root", str(artifact_root.resolve()), "--out-dir", str(output.resolve()),
               "--methods", ",".join(spec["methods"]), "--splits", "test",
               "--model-count", str(spec["model_count"]), "--test-local-indices",
               ",".join(map(str, spec["test_local_indices"])), "--seed", str(spec["seed"]),
               "--bootstrap-resamples", "10000"]
    for key, value in spec["config"].items():
        if key not in DESCRIPTIVE_KEYS:
            serialized = ",".join(map(str, value)) if isinstance(value, list) else str(value)
            command += ["--" + key.replace("_", "-"), serialized]
    for option, filename in [("ensemble-center-json", "ensemble_center.json"),
                             ("posterior-std-prior-json", "std_prior.json"),
                             ("external-calibration-json", "calibration.json")]:
        command += ["--" + option, str(ROOT / "configs/paper" / filename)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/paper")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    command = build_command(args.artifact_root, args.output)
    missing = [str(path) for path in required_artifacts(args.artifact_root.resolve()) if not path.is_file()]
    if args.dry_run:
        print(json.dumps({"cwd": str(ROOT), "argv": command, "missing_artifacts": missing}, indent=2))
        return 0
    if missing:
        print("Original artifacts are required; see docs/data_and_models.md:\n" + "\n".join(missing),
              file=sys.stderr)
        return 2
    if args.check_only:
        print("All required files exist. This checks presence, not checkpoint contents or provenance.")
        return 0
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty; archived results must not be overwritten")
    print("Working directory:", ROOT)
    print("Command (display only):", shlex.join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
