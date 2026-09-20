# Reproducibility

## Level 1: execute the public example

```bash
python -m neural_bvi doctor
python -m neural_bvi demo --seed 7 --output outputs/demo
python -m pytest
```

The example uses 12 training models and one held-out procedural model. It trains before BVI, checks a finite nonzero data-term gradient, and saves posterior arrays and a figure. No claim of accuracy improvement is required to pass the smoke test. CPU results are checked for repeatability; GPU/OS differences may change floating-point outputs and timing.

## Level 2: re-aggregate the paper results

```bash
python scripts/verify_results.py
```

This checks 360 unique model/view/method records (20 × 3 × 6), recomputes Image NRMSE, Data RMSE and CRPS means, and compares them with `results/paper/summary.json` and `publication_table.csv`. It does not rerun network training or infer missing uncertainty samples. Additional metrics and model-clustered confidence intervals remain in the archived CSV files.

## Level 3: full final evaluation with original artifacts

First provide the artifacts listed in [Data and models](data_and_models.md). Then:

```bash
python scripts/run_paper.py --artifact-root artifacts/paper --check-only
python scripts/run_paper.py --artifact-root artifacts/paper --dry-run
python scripts/run_paper.py --artifact-root artifacts/paper --output outputs/paper
```

The wrapper reads `configs/paper/run_spec.json`, supplies all supported frozen hyperparameters, six methods and test local indices 64–83, and uses the included calibration/center/prior assets. It prints the complete command. An absent or incomplete artifact directory fails early with a concrete file list. No data are downloaded implicitly.

The numerical kernels retain their source provenance. Relative-path serialization changes JSON byte hashes, so the runner may create a new deterministic PCA cache name; it still uses the same training-only basis construction and parameters. Original source/run hashes are evidence of the previous run, not checksums of the portable derivative files. A fresh execution writes its own run specification; do not mix it with the archived result directory or resume across differing specifications.

Full training/evaluation has not been rerun as part of repository packaging. Five-network training and repeated 256×256 wave solves are substantially more expensive than the demo. Timing in archived results is hardware-specific.

## Regeneration from input data

With the original model files and authorized residual-noise sources available:

```bash
python experiments/bvi_e2e/build_deepwave_dataset.py --model-dir data/models --field-root data/field --profile la010010_pipe_native --model-count 1000 --shard-size 8 --device cuda --out-dir experiments/bvi_e2e/datasets/deepwave_la010010_256
python experiments/bvi_e2e/run_la010010_publication.py --stage prepare
python experiments/bvi_e2e/run_la010010_publication.py --stage train
```

These are the dataset/training stages of the research pipeline. Their current protocol defaults are not a substitute for the archived prepared bundle: use the original snapshot, all five ensemble seeds, the physics backend and frozen development assets to match the final study. The legacy `--stage all` also runs historical diagnostics and is not the final six-method paper command. Consult each script's `--help` before expensive runs.

## Experiment history

The final table comes from the Phase-5 holdout. Earlier Phase-1 through Phase-4 exposed blocks informed development. The frozen configuration and holdout membership are released so that this history is explicit. The final study reports three best mean metrics; it does not report uniform superiority in calibrated coverage or spatial uncertainty ranking. Further tuning should use a new development experiment and an untouched test set.
