# Reproducibility

## Self-contained example

```bash
python -m neural_bvi doctor
python -m neural_bvi demo --seed 7 --output outputs/demo
python -m pytest
```

The example uses 12 training models and one held-out procedural model. It trains before BVI, checks a finite nonzero data-term gradient, and saves posterior arrays and a figure. No claim of accuracy improvement is required to pass the smoke test. CPU results are checked for repeatability; GPU/OS differences may change floating-point outputs and timing.

## Verify the reported results

```bash
python scripts/verify_results.py
```

This checks the identities and completeness of 360 model/view/method records (20 × 3 × 6), recomputes the metric means and standard deviations, and compares them with `results/paper/summary.json` and `publication_table.csv`. Additional metrics and model-clustered confidence intervals are included in the result files. This command checks saved results, not a new inference run.

## Paper evaluation with original artifacts

First provide the artifacts listed in [Data and models](data_and_models.md). Then:

```bash
python scripts/run_paper.py --artifact-root artifacts/paper --check-only
python scripts/run_paper.py --artifact-root artifacts/paper --dry-run
python scripts/run_paper.py --artifact-root artifacts/paper --output outputs/paper
```

The wrapper reads `configs/paper/run_spec.json`, supplies all supported frozen hyperparameters, six methods and test local indices 64–83, and uses the included calibration/center/prior assets. It prints the complete command. An absent or incomplete artifact directory fails early with a concrete file list. No data are downloaded implicitly.

PCA uses training errors only. A fresh execution records input hashes and writes its own run specification and outputs. Do not overwrite the released result directory. The compact configuration JSON files contain the final runtime parameters, not a hyperparameter search.

Full training/evaluation has not been rerun as part of repository packaging. Five-network training and repeated 256×256 wave solves are substantially more expensive than the demo. Timing in archived results is hardware-specific.

## Regeneration from input data

With the original model files and authorized residual-noise sources available:

```bash
python experiments/bvi_e2e/build_deepwave_dataset.py --model-dir data/models --field-root data/field --profile la010010_pipe_native --model-count 1000 --shard-size 8 --device cuda --out-dir experiments/bvi_e2e/datasets/deepwave_la010010_256
python experiments/bvi_e2e/run_la010010_publication.py --stage prepare
python experiments/bvi_e2e/run_la010010_publication.py --stage train
```

These commands prepare inputs and train the five-network ensemble. The training entry point creates the Deepwave physics checkpoint from the acquisition profile. The defaults define a new training run; exact replay requires the original artifact bundle and its matching protocol snapshot. Use `scripts/run_paper.py` for the six-method evaluation.

## Evaluation protocol

The final evaluation covers 20 models at three noise levels. Model identities are listed in `configs/paper/test_set.json`. Development data include previously evaluated model blocks, while this final subset was kept separate during parameter selection. The method and calibration are fixed for the reported evaluation. Further tuning requires separate development data and an untouched test set.

## Distribution

`RELEASE_FILES.txt` lists every public source-archive file. `MANIFEST.in` lists the same files for the source distribution. Build with `python -m build` or `python scripts/build_release.py --output dist/neural-bvi-source.zip`. The generated ZIP includes content checksums; local datasets, outputs, environments and Git metadata are excluded.
