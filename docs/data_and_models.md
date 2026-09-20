# Data, models and permissions

| Artifact | Included | Purpose |
| --- | --- | --- |
| Procedural demo generator | Yes | A completely local training/inference example |
| Frozen per-record/aggregate metrics | Yes | Recompute the reported table without wave simulation |
| Frozen calibration JSON and two synthetic-derived NPZ maps | Yes | Document exact final inference settings |
| Original 1,000 model files and Deepwave shards | No | Full neural training and exact benchmark inputs |
| Prepared artifact bundle | No | Frozen splits, observations and models for final evaluation |
| Five trained inverse-network checkpoints | No | Neural and ensemble baselines/final BVI center |
| Original measured radar data/residual noise bank | No | Original noise distribution and field experiment |

No public download URL or dataset license is asserted for unavailable artifacts. The source-code license does not grant redistribution rights to measured data. Do not commit measured recordings or checkpoints to Git; use a separately reviewed release/archival repository with hashes and an explicit license when ready.

For exact replay, supply a trusted local artifact directory containing:

```text
artifacts/paper/
  protocol.snapshot.json
  prepared/artifacts.pt
  surrogate/best.pt
  train/unet/seed_7/best.pt
  train/unet/seed_71/best.pt
  train/unet/seed_711/best.pt
  train/unet/seed_1701/best.pt
  train/unet/seed_2701/best.pt
```

`surrogate/best.pt` must describe `DeepwavePhysicsSurrogate`. The original prepared bundle embeds the original protocol hash; editing its protocol to bypass a mismatch invalidates the exact replay contract. The source export makes machine paths relative and retains hashes of the original inputs in `provenance/source_manifest.json`.

`python scripts/run_paper.py --artifact-root artifacts/paper --check-only` checks required file presence before invoking the research runner. PyTorch research artifacts use pickle-backed loading: only load artifacts you created or obtained from a trusted source. For a future hosted artifact bundle, publish SHA-256 checksums and test the download in an empty environment first.

For training on your own data, put physical permittivity arrays into MATLAB files named `model_*.mat` with variable `model` or `ep`; use explicit `--model-dir` and `--field-root` arguments. The released acquisition JSON defines sample intervals, trace spacing, source wavelet and preprocessing. Different input data/noise produces a new experiment, not an exact replay of the archived table.
