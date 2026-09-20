# Contributing

Start with the CPU demo and the method/reproducibility documentation. Open a focused issue before proposing a new inference method or changing the reported experiment protocol.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests scripts
python scripts/verify_results.py
python scripts/check_repository.py
```

Use four spaces, explicit units, type hints for new public APIs, and small functions. Add a regression test for every bug fix. Keep numerical changes separate from formatting and documentation. The preserved research modules are syntax-checked and regression-tested; linting is scoped to the new public package, scripts and tests so that packaging does not silently rewrite the research kernels.

## Scientific changes

- Record seeds, splits, configurations, environment and artifact hashes.
- Fit calibration or tune methods on development data only; never silently retune the released holdout.
- Do not replace archived metrics with improved values. Put new experiments under a new version and document differences.
- Explain whether an uncertainty map is sample dispersion, augmented dispersion or calibrated dispersion.
- For changed exported research files, retain the original `source_sha256`, update `export_sha256`, and describe the change in `provenance/source_manifest.json`.

## Pull requests

Describe the problem, scope, tests run and any behavior/numerical changes. Do not include raw radar files, checkpoints, local paths, credentials, generated caches or large outputs. Redact sensitive data from logs. Follow [the code of conduct](CODE_OF_CONDUCT.md).

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion are provided under Apache-2.0, consistent with Section 5 of [LICENSE](LICENSE). Submit only material you are entitled to contribute and retain applicable third-party notices. No separate contributor license agreement is required by this repository.
