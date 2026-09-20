# Local release validation

The release preparation was checked on Windows with Python 3.11.15 and the numerical versions listed in `requirements-tested.txt`. These are local results, not a statement that GitHub CI or the full benchmark has run.

| Check | Outcome |
| --- | --- |
| Pytest suite | 23 passed; 1 skipped because full smoke artifacts are absent; 4 backbone subtests passed |
| License packaging | Apache-2.0 metadata, LICENSE, NOTICE and third-party notices included |
| Public-interface lint | Passed (`src`, `scripts`, `tests`) |
| Export provenance | 46 source/export file hashes verified |
| Archived result aggregation | 360 distinct records; 112 mean/std comparisons passed |
| Procedural CPU demo | Training, likelihood-gradient check and two-component residual BVI completed |
| Same-seed repeatability | Posterior arrays and training-loss sequence match exactly in the tested CPU runtime |
| Deepwave derivative | Directional finite-difference agreement passed |
| Differentiable filtering | Matches SciPy SOS forward-backward filtering in tested orders |
| Built wheel | Installed into a fresh virtual environment; imports resolve there; demo succeeds outside the source tree |
| Frozen paper command | Parsed by the research CLI; uses six methods and test local indices 64–83 |

The wheel test inherits numerical dependencies from the existing scientific environment to avoid downloading another PyTorch installation. It does not establish a from-scratch dependency install on every OS. The CI matrix is configured for fresh Windows/Linux CPU jobs, but remains unexecuted until the owner pushes the repository.

The test suite emits one existing PyTorch Transformer nested-tensor optimization warning; it is not a numerical failure. No original datasets, trained networks, manuscript files or private recordings were included in the source candidate. Exact full training/evaluation was not rerun during packaging.
