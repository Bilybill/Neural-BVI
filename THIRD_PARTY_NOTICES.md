# Third-party components

Deepwave, PyTorch, NumPy, SciPy, Matplotlib and scikit-image are installed as dependencies, not copied into this repository. Their respective license texts are distributed with their packages. The optional measured-data workflow uses BM3D; review that package's terms before using or redistributing it, especially outside academic work.

External datasets, manuscript PDFs, trained weights and third-party binaries are not distributed with this source release.

The two NPZ files under `configs/paper/` are synthetic-development calibration maps, not measured radar recordings. Their hashes are in the adjacent configuration JSON files; their roles are described in the method documentation.

Project-authored materials are licensed under Apache-2.0; see [LICENSE_POLICY.md](LICENSE_POLICY.md) for scope. This does not relicense separately installed dependencies. Retain existing third-party notices when extending the code.
