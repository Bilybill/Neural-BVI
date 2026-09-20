# Third-party components and source provenance

The numerical research modules were exported from the authors' DeepwaveForGPR working tree. `provenance/source_manifest.json` records original and portable-file SHA-256 hashes and mechanical transformations. This is a source snapshot, not a claim that every file was tracked in the original Git history.

Deepwave, PyTorch, NumPy, SciPy, Matplotlib and scikit-image are installed as dependencies, not copied into this repository. Their respective license texts are distributed with their packages. The optional measured-data workflow uses BM3D; review that package's terms before using or redistributing it, especially outside academic work.

The original project also contained MATLAB FDTD code attributed to James Irving. Those MATLAB files, external datasets, manuscript PDFs, trained weights and third-party binaries are **not** part of this source snapshot. This release does not grant rights to those materials.

The two small NPZ files under `configs/paper/` are synthetic-development calibration maps. They are not measured radar recordings. Their original hashes and derivation roles are recorded in the manifest and method documentation.

Project-authored materials are licensed under Apache-2.0; see [LICENSE_POLICY.md](LICENSE_POLICY.md) for scope. This does not relicense separately installed dependencies. Retain existing third-party notices when extending the code.
