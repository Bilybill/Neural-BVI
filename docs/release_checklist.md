# Maintainer release checklist

The source repository is [Bilybill/Neural-BVI](https://github.com/Bilybill/Neural-BVI). Source publication alone does not imply a tagged release, DOI or successful remote CI run.

## Before the first push

- [x] Project owner selected Apache-2.0; full license, attribution and [scope](../LICENSE_POLICY.md) are included.
- [x] Owner confirmed `Bilybill/Neural-BVI`; author attribution is recorded in CITATION.cff.
- [ ] Review `provenance/source_manifest.json`, included calibration maps and archived metrics.
- [ ] Keep raw measured data, manuscript submission files and model weights out of Git.
- [ ] Run the checks below and inspect the generated `RELEASE_MANIFEST.json` in the archive.

```bash
python -m pytest
python -m ruff check src tests scripts
python scripts/verify_results.py
python scripts/check_repository.py
python -m build
python scripts/build_release.py --output dist/neural-bvi-source.zip
```

For subsequent changes, review `git diff --cached`, commit the selected source files, and push to the repository's default branch. Do not use the original research repository's remote implicitly. Enable branch protection with a passing CI requirement, dependency updates and available secret-scanning protections. Community files follow [GitHub's healthy-contribution guidance](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions).

Wait for the Windows/Linux CI matrix to pass before attaching a version tag. The repository URL is recorded in `CITATION.cff` and package metadata. Add badges only after their endpoints exist. Archive a tagged release and add a DOI only after it is actually minted.

## Artifact release, separately

For data/weights, first confirm redistribution permission, then publish a separately versioned bundle with acquisition/split metadata, size, SHA-256 and a data/weights license. Test download and evaluation on a clean machine. Until then keep the source-only availability statement visible.

## Scope of local checks

The unit tests verify deterministic procedural execution, forward differentiation, filtering, existing research regressions, frozen command construction, and archived aggregate consistency. They do not prove exact full-benchmark replay or field-data redistribution rights. One legacy smoke-artifact test is skipped when the full prepared bundle is absent.
