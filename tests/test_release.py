import csv
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import tomllib

import pytest
import yaml

from check_repository import ROOT, check, release_files
from run_paper import build_command, required_artifacts
from verify_results import verify


def test_license_metadata_and_distribution_files():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert project["license"] == citation["license"] == "Apache-2.0"
    assert {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"} <= set(project["license-files"])
    included = {p.relative_to(ROOT).as_posix() for p in release_files()}
    assert set(project["license-files"]) <= included
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Version 2.0, January 2004" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text


def test_archived_results():
    report = verify(ROOT / "results/paper")
    assert report["records"] == 360
    assert report["aggregate_checks"] == 112


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "metric", "summary", "identities", "names"])
def test_verifier_rejects_corrupted_results(tmp_path, mutation):
    shutil.copytree(ROOT / "results/paper", tmp_path / "paper")
    target = tmp_path / "paper"
    with (target / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation in {"nan", "metric"}:
        rows[0]["normalized_rmse"] = "nan" if mutation == "nan" else "0.9"
    elif mutation == "identities":
        for row in rows:
            row["global_index"] = str(int(row["global_index"]) + 100000)
    elif mutation == "names":
        rows[0]["model_name"] = "wrong_model.mat"
    else:
        summary_path = target / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["method_summary"]["nn"]["metrics"]["normalized_rmse"]["mean"] = 0.9
        summary_path.write_text(json.dumps(summary))
    with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        verify(target)


def test_frozen_command_and_holdout(tmp_path, monkeypatch):
    command = build_command(tmp_path / "artifacts", tmp_path / "out")
    flags = dict(zip(command[2::2], command[3::2]))
    assert flags["--model-count"] == "20"
    assert flags["--splits"] == "test"
    assert flags["--test-local-indices"] == ",".join(map(str, range(64, 84)))
    assert flags["--components"] == "2"
    assert flags["--posterior-mean-policy"] == "physics_line_search"
    assert flags["--mc-dropout-dispersion-scale"] == "7.5"
    assert len(required_artifacts(tmp_path)) == 8
    # Parse actual arguments rather than --help, which exits before unknown-option checks.
    from run_unified_neural_bvi_comparison import parse_args
    monkeypatch.setattr(sys, "argv", command[1:])
    parsed = parse_args()
    assert parsed.model_count == 20 and parsed.posterior_mean_policy == "physics_line_search"
    holdout = json.loads((ROOT / "configs/paper/test_set.json").read_text())
    with (ROOT / "results/paper/metrics.csv").open(encoding="utf-8", newline="") as stream:
        ids = {int(row["global_index"]) for row in csv.DictReader(stream)}
    assert ids == set(holdout["selected_global_indices"])


def test_missing_artifacts_fail_with_actionable_message(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_paper.py"),
                             "--artifact-root", str(tmp_path), "--check-only"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "prepared" in result.stderr and "best.pt" in result.stderr


def test_source_hashes_and_allowlist():
    assert check()["status"] == "pass"
    files = [p.relative_to(ROOT).as_posix() for p in release_files()]
    assert not any(p.startswith((".venv/", "outputs/", "artifacts/")) for p in files)
    assert not any(p.endswith((".pt", ".mat", ".rd3", ".lnk")) for p in files)


def test_repository_rejects_unapproved_arrays(tmp_path):
    target = tmp_path / "configs/private.npz"
    target.parent.mkdir()
    target.write_bytes(b"test")
    (tmp_path / "RELEASE_FILES.txt").write_text("configs/private.npz\n")
    with pytest.raises(ValueError, match="Unapproved array"):
        release_files(tmp_path)


def test_documentation_links_and_yaml():
    for path in release_files():
        if path.suffix in {".yml", ".yaml", ".cff"}:
            assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" not in target and not target.startswith(("#", "mailto:")):
                    assert (path.parent / target.split("#")[0]).exists(), (path, target)
            for target in re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', path.read_text(encoding="utf-8")):
                if "://" not in target:
                    assert (path.parent / target).is_file(), (path, target)


def test_readme_visual_assets_and_provenance():
    provenance = json.loads((ROOT / "docs/assets/assets.json").read_text(encoding="utf-8"))
    assert len(provenance["figures"]) == 3
    for asset in [*provenance["figures"], provenance["logo"]]:
        path = ROOT / asset["asset"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == asset["asset_sha256"]
        assert content[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", content[16:24])
        assert width >= 2000 and height >= 600
        assert path in release_files()
    assert provenance["logo"]["scientific_data_used"] is False


def test_sample_recentering_preserves_mean_and_bounds():
    import torch
    from run_unified_neural_bvi_comparison import project_samples_to_mean
    draws = torch.rand(16, 1, 4, 4, generator=torch.Generator().manual_seed(7))
    mean = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4)
    result = project_samples_to_mean(draws, mean)
    assert result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(result.mean(0, keepdim=True), mean, atol=1e-7, rtol=1e-6)
