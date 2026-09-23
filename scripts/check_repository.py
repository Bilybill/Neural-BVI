"""Validate the explicit public distribution manifest and Python syntax."""
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = {"configs/paper/center_bias.npz", "configs/paper/std_prior.npz"}


def release_files(root=ROOT):
    root = root.resolve()
    names = (root / "RELEASE_FILES.txt").read_text(encoding="utf-8").splitlines()
    names = [name.strip() for name in names if name.strip() and not name.startswith("#")]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate entry in RELEASE_FILES.txt")
    files = []
    for name in names:
        rel = Path(name)
        if rel.is_absolute() or name.startswith("/") or ".." in rel.parts or "\\" in name or ":" in name:
            raise ValueError(f"Invalid release path: {name}")
        path = root / rel
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Release path escapes repository: {name}")
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root):
            raise ValueError(f"Symlink in release: {name}")
        if not path.is_file():
            raise ValueError(f"Missing release file: {name}")
        if rel.parts[0] in {"outputs", "artifacts", "provenance", ".git", ".venv", "dist"}:
            raise ValueError(f"Private/generated directory in manifest: {name}")
        if rel.parts[0] == "data" and name != "data/README.md":
            raise ValueError(f"Raw data are not allowed in this release: {name}")
        if path.suffix == ".npz" and name not in ARRAYS:
            raise ValueError(f"Unapproved array: {name}")
        if path.suffix in {".pt", ".pth", ".ckpt", ".mat", ".rd3", ".rad", ".lnk", ".pem", ".key"}:
            raise ValueError(f"Unapproved file type: {name}")
        files.append(path)
    return sorted(files)


def check(root=ROOT):
    root = root.resolve()
    files = release_files(root)
    if (root / ".git").exists():
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        unexpected = {name for name in tracked if name and (root / name).is_file()} - {
            path.relative_to(root).as_posix() for path in files}
        if unexpected:
            raise ValueError(f"Tracked files outside public manifest: {sorted(unexpected)}")
    for path in files:
        if path.suffix in {".npz", ".png"}:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        if path.suffix == ".json":
            json.loads(text)
        if re.search(r"[A-Za-z]:[\\/](?:Users|Project)[\\/]", text):
            raise ValueError(f"Machine-specific absolute path: {path.relative_to(root)}")
        if path.suffix in {".md", ".json", ".csv"} and re.search(
            r"neural_bvi_optimization|Phase-[1-5]|20260721|source_manifest\.json", text
        ):
            raise ValueError(f"Development-session metadata: {path.relative_to(root)}")
        # This is a narrow local scan, not a substitute for hosted secret scanning.
        if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", text):
            raise ValueError(f"Private key: {path.relative_to(root)}")
    return {"status": "pass", "release_files": len(files),
            "bytes": sum(path.stat().st_size for path in files)}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
