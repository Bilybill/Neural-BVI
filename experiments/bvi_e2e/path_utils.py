"""Path helpers for local data directories and Windows shortcut files."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Return the DeepwaveForGPR repository root from a script or cwd."""

    anchor = (start or Path(__file__)).resolve()
    for candidate in [anchor.parent, *anchor.parents]:
        if (candidate / "experiments").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


def _shortcut_target(path: Path) -> Path | None:
    if os.name != "nt" or path.suffix.lower() != ".lnk" or not path.exists():
        return None
    command = (
        "$ErrorActionPreference='Stop'; "
        "$path = (Resolve-Path -LiteralPath $env:SHORTCUT_PATH).Path; "
        "$shell = New-Object -ComObject WScript.Shell; "
        "$shortcut = $shell.CreateShortcut($path); "
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        "Write-Output $shortcut.TargetPath"
    )
    env = os.environ.copy()
    env["SHORTCUT_PATH"] = str(path)
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
            env=env,
        )
    except Exception:
        return None
    target = proc.stdout.strip()
    return Path(target) if proc.returncode == 0 and target else None


def _resolve_shortcut_components(path: Path) -> Path:
    resolved = path
    parts = path.parts
    for index, part in enumerate(parts):
        if not part.lower().endswith(".lnk"):
            continue
        shortcut = Path(*parts[: index + 1])
        target = _shortcut_target(shortcut)
        if target is None:
            continue
        remainder = Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path()
        resolved = target / remainder
        break
    return resolved


def resolve_data_path(path: Path, *fallbacks: str | Path, must_exist: bool = True) -> Path:
    """Resolve a data path, including repo-adjacent ``.lnk`` shortcuts."""

    root = repository_root()
    raw = Path(path).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([Path.cwd() / raw, root / raw, root.parent / raw])
    for fallback in fallbacks:
        fallback_path = Path(fallback)
        candidates.extend([root / fallback_path, root.parent / fallback_path])

    tried: list[str] = []
    for candidate in candidates:
        has_shortcut_component = any(part.lower().endswith(".lnk") for part in candidate.parts)
        expanded = _resolve_shortcut_components(candidate)
        tried.append(str(expanded))
        if (not has_shortcut_component or expanded != candidate) and expanded.exists():
            return expanded.resolve()
        tried.append(str(candidate))
        if not has_shortcut_component and candidate.exists():
            return candidate.resolve()

    if must_exist:
        raise FileNotFoundError(f"Could not resolve data path {path!s}. Tried: {tried}")
    return _resolve_shortcut_components(raw)
