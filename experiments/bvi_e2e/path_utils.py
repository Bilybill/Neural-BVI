"""Portable paths for user-supplied data."""
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    anchor = (start or Path(__file__)).resolve()
    for candidate in anchor.parents:
        if (candidate / 'experiments').is_dir() and (candidate / 'pyproject.toml').is_file():
            return candidate
    return Path.cwd().resolve()


def resolve_data_path(path: Path, *fallbacks: str | Path, must_exist: bool = True) -> Path:
    raw = Path(path).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, repository_root() / raw]
    candidates.extend(repository_root() / Path(value) for value in fallbacks)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if must_exist:
        raise FileNotFoundError(f'Data path does not exist: {path}. Supply an explicit local path.')
    return candidates[0].resolve()
