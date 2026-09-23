"""Content identities for reusable experiment inputs."""
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(root: Path, seeds: list[int]) -> dict[str, str]:
    paths = ['protocol.snapshot.json', 'prepared/artifacts.pt', 'surrogate/best.pt']
    paths += [f'train/unet/seed_{seed}/best.pt' for seed in sorted(set(seeds))]
    return {name: file_sha256(root / name) for name in paths}


def fingerprint(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
