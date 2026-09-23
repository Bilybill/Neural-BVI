"""Regressions for public packaging and content-bound experiment inputs."""
import json

import pytest
import torch

from check_repository import ROOT, release_files
from runtime_integrity import fingerprint, input_hashes
from run_deepwave_map_bvi_synthetic import build_error_pca_basis


def test_allowlist_excludes_unlisted_notes_and_ignored_outputs(tmp_path):
    (tmp_path / 'README.md').write_text('Public documentation')
    (tmp_path / 'RELEASE_FILES.txt').write_text('README.md\n')
    for name in ['docs/internal_notes.md', 'experiments/bvi_e2e/debug/summary.json',
                 'outputs/run.json']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Not for distribution')
    assert [p.relative_to(tmp_path).as_posix() for p in release_files(tmp_path)] == ['README.md']


@pytest.mark.parametrize('entry', ['../outside.md', '/outside.md', 'C:/outside.md'])
def test_allowlist_rejects_escape(tmp_path, entry):
    (tmp_path / 'RELEASE_FILES.txt').write_text(entry + '\n')
    with pytest.raises(ValueError, match='Invalid release path'):
        release_files(tmp_path)


def test_sdist_manifest_exactly_matches_public_list():
    entries = (ROOT / 'MANIFEST.in').read_text().splitlines()
    includes = {line.removeprefix('include ') for line in entries if line.startswith('include ')}
    assert includes == {p.relative_to(ROOT).as_posix() for p in release_files()}
    assert entries[0] == 'global-exclude *'
    assert not any('*' in line for line in entries[1:])


def test_changed_input_bytes_change_run_identity(tmp_path):
    names = ['protocol.snapshot.json', 'prepared/artifacts.pt', 'surrogate/best.pt',
             'train/unet/seed_7/best.pt']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    before = input_hashes(tmp_path, [7])
    for name in names:
        (tmp_path / name).write_bytes(b'changed')
        after = input_hashes(tmp_path, [7])
        assert before[name] != after[name]
        assert fingerprint(before) != fingerprint(after)
        (tmp_path / name).write_bytes(b'original')


def test_pca_cache_requires_matching_input_identity(tmp_path):
    path = tmp_path / 'basis.pt'
    meta = dict(kind='training_error_pca_basis_v2', protocol_hash='protocol', latent_dim=1,
                include_mean=True, train_count=2, snrs=[0.0], seed=7, input_fingerprint='original')
    torch.save({'basis': torch.ones(1, 2, 2), 'latent_raw_mean': torch.zeros(1),
                'latent_raw_std': torch.ones(1), 'meta': meta}, path)
    kwargs = dict(protocol={'protocol_hash': 'protocol'}, artifacts={'splits': {'train': []}},
                  model=torch.nn.Identity(), latent_dim=1, include_mean=True, train_count=2,
                  snrs=[0.0], seed=7, cache_path=path, device=torch.device('cpu'))
    assert build_error_pca_basis(**kwargs, input_fingerprint='original')['basis'].shape == (1, 2, 2)
    for identity in ['changed', None]:
        with pytest.raises(ValueError, match='without training indices'):
            build_error_pca_basis(**kwargs, input_fingerprint=identity)


def test_compact_configs_retain_runtime_fields():
    config = json.loads((ROOT / 'configs/paper/ensemble_center.json').read_text())
    assert config['global_weights'] == [0.2] * 5
    assert config['bias_scale'] == 0.0
    assert not {'all_candidates', 'source_metrics', 'test_metrics_read'} & config.keys()
