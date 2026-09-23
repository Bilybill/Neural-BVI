"""Prepare synthetic inputs and train the five-network Neural-BVI ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from publication_data import load_protocol, prepare_protocol_artifacts, resolve_protocol_path
from publication_training import train_inversion


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=here / 'la010010_protocol.json')
    parser.add_argument('--root', type=Path, default=here.parents[1] / 'artifacts/paper')
    parser.add_argument('--stage', choices=['prepare', 'train'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, help='Train one ensemble member only')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = load_protocol(args.protocol.resolve())
    artifact_path = root / 'prepared/artifacts.pt'
    snapshot = root / 'protocol.snapshot.json'
    if args.stage == 'prepare':
        if root.exists() and any(root.iterdir()):
            parser.error('Preparation requires an empty output root')
        artifacts = prepare_protocol_artifacts(protocol, smoke=args.smoke)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts, artifact_path)
        snapshot.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
        print(f'Prepared inputs: {artifact_path}')
        return

    if not artifact_path.is_file() or not snapshot.is_file():
        parser.error('Run --stage prepare first, or provide a matching prepared bundle')
    saved_protocol = json.loads(snapshot.read_text(encoding='utf-8'))
    if saved_protocol['protocol_hash'] != protocol['protocol_hash']:
        parser.error('Protocol differs from the prepared snapshot')
    artifacts = torch.load(artifact_path, map_location='cpu', weights_only=False)
    if artifacts['protocol_hash'] != saved_protocol['protocol_hash']:
        parser.error('Prepared inputs do not match the protocol')
    profile_path = resolve_protocol_path(args.protocol.resolve(), protocol['acquisition_profile'])
    profiles = json.loads(profile_path.read_text(encoding='utf-8'))
    profile = profiles[artifacts['dataset_manifest_profile']]
    physics_path = root / 'surrogate/best.pt'
    if not physics_path.exists():
        physics_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'architecture': 'DeepwavePhysicsSurrogate', 'surrogate_kind': 'deepwave_physics',
                    'acquisition_profile': profile, 'physics_config': {}}, physics_path)
    seeds = [args.seed] if args.seed is not None else protocol['ensemble_seeds']
    if args.smoke:
        seeds = seeds[:1]
    for seed in seeds:
        output = root / 'train/unet' / f'seed_{seed}'
        if output.exists() and any(output.iterdir()):
            parser.error(f'Training output already exists: {output}; use a new root')
    for seed in seeds:
        train_inversion(protocol, artifacts, 'unet', int(seed), root / 'train/unet' / f'seed_{seed}',
                        'mixed', smoke=args.smoke)


if __name__ == '__main__':
    main()
