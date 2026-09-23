from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from e2e_bvi_gpr import run_bvi
from inversion_models import MODEL_REGISTRY, build_inversion_model
from publication_data import load_protocol, make_split, synthesize_noise_view


class DummyForward(nn.Module):
    def forward(self, model: torch.Tensor) -> torch.Tensor:
        return F.interpolate(model, size=(16, 12), mode="bilinear", align_corners=False)


class PublicationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(HERE / "la010010_protocol.json")

    def test_split_is_fixed_and_disjoint(self) -> None:
        split_a = make_split(1000, self.protocol["split"])
        split_b = make_split(1000, self.protocol["split"])
        self.assertEqual(split_a, split_b)
        self.assertEqual({key: len(value) for key, value in split_a.items()}, {"train": 700, "validation": 150, "test": 150})
        all_indices = split_a["train"] + split_a["validation"] + split_a["test"]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_noise_snr_and_reproducibility(self) -> None:
        clean = torch.linspace(-1.0, 1.0, 1 * 32 * 24).reshape(1, 32, 24)
        residuals = torch.stack([torch.randn(1, 32, 24, generator=torch.Generator().manual_seed(i)) for i in range(4)])
        view_a, meta_a = synthesize_noise_view(clean, residuals, 1234, 5.0, "mixed", 0.7, 0.3)
        view_b, meta_b = synthesize_noise_view(clean, residuals, 1234, 5.0, "mixed", 0.7, 0.3)
        self.assertTrue(torch.equal(view_a, view_b))
        self.assertEqual(meta_a, meta_b)
        empirical = 20.0 * np.log10(float(torch.sqrt(torch.mean(clean**2))) / float(torch.sqrt(torch.mean((view_a - clean) ** 2))))
        self.assertAlmostEqual(empirical, 5.0, places=5)

    def test_model_registry_shapes(self) -> None:
        observation = torch.randn(1, 1, 179, 137)
        for name in MODEL_REGISTRY:
            with self.subTest(name=name):
                output = build_inversion_model(name)(observation)
                self.assertEqual(tuple(output.shape), (1, 1, 256, 256))
                self.assertTrue(bool(torch.isfinite(output).all()))

    def test_bvi_posterior_is_finite(self) -> None:
        device = torch.device("cpu")
        base = torch.full((1, 1, 16, 16), 0.4)
        observation = DummyForward()(base)
        result = run_bvi(
            forward=DummyForward(),
            observation=observation,
            true_model=base,
            nn_mean=base,
            latent_dim=4,
            components=1,
            steps=1,
            samples_per_component=1,
            residual_scale=0.02,
            noise_std=0.05,
            kl_weight=0.02,
            model_prior_std=0.03,
            interrogation_threshold=0.5,
            basis_type="cosine",
            device=device,
            posterior_sample_count=16,
            return_samples=True,
        )
        samples = result[-1]
        self.assertEqual(tuple(samples.shape), (16, 1, 16, 16))
        self.assertTrue(bool(torch.isfinite(samples).all()))

if __name__ == "__main__":
    unittest.main()
