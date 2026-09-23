"""Procedural teaching example using the same residual BVI kernel as the paper.

The small grid, network and noise model are illustrative, not the paper benchmark.
Time is measured in microseconds and velocity in meters per microsecond.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import deepwave
import numpy as np
import torch
from torch import nn

from . import run_bvi


class DemoForward(nn.Module):
    """Differentiable zero-offset scalar propagation on a 24 x 24 grid."""

    def __init__(self):
        super().__init__()
        locations = torch.zeros(8, 1, 2, dtype=torch.long)
        locations[:, 0, 0] = 2
        locations[:, 0, 1] = torch.arange(3, 19, 2)
        self.register_buffer("locations", locations)
        pulse = deepwave.wavelets.ricker(480.0, 192, 0.00006, 0.002)
        self.register_buffer("pulse", pulse.reshape(1, 1, -1).repeat(8, 1, 1))

    def forward(self, models):
        outputs = []
        for model in models:
            velocity = 299.792458 / torch.sqrt(2.0 + 8.0 * model[0])
            wave = deepwave.scalar(
                velocity.contiguous(), 0.03, 0.00006,
                source_amplitudes=self.pulse,
                source_locations=self.locations,
                receiver_locations=self.locations,
                pml_width=8, pml_freq=480.0, max_vel=212.0, accuracy=2,
            )[-1]
            # Constant amplitude scale preserves a well-defined gradient.
            outputs.append(wave[:, 0, :].T / 20.0)
        return torch.stack(outputs).unsqueeze(1)


def procedural_models(count, seed):
    generator = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(24), torch.arange(24), indexing="ij")
    models = []
    for _ in range(count):
        cx = int(torch.randint(6, 18, (), generator=generator))
        cy = int(torch.randint(8, 17, (), generator=generator))
        radius = int(torch.randint(2, 5, (), generator=generator))
        background = 0.20 + 0.1 * torch.rand((), generator=generator)
        model = torch.full((24, 24), float(background))
        model[yy > 17] += 0.15
        model[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 0.80
        models.append(model)
    return torch.stack(models).unsqueeze(1)


def run_demo(output: Path, seed=7, epochs=20, bvi_steps=4, device="cpu"):
    if epochs < 1 or bvi_steps < 1:
        raise ValueError("epochs and bvi-steps must be positive")
    if (output / "summary.json").exists():
        raise FileExistsError(f"{output} already has a run; choose a new --output directory")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    started = time.perf_counter()
    target_device = torch.device(device)
    forward = DemoForward().to(target_device)
    # The last model is held out from neural training.
    models = procedural_models(13, seed).to(target_device)
    with torch.no_grad():
        clean = forward(models)
        noise_std = max(float(clean.square().mean().sqrt()) * 0.1, 1e-4)
        observations = clean + noise_std * torch.randn_like(clean)
    network = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((12, 8)),
        nn.Flatten(), nn.Linear(8 * 12 * 8, 24 * 24), nn.Sigmoid(),
        nn.Unflatten(1, (1, 24, 24)),
    ).to(target_device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = (network(observations[:12]) - models[:12]).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    network.eval()
    with torch.no_grad():
        center = network(observations[12:]).detach()
    probe = center.clone().requires_grad_(True)
    probe_loss = (forward(probe) - observations[12:]).square().mean()
    gradient = torch.autograd.grad(probe_loss, probe)[0]
    if not torch.isfinite(gradient).all() or float(gradient.abs().sum()) == 0:
        raise RuntimeError("The data term has no finite, nonzero model gradient")
    result, mean, std, event, samples = run_bvi(
        forward=forward, observation=observations[12:], true_model=models[12:],
        nn_mean=center, latent_dim=4, components=2, steps=bvi_steps,
        samples_per_component=2, residual_scale=0.04, noise_std=noise_std,
        kl_weight=0.02, model_prior_std=0.03, interrogation_threshold=0.5,
        basis_type="cosine", device=target_device, posterior_sample_count=32,
        return_samples=True, mixture_line_search_samples=2,
    )
    if not torch.isfinite(samples).all():
        raise RuntimeError("Non-finite posterior samples")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"truth": models[12:].detach().cpu().numpy(),
              "observation": observations[12:].detach().cpu().numpy(),
              "center": center.detach().cpu().numpy(),
              "mean": mean.detach().cpu().numpy(), "std": std.detach().cpu().numpy(),
              "event_probability": event.detach().cpu().numpy()}
    np.savez_compressed(output / "posterior.npz", **arrays)
    torch.save(network.state_dict(), output / "inverse_weights.pt")
    report = {"status": "complete", "scope": "procedural_demo_not_paper_benchmark",
              "seed": seed, "device": device, "train_models": 12, "test_models": 1,
              "epochs": epochs, "bvi_steps": bvi_steps, "training_loss": losses,
              "data_gradient_abs_sum": float(gradient.abs().sum()),
              "elapsed_seconds": time.perf_counter() - started, "metrics": asdict(result)}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.7), constrained_layout=True)
    for ax, key, title in zip(axes, ("truth", "center", "mean", "std"),
                            ("Truth", "Neural estimate", "BVI mean", "BVI std. (normalized)")):
        values = arrays[key].squeeze()
        handle = ax.imshow(values if key == "std" else 2 + 8 * values,
                           cmap="viridis" if key == "std" else "cividis",
                           vmin=0 if key == "std" else 2,
                           vmax=None if key == "std" else 10)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(handle, ax=ax, fraction=0.046)
    fig.savefig(output / "reconstruction.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"status": "complete", "output": str(output),
                      "seconds": report["elapsed_seconds"]}, indent=2))
    return report
