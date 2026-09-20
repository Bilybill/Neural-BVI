import numpy as np
import pytest
import scipy.signal
import torch

from deepwave_physics_surrogate import torch_sosfiltfilt
from neural_bvi.demo import DemoForward, procedural_models, run_demo


def test_procedural_models_repeatable_and_bounded():
    first = procedural_models(3, 7)
    assert torch.equal(first, procedural_models(3, 7))
    assert not torch.equal(first, procedural_models(3, 8))
    assert first.shape == (3, 1, 24, 24)
    assert first.min() >= 0 and first.max() <= 1


def test_wave_data_gradient_matches_finite_difference():
    forward = DemoForward().double()
    model = procedural_models(1, 7).double().requires_grad_(True)
    weights = torch.randn(1, 1, 192, 8, dtype=torch.float64,
                          generator=torch.Generator().manual_seed(23))
    value = (forward(model) * weights).sum()
    gradient = torch.autograd.grad(value, model)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    direction = gradient / gradient.norm()
    step = 1e-4
    with torch.no_grad():
        finite = ((forward(model + step * direction) - forward(model - step * direction))
                  * weights).sum() / (2 * step)
    analytic = (gradient * direction).sum()
    torch.testing.assert_close(analytic, finite, rtol=5e-3, atol=1e-7)


@pytest.mark.parametrize("order", [2, 4])
def test_differentiable_filter_matches_scipy(order):
    sos = scipy.signal.butter(order, [0.1, 0.4], btype="bandpass", output="sos")
    data = torch.randn(80, 3, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(5), requires_grad=True)
    actual = torch_sosfiltfilt(data, sos)
    expected = scipy.signal.sosfiltfilt(sos, data.detach().numpy(), axis=0)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-11, rtol=1e-10)
    assert torch.autograd.grad(actual.square().sum(), data)[0].abs().sum() > 0


def test_demo_repeats_numerically_and_refuses_overwrite(tmp_path):
    first = run_demo(tmp_path / "a", epochs=2, bvi_steps=1)
    second = run_demo(tmp_path / "b", epochs=2, bvi_steps=1)
    assert first["training_loss"] == second["training_loss"]
    assert first["data_gradient_abs_sum"] == second["data_gradient_abs_sum"]
    assert first["data_gradient_abs_sum"] > 0
    with np.load(tmp_path / "a/posterior.npz") as a, np.load(tmp_path / "b/posterior.npz") as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
        assert np.all(a["std"] >= 0)
        assert np.all((a["event_probability"] >= 0) & (a["event_probability"] <= 1))
    with pytest.raises(FileExistsError):
        run_demo(tmp_path / "a", epochs=2, bvi_steps=1)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"bvi_steps": 0}])
def test_demo_rejects_invalid_budget(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_demo(tmp_path, **kwargs)
