"""Unit tests for the operator networks."""

import pytest
import torch

from thermano.config import ModelConfig
from thermano.models import InputScaler, build_model


def _model(kind, n_sensors=6, out="softplus"):
    cfg = ModelConfig(type=kind, branch_layers=(16,), trunk_layers=(16, 16), latent_dim=8, output_transform=out)
    return build_model(cfg, n_sensors, ([0.0] * n_sensors, [1.0] * n_sensors), ([0, 0, 0, 0], [1, 2, 1, 1]))


@pytest.mark.parametrize("kind", ["deeponet", "en_deeponet", "pinn"])
def test_output_shape_and_positivity(kind):
    m = _model(kind)
    out = m(torch.rand(32, 6), torch.rand(32, 4))
    assert out.shape == (32, 1)
    assert torch.all(out >= 0)  # softplus output transform


def test_identity_output_can_be_negative():
    m = _model("deeponet", out="identity")
    with torch.no_grad():
        m.bias.fill_(-100.0)
    assert (m(torch.rand(8, 6), torch.rand(8, 4)) < 0).all()


def test_scaler_maps_bounds_to_unit_interval():
    s = InputScaler([0.0, 10.0], [2.0, 30.0])
    out = s(torch.tensor([[0.0, 10.0], [2.0, 30.0], [1.0, 20.0]]))
    assert torch.allclose(out, torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]))


def test_scaler_is_saved_with_weights(tmp_path):
    m = _model("en_deeponet")
    torch.save(m.state_dict(), tmp_path / "w.pt")
    sd = torch.load(tmp_path / "w.pt")
    assert "trunk_scaler.scale" in sd and "branch_scaler.bias" in sd


def test_branch_input_changes_output():
    """The operator must actually depend on the input function (not collapse to a PINN)."""
    m = _model("en_deeponet")
    y = torch.rand(16, 4)
    a, b = m(torch.zeros(16, 6), y), m(torch.ones(16, 6), y)
    assert not torch.allclose(a, b)


def test_second_derivatives_exist():
    """PDE loss needs d2u/dx2: smooth activations must give a non-zero Hessian."""
    m = _model("en_deeponet")
    y = torch.rand(10, 4, requires_grad=True)
    u = m(torch.rand(10, 6), y)
    g = torch.autograd.grad(u.sum(), y, create_graph=True)[0][:, 1]
    h = torch.autograd.grad(g.sum(), y)[0][:, 1]
    assert torch.isfinite(h).all() and h.abs().sum() > 0
