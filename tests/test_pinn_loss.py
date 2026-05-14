import math

import torch

from src.losses import compute_pinn_loss
from src.pinn_model import PINN, GatedPINN, ParametricPINN


def _assert_finite_positive(components):
    for key, value in components.items():
        assert math.isfinite(value), f"{key} is not finite: {value}"
        assert value >= 0, f"{key} is negative: {value}"


def test_pinn_loss_and_backward():
    torch.manual_seed(0)
    model = PINN(hidden_dim=32, hidden_layers=2)

    loss, components = compute_pinn_loss(
        model, n_interior=200, n_terminal=100, n_boundary=100,
    )

    _assert_finite_positive(components)
    loss.backward()


def test_gated_pinn_loss_and_backward():
    torch.manual_seed(0)
    model = GatedPINN(hidden_dim=32, hidden_layers=2)

    loss, components = compute_pinn_loss(
        model, n_interior=200, n_terminal=100, n_boundary=100,
    )

    _assert_finite_positive(components)
    loss.backward()


def test_parametric_pinn_loss_and_backward():
    torch.manual_seed(0)
    model = ParametricPINN(
        hidden_dim=32, hidden_layers=2, T=1.0, S_max=160.0, K_scale=160.0,
    )

    loss, components = compute_pinn_loss(
        model, K_min=20.0, K_max=120.0,
        n_interior=200, n_terminal=100, n_boundary=100,
    )

    _assert_finite_positive(components)
    loss.backward()
