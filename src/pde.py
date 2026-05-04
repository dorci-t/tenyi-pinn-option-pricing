"""
Black-Scholes PDE residual used in the PINN loss.

For the main benchmark, the neural network approximates the option price
function V(t, S). This module computes how much the network output violates
the Black-Scholes PDE at sampled interior points.
"""

import torch


def black_scholes_residual(model, t, S, r=0.05, sigma=0.2):
    """
    Compute the Black-Scholes PDE residual for the model output V(t, S).
    """
    V = model(t, S)

    V_t = torch.autograd.grad(
        V,
        t,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
    )[0]

    V_S = torch.autograd.grad(
        V,
        S,
        grad_outputs=torch.ones_like(V),
        create_graph=True,
    )[0]

    V_SS = torch.autograd.grad(
        V_S,
        S,
        grad_outputs=torch.ones_like(V_S),
        create_graph=True,
    )[0]

    residual = V_t + 0.5 * sigma**2 * S**2 * V_SS + r * S * V_S - r * V

    return residual