"""
Quick smoke test for the PINN building blocks.

This script checks whether the model, sampling functions, PDE residual and
loss calculation work together before implementing the full training loop.
"""

import torch

from src.losses import compute_pinn_loss
from src.pinn_model import PINN


def main():
    torch.manual_seed(0)

    # Using a smaller model and fewer points here so the check runs quickly.
    model = PINN(hidden_dim=32, hidden_layers=2)

    loss, components = compute_pinn_loss(
        model,
        n_interior=200,
        n_terminal=100,
        n_boundary=100,
    )

    print("One PINN loss evaluation on an untrained model:")
    print(f"total loss:    {components['total']:.6f}")
    print(f"PDE loss:      {components['pde']:.6f}")
    print(f"terminal loss: {components['terminal']:.6f}")
    print(f"boundary loss: {components['boundary']:.6f}")

    loss.backward()
    print("Backward pass completed successfully.")


if __name__ == "__main__":
    main()