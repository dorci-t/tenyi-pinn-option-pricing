"""
First training script for the Black-Scholes PINN.

This is intentionally kept simple: it trains the current PINN model for a
small number of epochs and saves the loss curve. The goal is to check that
the model can actually be optimised before adding more experiments.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import trange

from src.losses import compute_pinn_loss
from src.pinn_model import PINN


def plot_loss_curve(loss_history, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("PINN training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    torch.manual_seed(0)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    device = "cpu"

    model = PINN(
        hidden_dim=32,
        hidden_layers=2,
        T=1.0,
        S_max=160.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 1000
    loss_history = []

    for epoch in trange(n_epochs):
        optimizer.zero_grad()

        loss, components = compute_pinn_loss(
            model,
            n_interior=500,
            n_terminal=200,
            n_boundary=200,
            beta=1.0,
            device=device,
        )

        loss.backward()
        optimizer.step()

        loss_history.append(components["total"])

        if epoch % 100 == 0:
            print(
                f"epoch {epoch:4d} | "
                f"total={components['total']:.4f} | "
                f"pde={components['pde']:.4f} | "
                f"terminal={components['terminal']:.4f} | "
                f"boundary={components['boundary']:.4f}"
            )

    plot_loss_curve(
        loss_history,
        output_dir / "pinn_training_loss.png",
    )

    torch.save(model.state_dict(), "pinn_model.pt")

    print("Training finished.")
    print("Saved:")
    print("- figures/pinn_training_loss.png")
    print("- pinn_model.pt")


if __name__ == "__main__":
    main()