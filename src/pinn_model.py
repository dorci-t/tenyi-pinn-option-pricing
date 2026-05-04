"""
Initial PINN model skeleton.

This file is not complete yet. The next project step is to implement the PDE residual
and the training loop.
"""

import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Simple fully connected neural network.

    Input:
        t, S

    Output:
        V(t, S)
    """

    def __init__(self, hidden_dim=64, hidden_layers=4, T=1.0, S_max=160.0):
        super().__init__()

        self.T = T
        self.S_max = S_max

        layers = [
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        ]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t, S):
        t_scaled = t / self.T
        S_scaled = S / self.S_max

        x = torch.cat([t_scaled, S_scaled], dim=1)
        return self.net(x)