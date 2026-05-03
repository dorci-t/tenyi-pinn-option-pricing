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

    def __init__(self, hidden_dim=64, hidden_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, t, S):
        x = torch.cat([t, S], dim=1)
        return self.net(x)
