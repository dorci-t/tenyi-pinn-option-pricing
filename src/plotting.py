"""
Plotting utilities for the Black-Scholes benchmark.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_price_slices(S_values, slice_data, output_path):
    """
    Plot option price as a function of S for several fixed time values.
    """
    output_path = Path(output_path)

    plt.figure(figsize=(8, 5))

    for label, prices in slice_data.items():
        plt.plot(S_values, prices, label=label)

    plt.xlabel("Stock price S")
    plt.ylabel("Option price V(t, S)")
    plt.title("Analytic Black-Scholes European Call Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_surface(SS, TT, VV, output_path, title):
    """
    Plot a 3D surface.
    """
    output_path = Path(output_path)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(SS, TT, VV, linewidth=0, antialiased=True)

    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Option price V(t, S)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
