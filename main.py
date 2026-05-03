"""
Computes & visualises the analytic Black-Scholes price of a European call option. Later,
the PINN result will be compared against this benchmark.
"""

from pathlib import Path

import numpy as np

from src.black_scholes import european_call_price
from src.plotting import plot_price_slices, plot_surface


def main():
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Parameters for a simple European call example.
    K = 40.0        # strike price (kötési árfolyam)
    r = 0.05        # risk-free interest rate (kockázatmentes kamatláb)
    sigma = 0.2     # volatility (volatilitás)
    T = 1.0         # maturity time (lejárati idő)
    S_max = 160.0   # maximum stock price considered (legnagyobb vizsgált
                    # részvényárfolyam)

    # Plot a few fixed-time slices first.
    # We start from S=1 instead of S=0 because the formula contains log(S/K).
    S_values = np.linspace(1.0, S_max, 300)

    slice_data = {}
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        prices = european_call_price(S_values, t, K=K, r=r, sigma=sigma, T=T)
        slice_data[f"t = {t}"] = prices

    plot_price_slices(
        S_values,
        slice_data,
        output_dir / "analytic_price_slices.png",
    )

    # Now compute the same formula on a full (t, S) grid for a surface plot.
    # TODO: rotate the surface plot to show the time axis better. The current view is
    # not ideal because the time axis is almost vertical and hard to read. Probably not
    # worth spending too much time on this.
    S_grid = np.linspace(1.0, S_max, 100)
    t_grid = np.linspace(0.0, T, 100)
    SS, TT = np.meshgrid(S_grid, t_grid)

    VV = european_call_price(SS, TT, K=K, r=r, sigma=sigma, T=T)

    plot_surface(
        SS,
        TT,
        VV,
        output_dir / "analytic_price_surface.png",
        "Analytic Black-Scholes European Call Price",
    )

    print("Done.")
    print("Generated figures:")
    print("- figures/analytic_price_slices.png")
    print("- figures/analytic_price_surface.png")


if __name__ == "__main__":
    main()
