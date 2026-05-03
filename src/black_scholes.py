"""
Analytic Black-Scholes formula for a European call option.
"""

import numpy as np
from scipy.stats import norm


def european_call_price(S, t, K=40.0, r=0.05, sigma=0.2, T=1.0):
    """
    Compute the Black-Scholes price of a European call option.

    Parameters
    ----------
    S : float or np.ndarray
        Current stock price.
    t : float or np.ndarray
        Current time.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Maturity time.

    Returns
    -------
    float or np.ndarray
        European call option price.
    """
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)

    tau = np.maximum(T - t, 1e-10)

    # Avoid log(0) numerically.
    S_safe = np.maximum(S, 1e-10)

    d1 = (np.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    price = S_safe * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    # At maturity, the option value is exactly the payoff.
    maturity_mask = np.isclose(t, T)
    payoff = np.maximum(S - K, 0.0)

    return np.where(maturity_mask, payoff, price)
