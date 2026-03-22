"""
Black-Scholes European put and call option pricer.

Prices European vanilla options using the closed-form Black-Scholes formulae
and reports key first-order Greeks (delta, gamma, vega, theta, rho).
"""

import numpy as np
import scipy.stats as stats


def _d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, q, sigma):
    return _d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, q, sigma):
    """Black-Scholes European call price."""
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def put_price(S, K, T, r, q, sigma):
    """Black-Scholes European put price."""
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1)


def call_delta(S, K, T, r, q, sigma):
    return np.exp(-q * T) * stats.norm.cdf(_d1(S, K, T, r, q, sigma))


def put_delta(S, K, T, r, q, sigma):
    return -np.exp(-q * T) * stats.norm.cdf(-_d1(S, K, T, r, q, sigma))


def gamma(S, K, T, r, q, sigma):
    """Gamma is identical for calls and puts."""
    d1 = _d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, q, sigma):
    """Vega is identical for calls and puts (per 1-unit change in sigma)."""
    d1 = _d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)


def call_theta(S, K, T, r, q, sigma):
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)
    term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = q * S * np.exp(-q * T) * stats.norm.cdf(d1)
    term3 = -r * K * np.exp(-r * T) * stats.norm.cdf(d2)
    return term1 - term2 - term3


def put_theta(S, K, T, r, q, sigma):
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)
    term1 = -S * np.exp(-q * T) * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    term2 = -q * S * np.exp(-q * T) * stats.norm.cdf(-d1)
    term3 = r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
    return term1 + term2 + term3


def call_rho(S, K, T, r, q, sigma):
    d2 = _d2(S, K, T, r, q, sigma)
    return K * T * np.exp(-r * T) * stats.norm.cdf(d2)


def put_rho(S, K, T, r, q, sigma):
    d2 = _d2(S, K, T, r, q, sigma)
    return -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)


if __name__ == "__main__":
    # Example: ATM option on a non-dividend-paying stock
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20

    c = call_price(S, K, T, r, q, sigma)
    p = put_price(S, K, T, r, q, sigma)

    print(f"{'Call price':<20s} {c:.6f}")
    print(f"{'Put price':<20s} {p:.6f}")
    print(f"{'Put-call parity chk':<20s} {c - p:.6f}  (should ≈ {S * np.exp(-q * T) - K * np.exp(-r * T):.6f})")
    print()
    print(f"{'Call delta':<20s} {call_delta(S, K, T, r, q, sigma):.6f}")
    print(f"{'Put delta':<20s} {put_delta(S, K, T, r, q, sigma):.6f}")
    print(f"{'Gamma':<20s} {gamma(S, K, T, r, q, sigma):.6f}")
    print(f"{'Vega':<20s} {vega(S, K, T, r, q, sigma):.6f}")
    print(f"{'Call theta':<20s} {call_theta(S, K, T, r, q, sigma):.6f}")
    print(f"{'Put theta':<20s} {put_theta(S, K, T, r, q, sigma):.6f}")
    print(f"{'Call rho':<20s} {call_rho(S, K, T, r, q, sigma):.6f}")
    print(f"{'Put rho':<20s} {put_rho(S, K, T, r, q, sigma):.6f}")
