"""
Monte Carlo engine for option pricing under Geometric Brownian Motion.

Simulates GBM paths via the exact log-normal solution:
    S(t) = S(0) * exp((r - q - 0.5*sigma^2)*t + sigma*W(t))

Includes:
- European pricing with naive MC and antithetic variates
- American pricing via Longstaff-Schwartz Least Squares Monte Carlo (LSMC)
"""

import numpy as np


def simulate_gbm_paths(S, T, r, q, sigma, n_steps, n_paths, rng=None):
    """Simulate GBM sample paths using exact log-normal discretisation.

    Parameters
    ----------
    S : float
        Initial spot price.
    T : float
        Time to maturity (years).
    r : float
        Risk-free rate (continuous).
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility (annualised).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of Monte Carlo paths.
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    paths : ndarray, shape (n_paths, n_steps + 1)
        Simulated price paths, paths[:, 0] == S.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    # draw increments: (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z

    log_increments = drift + diffusion
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    paths = S * np.exp(log_paths)
    return t, paths


def mc_european_price(S, K, T, r, q, sigma, n_paths, option_type="call",
                      antithetic=False, rng=None):
    """Price a European option via Monte Carlo (single-step terminal sampling).

    Parameters
    ----------
    S, K, T, r, q, sigma : float
        Standard Black-Scholes parameters.
    n_paths : int
        Number of Monte Carlo paths.
    option_type : str
        'call' or 'put'.
    antithetic : bool
        If True, use antithetic variates for variance reduction.
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.

    Returns
    -------
    price : float
        Discounted expected payoff.
    std_err : float
        Standard error of the price estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    Z = rng.standard_normal(n_paths)

    nuT = (r - q - 0.5 * sigma**2) * T
    sigmaT = sigma * np.sqrt(T)

    if antithetic:
        ST_pos = S * np.exp(nuT + sigmaT * Z)
        ST_neg = S * np.exp(nuT - sigmaT * Z)
        if option_type == "call":
            payoff = 0.5 * (np.maximum(ST_pos - K, 0.0)
                            + np.maximum(ST_neg - K, 0.0))
        else:
            payoff = 0.5 * (np.maximum(K - ST_pos, 0.0)
                            + np.maximum(K - ST_neg, 0.0))
    else:
        ST = S * np.exp(nuT + sigmaT * Z)
        if option_type == "call":
            payoff = np.maximum(ST - K, 0.0)
        else:
            payoff = np.maximum(K - ST, 0.0)

    discounted = np.exp(-r * T) * payoff
    price = float(np.mean(discounted))
    std_err = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, std_err


def _basis_polynomial(S, degree=2):
    """Simple polynomial basis: 1, S, S^2, ..., S^degree."""
    return np.column_stack([S**k for k in range(degree + 1)])


def _basis_laguerre(S, degree=2):
    """Weighted Laguerre polynomial basis (Longstaff-Schwartz original).

    Uses L_0(x)=1, L_1(x)=1-x, L_2(x)=1-2x+x^2/2 with x = S/S[0] scaling.
    """
    x = S / np.mean(S)  # normalise for numerical stability
    L0 = np.ones_like(x)
    L1 = 1.0 - x
    L2 = 1.0 - 2.0 * x + 0.5 * x**2
    basis = [L0, L1, L2]
    if degree >= 3:
        L3 = 1.0 - 3.0 * x + 1.5 * x**2 - x**3 / 6.0
        basis.append(L3)
    return np.column_stack(basis[: degree + 1])


def lsmc_american_price(S, K, T, r, q, sigma, n_paths, n_steps=50,
                        option_type="put", basis="polynomial", degree=2,
                        rng=None):
    """Price an American option via Longstaff-Schwartz LSMC.

    Parameters
    ----------
    S, K, T, r, q, sigma : float
        Standard Black-Scholes parameters.
    n_paths : int
        Number of Monte Carlo paths.
    n_steps : int
        Number of exercise dates (time steps).
    option_type : str
        'call' or 'put'.
    basis : str
        'polynomial' for simple 1, S, S^2, ... basis;
        'laguerre' for weighted Laguerre polynomials.
    degree : int
        Degree of the basis (number of non-constant terms).
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.

    Returns
    -------
    price : float
        American option price estimate.
    std_err : float
        Standard error of the price estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    # simulate paths: (n_paths, n_steps + 1)
    _, paths = simulate_gbm_paths(S, T, r, q, sigma, n_steps, n_paths, rng=rng)

    dt = T / n_steps
    df = np.exp(-r * dt)  # one-step discount factor

    # intrinsic value at each node
    if option_type == "call":
        intrinsic = np.maximum(paths - K, 0.0)
    else:
        intrinsic = np.maximum(K - paths, 0.0)

    # cashflow matrix: when and how much each path receives
    cashflow = intrinsic[:, -1].copy()  # initialise with terminal payoff
    exercise_time = np.full(n_paths, n_steps)  # step index of exercise

    # select basis function builder
    basis_fn = _basis_laguerre if basis == "laguerre" else _basis_polynomial

    # backward induction
    for t in range(n_steps - 1, 0, -1):
        itm = intrinsic[:, t] > 0  # only regress on in-the-money paths
        if itm.sum() < degree + 2:
            # too few ITM paths to regress — just discount and continue
            cashflow *= df
            continue

        # discounted future cashflow for ITM paths
        # discount from exercise_time to step t
        steps_ahead = exercise_time[itm] - t
        disc_cf = cashflow[itm] * np.exp(-r * dt * steps_ahead)

        # build basis and regress
        X = basis_fn(paths[itm, t], degree=degree)
        coeffs, _, _, _ = np.linalg.lstsq(X, disc_cf, rcond=None)
        continuation = X @ coeffs

        # exercise decision: exercise now if intrinsic > continuation estimate
        exercise_now = intrinsic[itm, t] > continuation
        idx = np.where(itm)[0][exercise_now]
        cashflow[idx] = intrinsic[idx, t]
        exercise_time[idx] = t

    # discount all cashflows back to t=0
    pv = cashflow * np.exp(-r * dt * exercise_time)
    price = float(np.mean(pv))
    std_err = float(np.std(pv, ddof=1) / np.sqrt(n_paths))
    return price, std_err


if __name__ == "__main__":
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    n_paths = 100_000

    rng = np.random.default_rng(42)
    price_naive, se_naive = mc_european_price(S, K, T, r, q, sigma, n_paths,
                                              option_type="call", rng=rng)
    rng = np.random.default_rng(42)
    price_anti, se_anti = mc_european_price(S, K, T, r, q, sigma, n_paths,
                                            option_type="call", antithetic=True,
                                            rng=rng)

    print(f"{'Naive MC call':<25s} {price_naive:.4f}  (SE {se_naive:.4f})")
    print(f"{'Antithetic MC call':<25s} {price_anti:.4f}  (SE {se_anti:.4f})")

    # American put via LSMC
    rng = np.random.default_rng(42)
    am_poly, se_poly = lsmc_american_price(S, K, T, r, q, sigma, n_paths,
                                           option_type="put", basis="polynomial",
                                           rng=rng)
    rng = np.random.default_rng(42)
    am_lag, se_lag = lsmc_american_price(S, K, T, r, q, sigma, n_paths,
                                         option_type="put", basis="laguerre",
                                         rng=rng)
    print(f"\n{'LSMC put (poly)':<25s} {am_poly:.4f}  (SE {se_poly:.4f})")
    print(f"{'LSMC put (laguerre)':<25s} {am_lag:.4f}  (SE {se_lag:.4f})")
