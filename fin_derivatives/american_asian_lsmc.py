"""
Longstaff-Schwartz Monte Carlo (LSMC) pricing of American-style Asian options.

Prices a Bermudan Asian call option whose payoff at each exercise date is
max(running_average - K, 0). Two volatility specifications are compared:
  1. Constant (GBM) volatility with exact log-normal steps.
  2. Local volatility surface (calibrated via Andreasen-Huge) with a weekly
     Euler-Maruyama discretization.

The LSMC backward induction regresses discounted future cashflows on
polynomial basis functions of (S_t, A_t) to estimate the continuation value,
then compares against the immediate exercise payoff at each date.
"""

##################
# Imports
##################
from typing import Callable, Sequence
import os

import numpy as np
import pandas as pd
import random
np.random.seed(18)

##################
# Helpers
##################

def payoff_fn(path_avg: np.ndarray, strike: float) -> np.ndarray:
    """Asian call payoff: max(running_average - K, 0)."""
    return np.maximum(path_avg - strike, 0.0)

##################
# LocalVolSurface
##################

class LocalVolSurface:
    """
    Piecewise-constant local volatility surface sigma_tilde(S, T).

    Loaded from a CSV containing calibrated (T, K_abs, sigma_tilde) triplets
    produced by the Andreasen-Huge algorithm. Between maturity slices the
    surface is linearly interpolated in time; within each slice, sigma_tilde
    is linearly interpolated in spot price.

    The local percentage volatility returned by `sigma(S, t)` is
    sigma_tilde(S, t) / S, consistent with the dS = sigma_tilde(S,t) dW
    parameterization.
    """

    def __init__(self, file):
        self.K, self.T, self.sigma_tilde_by_T = self._load(file)

        self.sigma_tilde_by_T = {
            float(T): np.asarray(st, dtype=float)
            for T, st in self.sigma_tilde_by_T.items()
        }

        self.K_min = self.K[0]
        self.K_max = self.K[-1]

    def _load(self, csv_path: str = "local_vol_calibration.csv"):
        """
        Load local volatility surface data from CSV file.
        Calibration obtained in assignment 3. CSV is expected to contain
        columns T, K_abs, sigma_tilde.
        """
        df = pd.read_csv(csv_path)
        T_grid = np.sort(df["T"].unique())
        K_abs = np.sort(df["K_abs"].unique())
        sigma_tilde_by_T = {}
        for T in T_grid:
            sub = df[df["T"] == T].sort_values("K_abs")
            assert np.allclose(sub["K_abs"].values, K_abs)
            sigma_tilde_by_T[float(T)] = sub["sigma_tilde"].values

        return np.asarray(K_abs), np.asarray(T_grid), sigma_tilde_by_T

    def _sigma_at_T(self, S, T):
        """Interpolate sigma_tilde at a single maturity slice and return local % vol."""
        S = np.asarray(S, dtype=float)
        S_clipped = np.clip(S, self.K_min, self.K_max)

        sigma_tilde = self.sigma_tilde_by_T[T]
        st_interp = np.interp(S_clipped, self.K, sigma_tilde)
        return st_interp / S_clipped  # local % vol

    def sigma(self, S, t):
        """Return the local percentage volatility at spot level(s) S and time t, interpolating between maturity slices."""
        S = np.asarray(S, dtype=float)
        t = float(t)

        if t <= self.T[0]:
            return self._sigma_at_T(S, self.T[0])
        if t >= self.T[-1]:
            return self._sigma_at_T(S, self.T[-1])

        idx_hi = np.searchsorted(self.T, t)
        T_hi = self.T[idx_hi]
        T_lo = self.T[idx_hi - 1]

        sigma_lo = self._sigma_at_T(S, T_lo)
        sigma_hi = self._sigma_at_T(S, T_hi)

        w = (t - T_lo) / (T_hi - T_lo)
        return (1.0 - w) * sigma_lo + w * sigma_hi

##################
# Path generators
##################

def generate_constant_gbm_paths(
    n_paths: int,
    timeline: Sequence[float] | np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
    vol: float = 0.23,
    s0: float = 100.0,
) -> np.ndarray:
    """
    Simulate GBM paths with constant volatility using exact log-normal steps.

    Parameters:
        n_paths:  number of Monte Carlo paths.
        timeline: array of observation times (prepends t=0 if missing).
        r:        risk-free rate.
        q:        continuous dividend yield.
        vol:      constant Black-Scholes volatility.
        s0:       initial spot price.

    Returns:
        Array of shape (n_paths, len(timeline)) with spot prices at each
        observation date (excluding t=0).
    """
    timeline = np.asarray(timeline, dtype=float)

    if timeline[0] != 0.0:
        timeline = np.insert(timeline, 0, 0.0)

    t_steps = len(timeline)
    paths = np.empty((n_paths, t_steps), dtype=float)
    paths[:, 0] = s0

    for i in range(t_steps - 1):
        dt = timeline[i + 1] - timeline[i]

        z = np.random.normal(size=n_paths)
        St = paths[:, i]
        v = vol

        drift = (r - q - 0.5 * v**2) * dt
        diffusion = v * np.sqrt(dt) * z

        paths[:, i + 1] = St * np.exp(drift + diffusion)

    return paths[:,1:]


def generate_local_vol_paths_weekly(
    n_paths: int,
    T: float,
    r: float,
    q: float,
    s0: float,
    vol_grid: LocalVolSurface,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weekly Euler scheme for the local volatility model:
        dS_t = S_t * v(S_t, t) * dW_t
    Euler:
        S_{t+Δt} = S_t + (r-q) S_t Δt + v(S_t, t) S_t √Δt z

    Returns:
        paths_weekly: shape (n_paths, n_steps+1)
        timeline_weekly: shape (n_steps+1,)
    """
    n_steps = 52
    dt = T / n_steps
    timeline_weekly = np.linspace(0.0, T, n_steps + 1)

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0

    for k in range(n_steps):
        t = timeline_weekly[k]
        St = paths[:, k]
        z = np.random.normal(size=n_paths)

        # Local volatility per path
        v = vol_grid.sigma(St, t)

        drift = (r - q) * St * dt
        diffusion = v * St * np.sqrt(dt) * z
        paths[:, k + 1] = St + drift + diffusion

    return paths, timeline_weekly


##################
# LSMC algorithm
##################

def lsmc_price(
    paths: np.ndarray,
    timeline: Sequence[float] | np.ndarray,
    payoff: Callable[[np.ndarray, float], np.ndarray],
    poly_degree: int,
    r: float,
    K: float,
) -> tuple:
    """
    Longstaff–Schwartz Monte Carlo pricing for an American-style
    option where the state variables are (S_t, running average A_t).

    timeline is the grid of exercise dates only: [0, t1, t2, t3, t4].
    """
    timeline = np.asarray(timeline, dtype=float)
    n_paths, T = paths.shape

    assert len(timeline) == T, "timeline length must match number of columns in paths"

    # Cashflows matrix
    cashflows = np.zeros_like(paths)

    # Process A
    A = np.cumsum(paths, axis = 1) / np.arange(1, T+1)
    cashflows[:, -1] = payoff(A[:,-1], K)

    # Backward induction (exercise only at t1..t4)
    for t in range(T - 2, 0, -1):
        # Running average up to time index t
        A_t = A[:,t]
        S_t = paths[:,t]

        # Intrinsic value
        intrinsic = payoff(A_t, K)

        itm = intrinsic > 0.0
        if not np.any(itm):
            continue

        # Discount future cashflows from times t+1..T-1 back to time t
        dt_from_t = np.array(timeline[t + 1:]) - timeline[t]  # shape (T-1 - t,)
        disc_factors = np.exp(-r * dt_from_t)

        # Realized continuation value under current exercise policy
        Y = np.sum(
            cashflows[itm, t + 1:] * disc_factors,
            axis=1,
        )

        # Basis functions: 1, S, S^2, S^3, A, A^2, A^3
        S_itm = S_t[itm]
        A_itm = A_t[itm]

        S_feat = np.column_stack([S_itm**k for k in range(1, poly_degree + 1)])
        A_feat = np.column_stack([A_itm**k for k in range(1, poly_degree + 1)])
        ones = np.ones((S_itm.shape[0], 1))

        X = np.column_stack([ones, S_feat, A_feat])

        # Regression: estimate continuation value E[Y | S_t, A_t]
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        cont_itm = X @ beta

        continuation = np.zeros_like(intrinsic)
        continuation[itm] = cont_itm

        # Exercise decision: exercise if immediate > continuation
        exercise_now = (intrinsic > continuation) & itm

        cashflows[exercise_now, t] = intrinsic[exercise_now]
        cashflows[exercise_now, t + 1:] = 0.0  # kill future cashflows

    # Discount all remaining cashflows at t>0 back to t=0
    dt_from_0 = np.exp(-r * np.array(timeline))
    disc0 = np.exp(-r * dt_from_0)
    PV_path = cashflows @ disc0            # shape (n_paths,)
    option_price = np.mean(PV_path)
    std_error = np.std(PV_path, ddof=1) / np.sqrt(n_paths)

    return option_price, std_error

##################
# Simulation functions
##################

def simulation_constant_vol(
    n_sim: int,
    n_paths: int,
    timeline_ex: np.ndarray,
    r: float,
    q: float,
    vol: float,
    poly_degree: int,
    s0: float,
    K: float,
):
    """
    Run n_sim independent LSMC valuations under constant GBM volatility and
    return the grand mean price, its standard error, and the number of runs.
    """
    prices = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        paths = generate_constant_gbm_paths(
            n_paths=n_paths,
            timeline=timeline_ex,
            r=r,
            q=q,
            vol=vol,
            s0=s0,
        )

        prices[i], _ = lsmc_price(
            paths=paths,
            timeline=timeline_ex,
            payoff=payoff_fn,
            poly_degree=poly_degree,
            r=r,
            K=K,
        )

    mean_price = prices.mean()
    std_error = prices.std(ddof=1) / np.sqrt(n_sim)

    return mean_price, std_error, n_sim


def simulation_local_vol(
    n_sim: int,
    n_paths: int,
    timeline_ex: np.ndarray,
    T: float,
    r: float,
    q: float,
    poly_degree: int,
    s0: float,
    K: float,
    vol_grid: LocalVolSurface,
):
    """
    Local-vol simulation with weekly Euler, exercise only at timeline_ex.
    """
    prices = np.empty(n_sim, dtype=float)

    for i in range(n_sim):
        # weekly local-vol paths
        paths_weekly, timeline_weekly = generate_local_vol_paths_weekly(
            n_paths=n_paths,
            T=T,
            r=r,
            q=q,
            s0=s0,
            vol_grid=vol_grid,
        )

        # sample at exercise dates: 0.25, 0.5, 0.75, 1
        exercise_indices = np.searchsorted(timeline_weekly, timeline_ex)
        paths_ex = paths_weekly[:, exercise_indices]

        prices[i], _ = lsmc_price(
            paths=paths_ex,
            timeline=timeline_ex,
            payoff=payoff_fn,
            poly_degree=poly_degree,
            r=r,
            K=K,
        )

    mean_price = prices.mean()
    std_error = prices.std(ddof=1) / np.sqrt(n_sim)

    return mean_price, std_error, n_sim


##################
# Main
##################

def main() -> None:
    """
    Price an American Asian call (quarterly exercise, T=1yr, K=98) under
    constant vol (23%) and local vol (Andreasen-Huge calibration), printing
    LSMC mean prices and standard errors for both.
    """
    # Nb times we run LSMC
    n_sim = 100

    # Simulated paths per LSMC
    n_paths = 1000

    # Exercise timeline: t1, t2, t3, t4
    timeline_ex = np.array([0.25, 0.5, 0.75, 1.0])

    # Model params
    s0 = 100.0
    r = 0.0
    q = 0.02
    vol = 0.23

    # Payoff params
    K = 98.0
    poly_degree = 3

    # Part 1 - constant vol
    const_mean, const_std, const_runs = simulation_constant_vol(
        n_sim=n_sim,
        n_paths=n_paths,
        timeline_ex=timeline_ex,
        r=r,
        q=q,
        vol=vol,
        poly_degree=poly_degree,
        s0=s0,
        K=K,
    )

    # Load local-vol grid (calibrated via Andreasen-Huge)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vol_grid = LocalVolSurface(os.path.join(script_dir, "data", "local_vol_calibration.csv"))

    # Part 2 - local vol with weekly Euler
    local_mean, local_std, local_runs = simulation_local_vol(
        n_sim=n_sim,
        n_paths=n_paths,
        timeline_ex=timeline_ex,
        T=1.0,
        r=r,
        q=q,
        poly_degree=poly_degree,
        s0=s0,
        K=K,
        vol_grid=vol_grid,
    )

    print("Part 1 : constant vol @ 23%")
    print("-" * 30)
    print("Mean price      :", const_mean)
    print("Std error       :", const_std)
    print("Number of runs  :", const_runs)
    print("\n" + "=" * 30)

    print("Part 2 : local vol (weekly Euler)")
    print("-" * 30)
    print("Mean price      :", local_mean)
    print("Std error       :", local_std)
    print("Number of runs  :", local_runs)


if __name__ == "__main__":
    main()
