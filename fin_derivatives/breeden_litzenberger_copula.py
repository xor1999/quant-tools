"""
Risk-neutral density extraction and exotic option pricing via Gaussian copula Monte Carlo.

Extracts the risk-neutral probability density of terminal stock prices from
observed implied volatility smiles using the Breeden-Litzenberger (1978)
result: phi(K) = d^2 C(K) / dK^2 for undiscounted call prices.

Implied volatilities are first smoothed in total-variance space (d, w) via
cubic splines, then undiscounted calls and their second derivatives are
computed to recover the risk-neutral PDF and CDF for SPX and AMZN.

Terminal stock prices are then sampled through a Gaussian copula (with
correlation rho = 0.5) by inverting the marginal CDFs, enabling Monte Carlo
pricing of multi-asset exotic payoffs:
  1. Exchange option:  (S_SPX / S0_SPX  -  S_AMZN / S0_AMZN)^+
  2. Power put:        (K - S_AMZN^2)^+
"""

import numpy as np
import seaborn as sns
import os
from scipy import stats
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt


# MATHEMATICAL FUNCTIONS ----------------------------------------------------------------------------

def forward_price(S0: float, T: float, r: float, q: float) -> float:
    """Compute the forward price F = S0 * exp((r - q) * T)."""
    return float(S0) * np.exp((float(r) - float(q)) * float(T))


def smooth_iv_pairs_via_dw(pairs: list[tuple[float, float]], S0: float, T: float, r: float, q: float) -> list[tuple[float, float]]:
    """
    Steps 1-4: transform (K, sigma) -> (d=ln(K/F), w=sigma^2*T), fit UnivariateSpline(d->w),
    evaluate on dense d_grid, then map back to (K_grid, sigma_grid).

    Tuning via env vars:
      - DW_SMOOTH: smoothing factor multiplier for s = DW_SMOOTH * n * Var(w). Use 0 for no smoothing.
      - DW_NGRID: number of grid points for d_grid (default 400).
    """
    # Filter valid data
    clean = [(float(K), float(sig)) for (K, sig) in pairs if (K is not None and sig is not None and np.isfinite(K) and np.isfinite(sig) and K > 0 and sig >= 0)]
    if len(clean) < 3 or T <= 0:
        return clean

    K_arr = np.array([k for k, _ in clean], dtype=float)
    sig_arr = np.array([s for _, s in clean], dtype=float)
    F = forward_price(S0, T, r, q)
    d = np.log(K_arr / F)
    w = (sig_arr ** 2) * T

    # Sort by d
    order = np.argsort(d)
    d = d[order]
    w = w[order]

    # Spline smoothing
    try:
        s_factor = float(os.environ.get("DW_SMOOTH", "1e-6"))
    except Exception:
        s_factor = 1e-6
    n = len(d)
    varw = float(np.var(w)) if n > 1 else 0.0
    s = (s_factor * n * varw) if varw > 0 else 0.0

    spline = interpolate.UnivariateSpline(d, w, s=s, k=3)

    # Dense grid in d
    try:
        n_grid = int(os.environ.get("DW_NGRID", "400"))
    except Exception:
        n_grid = 400
    d_grid = np.linspace(d.min(), d.max(), max(n_grid, 50))
    w_smooth = spline(d_grid)
    # Clip small negatives due to numerical issues
    w_smooth = np.maximum(w_smooth, 0.0)

    # Back-transform
    K_grid = F * np.exp(d_grid)
    iv_grid = np.sqrt(w_smooth / T)

    return list(zip(K_grid.tolist(), iv_grid.tolist()))
def compute_undiscounted_call_price(S0: float, T: float, r:float, K: float, div:float, imp_vol: float):
    """
    Computes the value of an undiscounted call price.

    Parameters:
        S0: initial stock price,
        T : maturity date (T-t)
        r : interest rate (constant)
        K : strike price
        div : dividends if any
        imp_vol : implied volatility
    
    Returns:
        call_price : price of an undiscounted call
    """

    d1 = (np.log(S0/K) + (r - div + 0.5 * imp_vol **2)*T)/(imp_vol * np.sqrt(T))
    d2 = d1 - (imp_vol * np.sqrt(T))

    first_term = S0 * np.exp((r-div)*T) * stats.norm.cdf(d1)
    second_term = K * stats.norm.cdf(d2)

    call_price = first_term - second_term

    # Return high-precision value (no rounding) to avoid amplifying noise in BL second differences
    return float(call_price)

def breeden_litzenberger(strike_prices: np.ndarray, call_prices: np.ndarray) -> np.ndarray:
    """
    Breeden–Litzenberger density from UNDISCOUNTED call prices C(K): f(K) = d^2C/dK^2.

    Implementation: smoothing spline on (K, C) for stability on non-uniform, noisy grids;
    pdf = max(spline''(K), 0). Returns an array same length as K.

    Notes:
    - Still BL: we estimate the second derivative of C(K) w.r.t. K.
    - A small smoothing parameter reduces high-frequency noise that causes jagged densities.
    """
    from scipy.interpolate import UnivariateSpline

    K = np.asarray(strike_prices, float)
    C = np.asarray(call_prices, float)
    # Ensure strictly increasing strikes
    order = np.argsort(K)
    K = K[order]
    C = C[order]
    assert np.all(np.diff(K) > 0), "Strikes must be strictly increasing."

    # Heuristic smoothing: scale with variance and number of points
    # Tunable via env var BL_SMOOTH (factor multiplying n*Var(C)); set BL_SMOOTH=0 for no smoothing.
    n = len(K)
    varC = float(np.var(C)) if n > 1 else 0.0
    try:
        s_factor = float(os.environ.get("BL_SMOOTH", "1e-6"))  # even lighter default
    except Exception:
        s_factor = 1e-6
    s = (s_factor * n * varC) if varC > 0 else 0.0

    spline = UnivariateSpline(K, C, s=s, k=3)
    d2 = spline.derivative(n=2)(K)
    pdf = np.maximum(d2, 0.0)

    # For safety, map back to original ordering (already sorted, but keep API stable)
    return pdf

def pdf_sanitizer(K: np.ndarray, pdf_dirty: np.ndarray):
    """
    Replace NaNs by 0, clip negatives to 0, renormalize ∫ pdf dK = 1 via trapezoid.
    """
    K = np.asarray(K, float)
    f = np.where(np.isnan(pdf_dirty), 0.0, np.asarray(pdf_dirty, float))
    f = np.maximum(f, 0.0)
    mass = np.trapezoid(f, K)  # more common spelling than np.trapezoid
    if mass > 0:
        f /= mass
    # sanity checks
    assert np.all(f >= -1e-12), "PDF has negatives after sanitize."
    return f

def build_option_grid(ticker:str, pairs: list[tuple[float, float]], S0: float, T: float, r: float, q: float) -> pd.DataFrame:
    """
    Build a DataFrame with strikes, implied vols, undiscounted call prices,
    and the Breeden-Litzenberger risk-neutral density for a given ticker.

    Parameters:
        ticker: identifier label (e.g. "SPX", "AMZN").
        pairs:  list of (strike, implied_vol) tuples.
        S0:     spot price.
        T:      time to maturity in years.
        r:      risk-free rate.
        q:      continuous dividend yield.

    Returns:
        DataFrame with columns: K, implied_vol, und_calls, phi_ST, ticker.
    """
    df = pd.DataFrame(pairs, columns=["K", "implied_vol"])
    df.sort_values("K", inplace=True)

    # Compute undiscounted calls prices
    df["und_calls"] = [compute_undiscounted_call_price(S0, T, r, K, q, sigma) for (K, sigma) in df[["K", "implied_vol"]].to_numpy()]

    # Compute distribution of stock prices at maturity
    df["phi_ST"] = breeden_litzenberger(df['K'], df['und_calls'])

    # Add a ticker for simplicity and indexing
    df["ticker"] = ticker

    return df

def pdf_to_cdf(strikes, pdf):
    """
    Left-to-right cumulative trapezoid; enforces monotonicity and renormalizes to 1.
    """
    K = np.asarray(strikes, float)
    f = np.asarray(pdf, float)
    cdf = np.zeros_like(f)
    cdf[1:] = np.cumsum(0.5*(f[1:]+f[:-1])*(K[1:]-K[:-1]))
    # enforce [0,1] monotone
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    if cdf[-1] > 0:
        cdf /= cdf[-1]
    # sanity checks
    assert np.all(np.diff(cdf[~np.isnan(cdf)]) >= -1e-12), "CDF not monotone."
    return cdf

# PLOTTING FUNCTIONS ----------------------------------------------------------------------------

def plot_pdf(df: pd.DataFrame, ticker: str, folder: str = "results"):
    """
    Plots the risk-neutral PDF φ_{S_T}(K) for the specified ticker.
    Expects columns: 'ticker', 'K', 'pdf'.
    """
    sub = (
        df.loc[df["ticker"] == ticker, ["K", "pdf"]]
          .dropna()
          .sort_values("K")
    )

    g = sns.relplot(
        data=sub,
        kind="line",
        x="K",
        y="pdf",
        height=5,
        aspect=1.8
    )
    g.set_axis_labels("Strike K", r"$\phi_{S_T}(K)$")
    g.set_titles(f"{ticker}: Risk-neutral density (Breeden–Litzenberger)")

    # Add grid to the (single) facet
    for ax in g.axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(folder, exist_ok=True)
    g.figure.savefig(os.path.join(folder, f"density_{ticker}.png"), dpi=300, bbox_inches="tight")
    plt.close(g.figure)

def plot_cdf(K: np.ndarray, cdf: np.ndarray, ticker: str, folder: str = "results"):
    """
    Plot the CDF Φ_{S_T}(K).
    Provide the K (x-axis) and cdf (y-axis) arrays of equal length.
    """
    K = np.asarray(K)
    cdf = np.asarray(cdf)

    # Sort by K to avoid zig-zag lines if input is unsorted
    order = np.argsort(K)
    K = K[order]
    cdf = cdf[order]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(x=K, y=cdf, ax=ax)
    ax.set_xlabel("Strike K")
    ax.set_ylabel(r"$\Phi_{S_T}(K)$")
    ax.set_title(f"{ticker}: CDF of S_T")
    ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f"CDF_{ticker}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_calls(df: pd.DataFrame, ticker: str, folder: str = "results"):
    """
    Plots undiscounted call prices C(K) vs strike for a given ticker.
    Expects columns: 'ticker','K','und_calls'.
    """
    sub = (
        df.loc[df["ticker"] == ticker, ["K", "und_calls"]]
          .dropna()
          .sort_values("K")
    )

    g = sns.relplot(
        data=sub,
        kind="line",
        x="K",
        y="und_calls",
        height=4,
        aspect=1.8
    )
    g.set_axis_labels("Strike K", "Undiscounted call C(K)")
    g.set_titles(f"{ticker}: Undiscounted call vs strike")
    for ax in g.axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(folder, exist_ok=True)
    g.figure.savefig(os.path.join(folder, f"undiscounted_calls_{ticker}.png"), dpi=300, bbox_inches="tight")
    plt.close(g.figure)

def plot_payoff_distribution(payoffs: np.ndarray, title : str, results_name : str, folder: str = "results"):
    """
    Plots the empirical distribution (histogram as density) of simulated payoffs.
    """
    payoffs = np.asarray(payoffs).ravel()
    g = sns.displot(
        pd.DataFrame({"payoff": payoffs}),
        x="payoff",
        bins=50,
        stat="density",
        height=4,
        aspect=1.8
    )
    g.set_axis_labels("undiscounted payoff", "density")
    g.set_titles(title)
    # Add grid on the single facet
    for ax in g.axes.flatten():
        ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs(folder, exist_ok=True)
    g.figure.savefig(os.path.join(folder, results_name), dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def main():
    """
    End-to-end pipeline: load SPX/AMZN implied vols, extract risk-neutral
    densities via Breeden-Litzenberger, build a Gaussian copula for the
    joint distribution, and Monte Carlo price two exotic payoffs.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # load_data
    folder = os.path.join(script_dir, "data", "Impvols_SPX_AMZN.xlsx")
    data = pd.read_excel(io=folder, skiprows= lambda x: x in [0, 3], engine = "calamine")
    data = data.drop(columns = [x for x in data.columns if "Unnamed" in x])
    
    # recover the data
    spx_pairs = [
        (K, sigma)
        for K, sigma in data[['spx_strikes', 'spx implied vols']].itertuples(index=False, name=None)
        if not (pd.isna(K) or pd.isna(sigma))
    ]

    amzn_pairs = [
        (K, sigma)
        for K, sigma in data[['amzn_strikes', 'amzn implied vols']].itertuples(index=False, name=None)
        if not (pd.isna(K) or pd.isna(sigma))
    ] 
    
    # generate unconditional call prices
    T = 105/365
    r = 0.048

    # individual parameters
    spx_S0 = 6600
    amzn_S0 = 231
    spx_div = 0.013
    amzn_div = 0.0

    # Smooth IVs in (d,w) space then back to (K, sigma)
    spx_pairs_smooth = smooth_iv_pairs_via_dw(spx_pairs, spx_S0, T, r, spx_div)
    amzn_pairs_smooth = smooth_iv_pairs_via_dw(amzn_pairs, amzn_S0, T, r, amzn_div)

    # generate table compiling all info using smoothed pairs and concatenate
    spx_df = build_option_grid("SPX", spx_pairs_smooth, spx_S0, T, r, spx_div)
    spx_df['pdf'] = pdf_sanitizer(spx_df['K'], spx_df['phi_ST'])

    amzn_df = build_option_grid("AMZN", amzn_pairs_smooth, amzn_S0, T, r, amzn_div)
    amzn_df['pdf'] = pdf_sanitizer(amzn_df['K'], amzn_df['phi_ST'])
    all_df = pd.concat([spx_df, amzn_df], ignore_index=True)

    # save results
    spx_df.to_csv(os.path.join(results_dir, "spx_grid_with_phi.csv"), index=False)
    amzn_df.to_csv(os.path.join(results_dir, "amzn_grid_with_phi.csv"), index=False)

    # free some space
    del data

    # generate cdf
    spx_cdf = pdf_to_cdf(all_df[all_df['ticker'] == 'SPX']['K'], all_df[all_df['ticker'] == 'SPX']['pdf'])
    amzn_cdf = pdf_to_cdf(all_df[all_df['ticker'] == 'AMZN']['K'], all_df[all_df['ticker'] == 'AMZN']['pdf'])

    # generate a multivariate random variable ~ N ( [0, 0], [[1, rho], [rho, 1]])
    mean = np.array([0,0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    rng = np.random.default_rng(42)
    Xs = rng.multivariate_normal(mean, cov, size=10_000)

    # map each couple of X to a uniform
    Us = stats.norm.cdf(Xs)

    # map uniforms to S_T via inverse CDFs
    inv_spx = interpolate.interp1d(spx_cdf, spx_df['K'].to_numpy(),
                               kind="linear", bounds_error=False,
                               fill_value=(spx_df['K'].min(), spx_df['K'].max()))
    
    inv_amzn = interpolate.interp1d(amzn_cdf, amzn_df['K'].to_numpy(),
                                    kind="linear", bounds_error=False,
                                    fill_value=(amzn_df['K'].min(), amzn_df['K'].max()))

    terminal_stock_prices = np.column_stack([
        inv_spx(Us[:, 0]),
        inv_amzn(Us[:, 1]),
    ]).astype(float)

    # exchange payoff: (S_SPX_T/SPX_0 - S_AMZN_T/AMZN_0)^+
    diff = terminal_stock_prices[:, 0] / spx_S0 - terminal_stock_prices[:, 1] / amzn_S0
    payoffs1 = np.maximum(diff, 0.0)

    # discount and estimate price + MC standard error
    disc = np.exp(-r * T)
    price1 = disc * payoffs1.mean()
    se1 = disc * payoffs1.std(ddof=1) / np.sqrt(payoffs1.size)

    
    print("Exchange option (SPX/AMZN):")
    print(f"  MC price: {price1:.6f} | SE: {se1:.6f}")

    Xs = stats.norm.rvs(loc = 0, scale = 1, size = 10000, random_state = rng)
    Us = stats.norm.cdf(Xs)
    inv_amzn2 = interpolate.interp1d(amzn_cdf, amzn_df['K'].to_numpy(),
                                    kind="linear", bounds_error=False,
                                    fill_value=(amzn_df['K'].min(), amzn_df['K'].max()))
    terminal_amzn_prices = inv_amzn2(Us).astype(float)
    K = 218**2
    diff = K - terminal_amzn_prices**2
    payoffs2 = np.maximum(diff, 0.0)
    price2 = disc * payoffs2.mean()
    se2 = disc * payoffs2.std(ddof=1) / np.sqrt(payoffs2.size)

    print("Power put on AMZN^2:")
    print(f"  MC price: {price2:.6f} | SE: {se2:.6f}")

    # quick plots
    sns.set_theme(style="whitegrid")
    plot_pdf(df=spx_df, ticker="SPX", folder=results_dir)
    plot_pdf(df=amzn_df, ticker="AMZN", folder=results_dir)
    plot_cdf(K=spx_df['K'], cdf=spx_cdf, ticker="SPX", folder=results_dir)
    plot_cdf(K=amzn_df['K'], cdf=amzn_cdf, ticker="AMZN", folder=results_dir)
    plot_calls(df=spx_df, ticker="SPX", folder=results_dir)
    plot_calls(df=amzn_df, ticker="AMZN", folder=results_dir)
    plot_payoff_distribution(
        payoffs1,
        results_name="payoff_distribution_ex1.png",
        title=r"Distribution of $(S_{\mathrm{SPX}}/S_{0,\mathrm{SPX}} - S_{\mathrm{AMZN}}/S_{0,\mathrm{AMZN}})^+$ at $T$",
        folder=results_dir,
    )
    plot_payoff_distribution(
        payoffs2,
        results_name="payoff_distribution_ex2.png",
        title=r"Distribution of $(K - S_{\mathrm{AMZN}}^2)^+$ at $T$",
        folder=results_dir,
    )

    return 0

if __name__ == "__main__":
    SystemExit(main())