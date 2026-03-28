"""
3-D visualisation of Black-Scholes option prices and Greeks.

Renders option-price surfaces over (Spot, Time-to-maturity) and overlays
quiver arrows that make the derivative interpretation of each Greek concrete:

  * Delta arrows on the price surface   — tangent in the S direction  (∂C/∂S)
  * Theta arrows on the price surface   — tangent in the T direction  (∂C/∂t)
  * Gamma arrows on the delta surface   — curvature in S              (∂²C/∂S²)
  * Vega, Theta, Rho shown as standalone Greek surfaces

Run:
    python bs_visualisation.py          (saves PNGs to results/ and opens plots)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bs_put_call_pricer import (
    call_price, put_price,
    call_delta, put_delta,
    gamma, vega,
    call_theta, put_theta,
    call_rho, put_rho,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_surfaces(S_arr, T_arr, K, r, q, sigma):
    """Evaluate every pricing function on the (S, T) meshgrid."""
    Sg, Tg = np.meshgrid(S_arr, T_arr)
    return Sg, Tg, {
        "call":       call_price(Sg, K, Tg, r, q, sigma),
        "put":        put_price(Sg, K, Tg, r, q, sigma),
        "call_delta": call_delta(Sg, K, Tg, r, q, sigma),
        "put_delta":  put_delta(Sg, K, Tg, r, q, sigma),
        "gamma":      gamma(Sg, K, Tg, r, q, sigma),
        "vega":       vega(Sg, K, Tg, r, q, sigma),
        "call_theta": call_theta(Sg, K, Tg, r, q, sigma),
        "put_theta":  put_theta(Sg, K, Tg, r, q, sigma),
        "call_rho":   call_rho(Sg, K, Tg, r, q, sigma),
        "put_rho":    put_rho(Sg, K, Tg, r, q, sigma),
    }


def _add_surface(ax, Sg, Tg, Z, cmap=cm.viridis, alpha=0.70):
    ax.plot_surface(Sg, Tg, Z, cmap=cmap, alpha=alpha,
                    rstride=2, cstride=2, edgecolor="none")


def _style_ax(ax, xlabel="Spot  S", ylabel="Time  T", zlabel="Price"):
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=6)
    ax.set_zlabel(zlabel, fontsize=9, labelpad=6)
    ax.tick_params(labelsize=7)


# ── Figure 1 — Price surfaces with delta / theta arrows ─────────────────────

def _plot_price_panel(ax, Sg, Tg, price, Ssp, Tsp, price_sp, delta_sp,
                      theta_sp, title):
    """Price surface + quiver arrows for delta (red) and theta (blue)."""
    _add_surface(ax, Sg, Tg, price)

    # ---- delta arrows (tangent in S direction) ----
    # arrow from (S, T, C) in direction (dS, 0, delta*dS)
    dS = 4.0
    ax.quiver(
        Ssp, Tsp, price_sp,
        np.full_like(Ssp, dS), np.zeros_like(Tsp), delta_sp * dS,
        color="red", arrow_length_ratio=0.12, linewidth=1.0,
        normalize=False,
    )

    # ---- theta arrows (tangent in T direction) ----
    # arrow from (S, T, C) in direction (0, dT, theta*dT)
    dT = 0.18
    ax.quiver(
        Ssp, Tsp, price_sp,
        np.zeros_like(Ssp), np.full_like(Tsp, dT), theta_sp * dT,
        color="royalblue", arrow_length_ratio=0.12, linewidth=1.0,
        normalize=False,
    )

    _style_ax(ax)
    ax.set_title(title, fontsize=10, pad=10)

    # manual legend entries (quiver has no native legend)
    ax.plot([], [], "-", color="red",       lw=2, label="Δ  delta  (∂/∂S)")
    ax.plot([], [], "-", color="royalblue", lw=2, label="Θ  theta  (∂/∂t)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)


def figure_prices(dense, sparse, K, r, sigma):
    """Return a figure with call & put price surfaces + Greek arrows."""
    Sg, Tg, d = dense
    Ssp, Tsp, s = sparse

    fig = plt.figure(figsize=(17, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    _plot_price_panel(ax1, Sg, Tg, d["call"], Ssp, Tsp,
                      s["call"], s["call_delta"], s["call_theta"],
                      "Call Price  C(S, T)")
    ax1.view_init(elev=25, azim=-50)

    ax2 = fig.add_subplot(122, projection="3d")
    _plot_price_panel(ax2, Sg, Tg, d["put"], Ssp, Tsp,
                      s["put"], s["put_delta"], s["put_theta"],
                      "Put Price  P(S, T)")
    ax2.view_init(elev=25, azim=-130)

    fig.suptitle(
        f"Black-Scholes Price Surfaces   (K = {K},  r = {r},  σ = {sigma})\n"
        "Red arrows = Δ  (first derivative w.r.t. spot)        "
        "Blue arrows = Θ  (first derivative w.r.t. time)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


# ── Figure 2 — Greek surfaces with gamma arrows on delta ────────────────────

def figure_greeks(dense, sparse, K, r, sigma):
    """Return a 2×2 figure: delta+gamma arrows, gamma, theta, vega."""
    Sg, Tg, d = dense
    Ssp, Tsp, s = sparse

    fig = plt.figure(figsize=(17, 14))

    # ---- panel 1: call delta surface + gamma arrows ----
    ax1 = fig.add_subplot(221, projection="3d")
    _add_surface(ax1, Sg, Tg, d["call_delta"], cmap=cm.coolwarm)

    # gamma arrow: from (S, T, Δ) in direction (dS, 0, gamma*dS)
    dS = 4.0
    ax1.quiver(
        Ssp, Tsp, s["call_delta"],
        np.full_like(Ssp, dS), np.zeros_like(Tsp), s["gamma"] * dS,
        color="darkgreen", arrow_length_ratio=0.12, linewidth=1.0,
        normalize=False,
    )
    _style_ax(ax1, zlabel="Delta")
    ax1.set_title("Call Delta  Δ(S, T)\nGreen arrows = Γ  (∂Δ/∂S = ∂²C/∂S²)",
                  fontsize=10, pad=8)
    ax1.plot([], [], "-", color="darkgreen", lw=2, label="Γ  gamma (∂²/∂S²)")
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.8)
    ax1.view_init(elev=25, azim=-50)

    # ---- panel 2: gamma surface ----
    ax2 = fig.add_subplot(222, projection="3d")
    _add_surface(ax2, Sg, Tg, d["gamma"], cmap=cm.inferno)
    _style_ax(ax2, zlabel="Gamma")
    ax2.set_title("Gamma  Γ(S, T) = ∂²C/∂S²", fontsize=10, pad=8)
    ax2.view_init(elev=25, azim=-50)

    # ---- panel 3: vega surface ----
    ax3 = fig.add_subplot(223, projection="3d")
    _add_surface(ax3, Sg, Tg, d["vega"], cmap=cm.plasma)
    _style_ax(ax3, zlabel="Vega")
    ax3.set_title("Vega  ν(S, T) = ∂C/∂σ", fontsize=10, pad=8)
    ax3.view_init(elev=25, azim=-50)

    # ---- panel 4: call rho surface ----
    ax4 = fig.add_subplot(224, projection="3d")
    _add_surface(ax4, Sg, Tg, d["call_rho"], cmap=cm.RdYlBu)
    _style_ax(ax4, zlabel="Rho")
    ax4.set_title("Call Rho  ρ(S, T) = ∂C/∂r", fontsize=10, pad=8)
    ax4.view_init(elev=25, azim=-50)

    fig.suptitle(
        f"Black-Scholes Greeks   (K = {K},  r = {r},  σ = {sigma})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Runner ───────────────────────────────────────────────────────────────────

def run_and_save():
    K, r, q, sigma = 100.0, 0.05, 0.0, 0.20

    # Dense grid (smooth surface)
    S_dense = np.linspace(60, 140, 80)
    T_dense = np.linspace(0.05, 2.0, 80)
    Sg, Tg, d = _compute_surfaces(S_dense, T_dense, K, r, q, sigma)
    dense = (Sg, Tg, d)

    # Sparse grid (arrow origins — keep it uncluttered)
    S_sparse = np.linspace(70, 130, 7)
    T_sparse = np.linspace(0.15, 1.85, 6)
    Ssp, Tsp, s = _compute_surfaces(S_sparse, T_sparse, K, r, q, sigma)
    sparse = (Ssp, Tsp, s)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True)

    fig1 = figure_prices(dense, sparse, K, r, sigma)
    fig1.savefig(os.path.join(out, "price_surfaces.png"), dpi=200,
                 bbox_inches="tight")

    fig2 = figure_greeks(dense, sparse, K, r, sigma)
    fig2.savefig(os.path.join(out, "greek_surfaces.png"), dpi=200,
                 bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    run_and_save()
