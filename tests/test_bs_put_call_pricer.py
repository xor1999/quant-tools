"""
Pytest suite for the Black-Scholes put/call pricer and Greeks.

Tests are organised into five groups:
  1. Pricing — known analytical values and put-call parity.
  2. Greeks — numerical finite-difference checks against each closed-form Greek.
  3. Boundary / limiting cases — deep ITM/OTM, short expiry.
  4. Put-call symmetry relations — delta, theta, rho parity identities.
  5. Vectorised (numpy array) inputs.
"""

import sys, os
import numpy as np
import pytest

# Ensure the package root is on the path so the import works from any cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fin_derivatives.bs_put_call_pricer import (
    call_price, put_price,
    call_delta, put_delta,
    gamma, vega,
    call_theta, put_theta,
    call_rho, put_rho,
)

# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture()
def atm():
    """ATM option, no dividends."""
    return dict(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20)


@pytest.fixture()
def otm_call():
    """OTM call (ITM put), with dividend yield."""
    return dict(S=100.0, K=120.0, T=0.5, r=0.03, q=0.02, sigma=0.25)


@pytest.fixture()
def itm_call():
    """ITM call (OTM put), no dividends."""
    return dict(S=100.0, K=80.0, T=0.25, r=0.05, q=0.0, sigma=0.30)


# ── 1. Pricing tests ────────────────────────────────────────────────────────

class TestPricing:
    """Verify absolute price levels and put-call parity."""

    def test_call_price_atm(self, atm):
        c = call_price(**atm)
        # Well-known ATM BS value ≈ 10.4506
        assert c == pytest.approx(10.4506, abs=0.001)

    def test_put_price_atm(self, atm):
        p = put_price(**atm)
        assert p == pytest.approx(5.5735, abs=0.001)

    def test_put_call_parity(self, atm):
        """C - P = S*exp(-qT) - K*exp(-rT)."""
        c = call_price(**atm)
        p = put_price(**atm)
        forward_diff = atm["S"] * np.exp(-atm["q"] * atm["T"]) - atm["K"] * np.exp(-atm["r"] * atm["T"])
        assert c - p == pytest.approx(forward_diff, abs=1e-10)

    def test_put_call_parity_with_dividends(self, otm_call):
        c = call_price(**otm_call)
        p = put_price(**otm_call)
        forward_diff = otm_call["S"] * np.exp(-otm_call["q"] * otm_call["T"]) - otm_call["K"] * np.exp(-otm_call["r"] * otm_call["T"])
        assert c - p == pytest.approx(forward_diff, abs=1e-10)

    def test_call_positive(self, atm):
        assert call_price(**atm) > 0

    def test_put_positive(self, atm):
        assert put_price(**atm) > 0


# ── 2. Greeks — finite-difference verification ──────────────────────────────

def _fd_greek(func, params, bump_key, bump=1e-5):
    """Central finite-difference approximation of d(func)/d(bump_key)."""
    up = dict(params)
    dn = dict(params)
    up[bump_key] = params[bump_key] + bump
    dn[bump_key] = params[bump_key] - bump
    return (func(**up) - func(**dn)) / (2 * bump)


class TestGreeksFiniteDiff:
    """Each analytic Greek should match its finite-difference counterpart."""

    def test_call_delta(self, atm):
        fd = _fd_greek(call_price, atm, "S")
        assert call_delta(**atm) == pytest.approx(fd, rel=1e-5)

    def test_put_delta(self, atm):
        fd = _fd_greek(put_price, atm, "S")
        assert put_delta(**atm) == pytest.approx(fd, rel=1e-5)

    def test_gamma_from_call(self, atm):
        fd = _fd_greek(call_delta, atm, "S")
        assert gamma(**atm) == pytest.approx(fd, rel=1e-4)

    def test_gamma_from_put(self, atm):
        fd = _fd_greek(put_delta, atm, "S")
        assert gamma(**atm) == pytest.approx(fd, rel=1e-4)

    def test_vega(self, atm):
        fd = _fd_greek(call_price, atm, "sigma")
        assert vega(**atm) == pytest.approx(fd, rel=1e-5)

    def test_call_theta(self, atm):
        # theta = -dC/dT  (price decreases as T shrinks), but our function
        # returns dC/dT with the conventional negative sign already baked in,
        # so compare against -fd(dC/dT).
        fd = -_fd_greek(call_price, atm, "T")
        assert call_theta(**atm) == pytest.approx(fd, rel=1e-4)

    def test_put_theta(self, atm):
        fd = -_fd_greek(put_price, atm, "T")
        assert put_theta(**atm) == pytest.approx(fd, rel=1e-4)

    def test_call_rho(self, atm):
        fd = _fd_greek(call_price, atm, "r")
        assert call_rho(**atm) == pytest.approx(fd, rel=1e-5)

    def test_put_rho(self, atm):
        fd = _fd_greek(put_price, atm, "r")
        assert put_rho(**atm) == pytest.approx(fd, rel=1e-5)

    def test_greeks_otm(self, otm_call):
        """Repeat delta/gamma/vega checks on an OTM-call scenario."""
        assert call_delta(**otm_call) == pytest.approx(
            _fd_greek(call_price, otm_call, "S"), rel=1e-5
        )
        assert gamma(**otm_call) == pytest.approx(
            _fd_greek(call_delta, otm_call, "S"), rel=1e-4
        )
        assert vega(**otm_call) == pytest.approx(
            _fd_greek(call_price, otm_call, "sigma"), rel=1e-5
        )


# ── 3. Boundary / limiting cases ────────────────────────────────────────────

class TestBoundary:

    def test_deep_itm_call_approaches_intrinsic(self):
        """A deep-ITM call with very short expiry ≈ intrinsic value."""
        S, K, T, r, q, sigma = 200.0, 100.0, 0.001, 0.05, 0.0, 0.20
        intrinsic = S - K * np.exp(-r * T)
        assert call_price(S, K, T, r, q, sigma) == pytest.approx(intrinsic, rel=1e-3)

    def test_deep_otm_call_near_zero(self):
        """A deep-OTM call is nearly worthless."""
        c = call_price(S=50.0, K=200.0, T=0.25, r=0.05, q=0.0, sigma=0.20)
        assert c < 1e-10

    def test_deep_itm_put_approaches_intrinsic(self):
        S, K, T, r, q, sigma = 50.0, 200.0, 0.001, 0.05, 0.0, 0.20
        intrinsic = K * np.exp(-r * T) - S
        assert put_price(S, K, T, r, q, sigma) == pytest.approx(intrinsic, rel=1e-3)

    def test_deep_otm_put_near_zero(self):
        p = put_price(S=200.0, K=50.0, T=0.25, r=0.05, q=0.0, sigma=0.20)
        assert p < 1e-10

    def test_call_delta_deep_itm(self):
        """Deep-ITM call delta → 1."""
        d = call_delta(S=200.0, K=50.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_put_delta_deep_itm(self):
        """Deep-ITM put delta → −1."""
        d = put_delta(S=50.0, K=200.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert d == pytest.approx(-1.0, abs=1e-6)


# ── 4. Put-call symmetry identities ─────────────────────────────────────────

class TestPutCallSymmetry:

    def test_delta_relation(self, atm):
        """call_delta - put_delta = exp(-qT)."""
        diff = call_delta(**atm) - put_delta(**atm)
        assert diff == pytest.approx(np.exp(-atm["q"] * atm["T"]), abs=1e-10)

    def test_theta_relation(self, atm):
        """call_theta - put_theta = qS*exp(-qT) - rK*exp(-rT)."""
        diff = call_theta(**atm) - put_theta(**atm)
        expected = atm["q"] * atm["S"] * np.exp(-atm["q"] * atm["T"]) - atm["r"] * atm["K"] * np.exp(-atm["r"] * atm["T"])
        assert diff == pytest.approx(expected, abs=1e-10)

    def test_rho_relation(self, atm):
        """call_rho + put_rho = K*T*exp(-rT) * (N(d2) - N(-d2)) ... simplified to K*T*exp(-rT)."""
        # Actually: call_rho + put_rho = K*T*e^{-rT}*(N(d2) - N(-d2)) is not a clean identity.
        # The correct identity from put-call parity: call_rho - put_rho relates to derivative
        # of S*e^{-qT} - K*e^{-rT} w.r.t. r = K*T*e^{-rT}.
        diff = call_rho(**atm) - put_rho(**atm)
        expected = atm["K"] * atm["T"] * np.exp(-atm["r"] * atm["T"])
        assert diff == pytest.approx(expected, abs=1e-10)


# ── 5. Vectorised inputs ────────────────────────────────────────────────────

class TestVectorised:

    def test_call_price_array(self):
        S = np.array([90.0, 100.0, 110.0])
        result = call_price(S, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert result.shape == (3,)
        # Monotonically increasing in S
        assert np.all(np.diff(result) > 0)

    def test_put_price_array(self):
        K = np.array([80.0, 100.0, 120.0])
        result = put_price(S=100.0, K=K, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert result.shape == (3,)
        # Monotonically increasing in K
        assert np.all(np.diff(result) > 0)

    def test_greeks_array_shapes(self):
        S = np.array([95.0, 100.0, 105.0])
        kwargs = dict(S=S, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert call_delta(**kwargs).shape == (3,)
        assert gamma(**kwargs).shape == (3,)
        assert vega(**kwargs).shape == (3,)
