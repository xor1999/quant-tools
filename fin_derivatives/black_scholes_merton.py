"""
Black-Scholes and Merton Jump-Diffusion option pricing.

Implements European call pricing under the classical Black-Scholes model and
the Merton (1976) jump-diffusion extension. Includes a robust implied
volatility solver that combines bisection bracketing with Newton-Raphson
polishing.

The runner (`run_and_save`) prices calls under Merton JD for a grid of
maturities and moneyness levels, then back-solves the Black-Scholes implied
volatility to visualize the jump-induced smile/skew.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import os

# --------------- Black-Scholes ---------------

class EuropeanCallPricer:
    """
    European call option pricer under Black-Scholes and Merton Jump-Diffusion.

    Supports:
      - Closed-form Black-Scholes call price and vega.
      - Merton (1976) jump-diffusion call price via Poisson-weighted sum of
        shifted Black-Scholes prices.

    Parameters:
        S:     current stock price (spot).
        K:     strike price.
        T:     time to maturity in years.
        r:     risk-free interest rate (annualized).
        q:     continuous dividend yield (annualized), default 0.0.
        sigma: volatility of the underlying (annualized), default 0.2.
    """

    def __init__(self, S, K, T, r, q=0.0, sigma=0.2):
        self.S = S        # Spot price
        self.K = K        # Strike price
        self.T = T        # Time to maturity
        self.r = r        # Risk-free rate
        self.q = q        # Dividend yield
        self.sigma = sigma # Volatility

    def d1(self):
        """Compute d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))."""
        return (np.log(self.S/self.K) + (self.r - self.q + 0.5*self.sigma**2)*self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        """Compute d2 = d1 - sigma * sqrt(T)."""
        return self.d1() - self.sigma * np.sqrt(self.T)

    def BS_call_price(self):
        """Return the Black-Scholes European call price: S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)."""
        d1 = self.d1()
        d2 = self.d2()
        call_price = (self.S * np.exp(-self.q * self.T) * stats.norm.cdf(d1)) - (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        return call_price
    
    def Merton_call_price(self, lamQ, gamma, tail_tol=1e-12):
        """
        Calculate the Call option price using Merton Jump-Diffusion model.
        Parameters:
            lamQ : float : Jump intensity (average number of jumps per year)
            gamma : float : Average jump size (percentage change in stock price due to a jump, e.g., -0.1 for a 10% drop)
            tail_tol : float : Tolerance for the tail probability to stop summation (default is 1e-12)
        Returns:
            float : The Merton Jump-Diffusion Call option price
        """
        j = 0 # initialize counter of jumps
        weight = np.exp(- lamQ * self.T) # initial weight for j=0 jump events
        total = 0.0 # cumulative probability
        price = 0.0 # cumulative price
        q_eff = lamQ * gamma  # risk-neutral compensator encoded as an "effective dividend yield"


        # For each possible number of jumps, calculate the weighted BS call price. Stop when the tail probability is below tolerance or after 1000 jumps.
        while True:
            S_j = self.S * ((1.0 + gamma) ** j)
            pj = weight
            price += pj * EuropeanCallPricer(S_j, self.K, self.T, self.r, q=q_eff, sigma=self.sigma).BS_call_price()
            total += pj
            j += 1
            weight *=  lamQ * self.T / j

            if 1 - total < tail_tol or j > 1000: # stopping rule
                break

        return float(price)

    def BS_vega(self):
        """Return the Black-Scholes vega: dC/d(sigma) = S*e^{-qT}*phi(d1)*sqrt(T)."""
        d1 = self.d1()
        return self.S * np.exp(-self.q * self.T) * stats.norm.pdf(d1) * np.sqrt(self.T)

# --------------- Implied Volatility (Brent-like bracket + Newton polish) ---------------

def implied_vol_call(price, S, K, T, r, q=0.0, tol=1e-12):
    """
    From a given European Call option market price, compute the Black-Scholes implied volatility 
    using a combination of a Brent-like bracketing method and Newton-Raphson polishing.
    Parameters:
        price : float : Market price of the European Call option or price from another model
        S : float : Current stock price (spot price)
        K : float : Strike price of the option
        T : float : Time to maturity in years
        r : float : Risk-free interest rate (annualized)
        q : float : Dividend yield (annualized), default is 0.0
        tol : float : Tolerance for convergence (default is 1e-12)
    Returns:
        float : The Black-Scholes implied volatility
    """

    # Returns the difference between BS call price and the given price
    def f(sig): return EuropeanCallPricer(S, K, T, r, q, sig).BS_call_price() - price

    def bracketing(a = 1e-12, b = 5, n_max = 20, f = f):
        """
        It allows to find a bracket [a,b] such that f(a) and f(b) have opposite signs.
        Parameters:
            a : float : Initial left endpoint of the bracket (default is 1e-12)
            b : float : Initial right endpoint of the bracket (default is 5)
            n_max : int : Maximum number of iterations to expand the bracket (default is 20)
            f : function : The function for which we want to find a root
        """
        fa, fb = f(a), f(b)
        n = 0
        while fa*fb > 0 and n < n_max:
            b *= 2.0
            fb = f(b)
            n += 1
        return a, b, fa, fb
    
    def bisection(a, b, f, tol):
        """
        Perform the bisection method to find a root of the function f in the interval [a, b].
        Parameters:
            a : float : Left endpoint of the interval
            b : float : Right endpoint of the interval
            f : function : The function for which we want to find a root
            tol : float : Tolerance for convergence
        """
        fa, fb = f(a), f(b)

        if fa*fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        # Takes the two current brackent points [a, b] that have opposite signs.
        # Takes the mid point and evaluate if its a good approximation by computing f(mid)
        # or having already a small interval b-a
        # If it is not a good interval, squeeze the interval whether by decreasing b or increasing a
        for _ in range(100):
            mid = 0.5*(a + b)
            fmid = f(mid)

            if abs(fmid) < tol or (b - a) < tol:
                return mid
            
            if fa*fmid < 0:
                b, fb = mid, fmid

            else:
                a, fa = mid, fmid

        return 0.5*(a + b)
    
    def newtonraph_method(price, sigma_guess, S, K, T, r, q):
        """
        From an educated guess, use Newton-Raphson polishing to find the implied volatility.
        """
        sig = sigma_guess

        for _ in range(10):
            call = EuropeanCallPricer(S, K, T, r, q, sig)
            v = call.BS_vega()

            if v < 1e-16:
                break
            diff = call.BS_call_price() - price
            step = diff / v

            new_sig = sig - step

            if new_sig <= 0 or abs(step) > 2.0:
                break

            sig = new_sig
        
        return float(max(0.0, sig))

    F = S*np.exp((r - q)*T) # forward price
    lower = np.exp(-r*T)*max(F - K, 0.0) # corresponding lower bound for call price (no arbitrage)
    upper = S*np.exp(-q*T) # corresponding upper bound for call price (no arbitrage)

    # Handle edge cases for prices outside no-arbitrage bounds, if the model admits arbitrage then return NaN or 0 accordingly
    if price <= lower + 1e-14:
        return 0.0
    if price >= upper - 1e-14:
        return np.nan

    # Refine the bracket using a Brent-like method
    left, right, _, _ = bracketing()

    # Generate a first guess using bisection
    sig_first_guess = bisection(left, right, f, tol = tol)

    # Find the implied vol by polishing the first guess through the Newton-Raphson method
    implied_vol = newtonraph_method(price, sig_first_guess, S, K, T, r, q)

    return implied_vol

# --------------- Runner ---------------

def run_and_save():
    """Price Merton JD calls across maturities/strikes, invert for BS implied vols, and plot the smile."""

    # Parameters
    S0 = 100.0 # initial stock price
    r = 0.04 # risk-free rate
    sigma = 0.20 # volatility
    lamQ = 0.20 # jump intensity
    gamma = -0.08  # downward jump => negative gamma (assignment sign convention)

    Ts = [0.02, 0.08, 0.25, 0.50] # maturities
    K_over_S = [0.8, 0.9, 1.0, 1.1] # strikes/initial price ratios
    Ks = [S0 * k for k in K_over_S] # strikes

    # Calculate prices and implied vols
    rows = []
    for T in Ts:
        for K in Ks:
             # Calculates the call price according to MERTON JD model
            price = EuropeanCallPricer(S = S0, K = K, T = T, r = r, q=0, sigma = sigma).Merton_call_price(lamQ, gamma, tail_tol=1e-14)
            iv = implied_vol_call(price, S0, K, T, r, q=0.0) # Calculates the BS implied vol from the JD call price using Brent-like + Newton method
            rows.append((T, K/S0, price, iv)) # Store results

    df = pd.DataFrame(rows, columns=['T', 'K_over_S0', 'JD_call_price', 'BS_implied_vol'])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'jd_iv_results.csv'), index=False)

    # Plotting the implied volatility for each maturity
    for T in Ts:
        sub = df[df['T']==T].sort_values('K_over_S0')
        plt.figure(figsize=(6,4))
        plt.plot(sub['K_over_S0'], sub['BS_implied_vol'], marker='o')
        plt.title(f'Implied Volatility vs Strike (T = {T:.2f} years)')
        plt.xlabel('K / S')
        plt.ylabel('Implied Volatility')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'IV_vs_strike_T_{T:.2f}.png'), dpi=200, bbox_inches='tight')

if __name__ == "__main__":
    run_and_save()
