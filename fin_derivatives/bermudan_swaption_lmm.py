"""
Bermudan receiver swaption pricing under a LIBOR Market Model (LMM).

Simulates forward LIBOR rates under the spot measure using a two-factor
correlation structure rho_{k,j} = cos(theta_k - theta_j) with
theta_k = (pi/2) * k / (K-1). Forward dynamics follow log-Euler
discretization with a predictor-corrector drift and a terminal-rate
correction for the last accrual period.

The Bermudan exercise strategy is learned via a threshold-based backward
induction: at each exercise date the holder exercises if the swaption
intrinsic value exceeds an optimal threshold H_j, calibrated on training
paths by grid search. Pricing is then performed on an independent set of
test paths using the learned thresholds.
"""

import numpy as np
from itertools import product
rng1 = np.random.default_rng(0)
rng2 = np.random.default_rng(42)

def check_array(array):
    """Validate that the input is a numpy ndarray; raise ValueError otherwise."""
    if isinstance(array, np.ndarray):
        return array
    else:
        raise ValueError("This is not an array:", array)


class Simulation():
    """
    Monte Carlo simulator for forward LIBOR rates under the spot measure.

    Generates correlated Brownian increments via a two-factor decomposition
    dW_k = cos(theta_k)*dW^(1) + sin(theta_k)*dW^(2), then evolves the full
    forward rate curve F_k(T_j) at each tenor step using a log-Euler scheme
    with risk-neutral drift.

    Parameters:
        number_paths:              number of Monte Carlo paths.
        initial_forward_structure: flat or shaped array of initial forward rates F_k(0).
        volatility_vector:         piecewise-constant volatility sigma_k per forward.
        exercise_dates:            array of Bermudan exercise dates.
        final_payment_date:        last payment date T_N of the underlying swap.
        tau:                       accrual period (year fraction per tenor step).
        seed:                      numpy Generator for reproducibility.
    """

    def __init__(self, number_paths, initial_forward_structure, volatility_vector, exercise_dates, final_payment_date, tau, seed = rng1):
        ### General parameters of simulations
        self.N_sim : int = number_paths
        self.forward_structure : np.ndarray = check_array(initial_forward_structure)
        self.vol_vector : np.ndarray = check_array(volatility_vector)
        self.exercise_dates : np.ndarray = check_array(exercise_dates)
        self.last_exercise_date : float  = self.exercise_dates[-1]
        self.final_payment_date : float = final_payment_date
        self.tau : float = tau
        self.set_dates : np.ndarray = np.arange(0.0, self.final_payment_date+1e-12, tau)
        self._theta_k = lambda k : (np.pi / 2) * (k / 15) 

        ### Build correlated brownian motions that drive the forward rate F
        self.rng = seed
        self.nSteps = len(self.set_dates) - 1 
        self.nForwards = len(self.set_dates) - 1 
        self.correlated_brownian_motions = self._factored_brownian_motions()
        self.corr = self._precompute_corr()

        ### Store simulations
        self.simulated_paths = []
        self.simulated_paths = self._launch_simulation()
    
    def _independent_brownian_motion_generator(self):
        """Generate independent 2-factor Brownian increments of shape (N_sim, nSteps, 2)."""
        B = 2
        T = self.nSteps
        N = self.N_sim

        brownian_Q = self.rng.normal(scale=np.sqrt(self.tau), loc = 0.0, size=(N, T, B))

        return brownian_Q
    
    def _factors_vector_builder(self):
        """
        Recall that dW_k = cos(theta_k) * dW^(1) + sin(theta_k) * dW^(2)

        We want to build u_k = [cos(theta_k) sin(theta_k)].T for all k = 1...16 
        """
        K = self.nForwards  # number of increments
        k = np.arange(K)
        theta = self._theta_k(k)
        u = np.column_stack([np.cos(theta), np.sin(theta)])

        return u                                                   # (nForwards, 2)
    
    def _factored_brownian_motions(self):
        """Project 2-factor independent increments onto per-forward correlated increments via einsum."""
        dW = self._independent_brownian_motion_generator()  # (N,nSteps,2)
        u  = self._factors_vector_builder()                 # (nForwards,2)
        return np.einsum("nmb,kb->nmk", dW, u)              # (N,nSteps,nForwards)
    
    def _launch_simulation(self):
        """Simulate all N_sim forward-rate paths and return as a list of 2-D arrays."""
        return [self._construct_one_path(n) for n in range(self.N_sim)]

    def _build_structure_path(self):
        """Allocate a (time_steps x forwards) matrix initialized with the t=0 curve."""
        axis_1 = self.set_dates[1:]
        axis_0 = np.arange(0.0, self.last_exercise_date + 1e-12, self.tau)

        path = np.full(shape=(len(axis_0), len(axis_1)), fill_value=np.nan)
        path[0,] = self.forward_structure

        return path, axis_0, axis_1

    def _construct_one_path(self, n):
        """Evolve the full forward curve for path n through all tenor steps."""
        simulation = self.correlated_brownian_motions[n, ...]

        path, axis_0, axis_1 = self._build_structure_path()

        for id_j, Tj_prev in enumerate(axis_0[:-1]):
            for id_k in range(id_j, len(axis_1)):
                Tk = axis_1[id_k]
                path[id_j+1,id_k] = self._compute_next_forward_k(Tj_prev, Tk, id_j, id_k, axis_1, simulation, path)

        return path

    def _compute_next_forward_k(self, Tj_prev, Tk, id_j, id_k, axis_1, sim, path):
        """
        Log-Euler step for forward F_k from T_{j} to T_{j+1}.

        Uses the standard LMM drift when Tj_prev < Tk - tau, and a
        terminal-rate correction (variance / 6, diffusion / sqrt(3)) when
        the forward is one accrual period from fixing.
        """
        mu_k    = self._compute_mu_k(Tj_prev, Tk, id_k, id_j, axis_1, path)
        sigma_k = self._obtain_sigma_associated(id_k)
        g_k = self._compute_volatility_factor(Tk, Tj_prev)
        sigm_eff = g_k * sigma_k

        dW_k    = sim[id_j, id_k]
        F_prev  = path[id_j, id_k]
        
        if Tj_prev < (Tk - self.tau):
            F_new = F_prev * np.exp(self.tau * (mu_k - 0.5 * sigm_eff**2) + sigm_eff * dW_k)

        elif np.isclose(Tj_prev, Tk - self.tau):
            F_new = F_prev * np.exp(self.tau * (mu_k - (1/6)*sigma_k**2) + sigma_k * dW_k / np.sqrt(3))

        else:
            F_new = np.nan

        return F_new

    def _compute_mu_k(self, Tj_prev, Tk, id_k, id_j, axis_1, path):
        """Compute the risk-neutral drift of forward k at step j: mu_k = sigma_k * g_k * sum_j(...)."""
        vol_k = self._obtain_sigma_associated(id_k)
        g_k   = self._compute_volatility_factor(Tk, Tj_prev)

        s = 0.0
        for j in range(0, id_k+1):
            Tj_fwd = axis_1[j]
            rho_kj = self.corr[id_k, j]
            vol_j  = self._obtain_sigma_associated(j)
            g_j    = self._compute_volatility_factor(Tj_fwd, Tj_prev)
            R_j    = path[id_j, j]
            if np.isnan(R_j):
                continue

            s += self.tau * rho_kj * vol_j * g_j * R_j / (1.0 + self.tau * R_j)

        return vol_k * g_k * s

    def _obtain_sigma_associated(self, k):
        """Return the piecewise-constant volatility sigma_k for forward k."""
        return self.vol_vector[k]

    def _compute_volatility_factor(self, Tk, Tj):
        """Volatility damping factor g_k = min(max(Tk - Tj, 0) / tau, 1)."""
        num = np.maximum(Tk - Tj, 0)
        return np.minimum(num/self.tau, 1)

    def _precompute_corr(self):
        """Build the K x K correlation matrix rho_{k,j} = cos(theta_k - theta_j)."""
        K = self.nForwards
        theta = (np.pi/2) * np.arange(K) / (K-1)
        return np.cos(theta[:, None] - theta[None, :])


class SwaptionPricing():
    """
    Evaluates receiver swaption values and discount factors along simulated
    forward-rate paths, and learns optimal exercise thresholds via backward
    induction over a discrete threshold grid.

    At each exercise date T_j the intrinsic value is
        V_j = tau * sum_{i=j+1}^{N} P(T_j, T_i) * (K - F_i(T_j))
    where P(T_j, T_i) is the zero-coupon bond price bootstrapped from
    simulated forwards.

    The `learn` method sweeps thresholds from T_{x-1} back to T_s and
    picks the threshold h* at each date that maximizes the expected
    discounted payoff under the current (partially optimized) policy.
    """

    def __init__(self, exercise_dates, final_payment_date, simulated_paths, tau, strike, thresholds, strategy = None, H = None):
        self.exercise_dates : np.ndarray = check_array(exercise_dates)
        self.first_exercise_date : float = self.exercise_dates[0]
        self.last_exercise_date : float  = self.exercise_dates[-1]
        self.final_payment_date : float = final_payment_date
        self.tau : float = tau
        self.set_dates : np.ndarray = np.arange(0.0, self.final_payment_date+1e-12, tau)

        self.sim_paths : np.ndarray = simulated_paths
        self.strike : float = strike

        self.thresholds : np.ndarray = thresholds
        self.swaptions_values : np.ndarray = np.array(self._swaptions_values())
        self.discount_factors : np.ndarray = np.array(self._discount_factors())

        self.H = H

    def learn(self):
        """Learn optimal exercise thresholds H via backward induction over the threshold grid."""
        n_paths = len(self.sim_paths)
        n_ex    = len(self.exercise_dates)

        opt_h        = np.zeros(n_ex, dtype=float)

        Ts = float(self.first_exercise_date)
        Tx = float(self.last_exercise_date)

        ex_to_idx = {float(np.round(t / self.tau) * self.tau): i
                    for i, t in enumerate(self.exercise_dates)}
        id_Tx = ex_to_idx[Tx]

        # backward induction over exercise dates
        Tj = Tx - self.tau
        
        while Tj >= Ts - 1e-12:
            id_Tj = ex_to_idx[Tj]
            best_value = -np.inf
            best_h     = None
        
            for h in self.thresholds:
                payoff_sum = 0

                for n in range(n_paths):
                    optimal_t = None

                    if self.swaptions_values[n, id_Tj] > h:
                        optimal_t = id_Tj
                    
                    else:
                        for t in range(id_Tj+1, id_Tx):
                            if self.swaptions_values[n, t] > opt_h[t]:
                                optimal_t = t
                                break

                        if optimal_t is None:
                            if self.swaptions_values[n, id_Tx] > 0.0:
                                optimal_t = id_Tx
                    
                    if optimal_t is None:
                        payoff = 0
                    else:
                        payoff = self.discount_factors[n, optimal_t] * self.swaptions_values[n, optimal_t]
                
                    payoff_sum += payoff

                Vhat = payoff_sum / n_paths

                if Vhat > best_value:
                    best_value = Vhat
                    best_h = h
            
            opt_h[id_Tj] = best_h
            Tj -= self.tau
        
        self.H = opt_h
    
    def _swaptions_values(self):
        """Compute the receiver swaption intrinsic value at each exercise date for every path."""
        Ts    = self.first_exercise_date
        Tx    = self.last_exercise_date
        TN    = self.final_payment_date
        id_TN = np.where(self.set_dates == TN)[0][0]
        id_Ts = np.where(self.set_dates == Ts)[0][0]
        id_Tx = np.where(self.set_dates == Tx)[0][0]

        out = []

        for path in self.sim_paths:
            grid_swap_values = []

            for t in range(id_Ts, id_Tx+1):
                grid_swap_values.append(self._compute_swaption_value(t, id_TN, path))
            
            out.append(grid_swap_values)
            
        return out
    
    def _compute_swaption_value(self, t, id_TN, path):
        """Intrinsic value of the receiver swap at time index t: tau * sum P(t,i)*(K - F_i)."""
        s = 0.0
        for i in range(t+1, id_TN+1):
            p_ti = self._compute_zcb(t, i, path)
            R_it = path[t, i-1]
            s += p_ti * (self.strike - R_it)   # receiver: K - forward
        return self.tau * s

    def _compute_zcb(self, t, i, path):
        """Zero-coupon bond P(t, i) = prod_{x=t+1}^{i} 1/(1 + tau * F_x(t))."""
        x = t + 1
        res = 1

        while x <= i:
            R_xt = path[t,x-1]
            den = 1 + R_xt * self.tau
            res *= 1/den
            x += 1
        
        return res

    def _discount_factors(self):
        """Compute the money-market discount factor D(0, T_e) for each exercise date and path."""
        Ts    = self.first_exercise_date
        Tx    = self.last_exercise_date
        id_Ts = np.where(self.set_dates == Ts)[0][0]
        id_Tx = np.where(self.set_dates == Tx)[0][0]

        out = []
        
        for path in self.sim_paths:
            grid_discount_factors = []

            for e in range(id_Ts, id_Tx + 1):
                D = 1
                for j in range(1, e+1):
                    R_j_j = path[j-1, j-1]
                    den = 1 + self.tau * R_j_j
                    D *= 1/den

                grid_discount_factors.append(D)
        
            out.append(grid_discount_factors)

        return out

class BermudanSwaptionMC:
    """
    High-level interface for training and pricing a Bermudan receiver swaption.

    Wraps SwaptionPricing to provide a clean train/price API:
      1. `train(paths)` learns optimal exercise thresholds on training paths.
      2. `price(paths)` applies the learned thresholds to independent test
         paths and returns the Monte Carlo price and standard error.
    """

    def __init__(self, exercise_dates, final_payment_date, tau, strike, thresholds):
        self.exercise_dates = check_array(exercise_dates).astype(float)
        self.final_payment_date = float(final_payment_date)
        self.tau = float(tau)
        self.strike = float(strike)
        self.thresholds = check_array(thresholds).astype(float)

        self.optimal_H = None

    def train(self, simulated_paths):
        """Learn optimal thresholds H from training paths and return them."""
        trainer = SwaptionPricing(
            exercise_dates=self.exercise_dates,
            final_payment_date=self.final_payment_date,
            simulated_paths=simulated_paths,
            tau=self.tau,
            strike=self.strike,
            thresholds=self.thresholds,
        )

        trainer.learn()

        self.optimal_H = trainer.H

        return self.optimal_H

    def price(self, simulated_paths):
        """Apply learned thresholds to test paths and return (MC price, standard error)."""
        assert self.optimal_H is not None, "Strategy not trained"

        pricer = SwaptionPricing(
            exercise_dates=self.exercise_dates,
            final_payment_date=self.final_payment_date,
            simulated_paths=simulated_paths,
            tau=self.tau,
            strike=self.strike,
            thresholds=self.thresholds,
        )

        payoffs = np.zeros(len(simulated_paths), dtype=float)
        for n in range(len(simulated_paths)):
            payoffs[n] = self._apply_thresholds_to_path(n, pricer)

        price = float(payoffs.mean())
        std = float(payoffs.std(ddof=1))
        stderr = std / np.sqrt(len(payoffs))

        return price, stderr

    def _apply_thresholds_to_path(self, n: int, pricer: SwaptionPricing) -> float:
        """Return the discounted payoff for path n under the threshold exercise policy."""
        x = len(self.exercise_dates) - 1
        optimal_time = None

        # try early exercise dates up to x-1
        for j in range(0, x):
            if pricer.swaptions_values[n, j] > self.optimal_H[j]:
                optimal_time = j
                break

        # if never exercised early, check last date ITM rule
        if optimal_time is None:
            if pricer.swaptions_values[n, x] > 0.0:
                optimal_time = x

        if optimal_time is None:
            return 0.0

        return pricer.discount_factors[n, optimal_time] * pricer.swaptions_values[n, optimal_time]

def main():
    """
    Price a 1y-into-2y Bermudan receiver swaption (quarterly exercise,
    K = 5%, T_N = 4yr) under a flat 5% forward curve with piecewise-constant
    volatilities (20%, 22%, 24%). Trains thresholds on 10k paths, prices on 2k.
    """
    tau = 0.25
    exercise_dates = np.arange(1.0, 2.0 + 1e-12, tau)
    TN = 4.0
    length_dates = int(TN / tau)

    strike = 0.05
    term_structure = np.ones(length_dates, dtype=float) * 0.05

    vol_k1 = np.ones(7) * 0.20
    vol_k2 = np.ones(4) * 0.22
    vol_k3 = np.ones(5) * 0.24
    vector_vol = np.concatenate([vol_k1, vol_k2, vol_k3]).astype(float)

    sim_train = Simulation(
        number_paths=10000,
        initial_forward_structure=term_structure,
        volatility_vector=vector_vol,
        exercise_dates=exercise_dates,
        final_payment_date=TN,
        tau=tau,
        seed=rng1,
    )

    sim_test = Simulation(
        number_paths=2000,
        initial_forward_structure=term_structure,
        volatility_vector=vector_vol,
        exercise_dates=exercise_dates,
        final_payment_date=TN,
        tau=tau,
        seed=rng2,
    )

    engine = BermudanSwaptionMC(
        exercise_dates=exercise_dates,
        final_payment_date=TN,
        tau=tau,
        strike=strike,
        thresholds=np.linspace(0.0, 0.05, 1000),
    )

    H = engine.train(sim_train.simulated_paths)
    price, stderr = engine.price(sim_test.simulated_paths)

    print("Optimal thresholds H:", H)
    print(f"Bermudan swaption price: {price}")
    print(f"MC stderr: {stderr}")


if __name__ == "__main__":
    main()