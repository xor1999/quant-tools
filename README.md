# quant-tools

A collection of quantitative-finance tools for derivatives pricing, Greeks computation, and risk-neutral analysis.

## Modules

All modules live under `fin_derivatives/`.

| Module | Description |
|--------|-------------|
| `bs_put_call_pricer.py` | Black-Scholes European put/call pricer with closed-form Greeks (delta, gamma, vega, theta, rho) |
| `bs_visualisation.py` | 3-D surface plots of option prices and Greeks with quiver arrows showing first- and second-order sensitivities |
| `black_scholes_merton.py` | Black-Scholes call pricer, Merton (1976) jump-diffusion extension, and implied-volatility solver (bisection + Newton-Raphson) |
| `american_asian_lsmc.py` | Longstaff-Schwartz Monte Carlo pricing of Bermudan Asian call options under constant and local volatility |
| `bermudan_swaption_lmm.py` | Bermudan receiver swaption pricing under a two-factor LIBOR Market Model with threshold-based exercise |
| `monte_carlo.py` | Monte Carlo engine: GBM path simulation, European pricing (naive + antithetic variates), American pricing via Longstaff-Schwartz LSMC |
| `breeden_litzenberger_copula.py` | Risk-neutral density extraction (Breeden-Litzenberger) and Gaussian-copula Monte Carlo for multi-asset exotics |

## Setup

```bash
pip install -r requirements.txt
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `mc_demo.ipynb` | Monte Carlo demo: GBM sample paths, European pricing convergence (naive vs antithetic), and LSMC American put pricing with early exercise premium analysis |

## Usage

Each module can be run as a standalone script. For example:

```bash
# Price a put/call and print Greeks
python fin_derivatives/bs_put_call_pricer.py

# Generate 3-D price and Greek surface plots
python fin_derivatives/bs_visualisation.py

# Run Merton jump-diffusion pricing and implied-vol smile inversion
python fin_derivatives/black_scholes_merton.py

# Run Monte Carlo European and American pricing
python fin_derivatives/monte_carlo.py
```

## Tests

```bash
pytest tests/
```

## License

MIT