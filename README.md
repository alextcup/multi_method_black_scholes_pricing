# multi-method-bs-pricing

A Python package for pricing European and American options via the Black-Scholes equations using five independent methods: analytic (Black-Scholes formulas), binomial tree, Monte Carlo, finite differences, and a physics-informed neural network. Each method supports continuous dividend yield and computation of the Greeks.

## Installation

```bash
pip install multi-method-bs-pricing
```

The PINN pricer requires TensorFlow, which is an optional extra:

```bash
pip install "multi-method-bs-pricing[pinn]"
```

## Example

```python
from bspricepal import Option, analytic_bs_pricer, binomial_pricer, mc_pricer, fd_pricer

option = Option(
    S0=100.0,      # spot
    K=105.0,       # strike
    T=1.0,         # years to expiry
    r=0.05,        # risk-free rate
    sigma=0.2,     # volatility
    div_yield=0.0,
    option_type="call",     # "call" | "put"
    exercise="european",    # "european" | "american"
)

print(analytic_bs_pricer(option))
print(binomial_pricer(option, n_steps=500))
print(mc_pricer(option, n_paths=100000, n_steps=100))
print(fd_pricer(option, n_space=400, n_steps=400))
```

## Greeks

Exact solution for European options:

```python
from bspricepal import analytic_bs_greeks

print(analytic_bs_greeks(option))
```

Finite difference approximation for any pricer, particularly for American options:

```python
from bspricepal import all_greeks, delta, binomial_pricer

print(all_greeks(binomial_pricer, option, n_steps=500))
print(delta(binomial_pricer, option, h=1e-2, n_steps=500))
```

Individual Greeks available: `delta`, `gamma`, `vega`, `rho`, `theta`.

## API

| Function | Notes |
| --- | --- |
| `analytic_bs_pricer(option)` | Closed form; European only |
| `analytic_bs_greeks(option)` | Closed form; European only |
| `binomial_pricer(option, n_steps=10)` | Binomial tree; European and American |
| `mc_pricer(option, n_paths, n_steps=10, rng=None)` | Monte Carlo; European and American (Longstaff-Schwartz) |
| `fd_pricer(option, n_space, n_steps, xmin=-3, xmax=3)` | Finite difference; European and American (PSOR) |
| `pinn_pricer(option, **kwargs)` | Physics-informed neural network: European and American |
| `all_greeks(pricer, option, **kwargs)` | Finite-difference Greeks for any pricer |
