"""Multi-method Black-Scholes option pricing.

Prices European and American options by analytic formula, binomial tree,
Monte Carlo, Crank-Nicolson finite differences and a physics-informed
neural network.
"""

__version__ = "0.1.0"

from .option import Option

from .pricing_methods.binomial import binomial_pricer
from .pricing_methods.black_scholes_analytic import (
    analytic_bs_pricer,
    analytic_bs_greeks,
)
from .pricing_methods.greeks import (
    delta,
    gamma,
    vega,
    rho,
    theta,
    all_greeks,
)
from .pricing_methods.finite_difference import (
    fd_pricer,
    payoff as fd_payoff,
    crank_nicolson_transformed_am,
)
from .pricing_methods.monte_carlo import (
    mc_pricer,
    payoff as mc_payoff,
)


_PINN_EXPORTS = {
    "pinn_payoff": "payoff",
    "build_model": "build_model",
    "train": "train",
    "pinn_pricer": "pinn_pricer",
}


def __getattr__(name):
    if name in _PINN_EXPORTS:
        from .pricing_methods import pinn

        value = getattr(pinn, _PINN_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "Option",
    "binomial_pricer",
    "analytic_bs_pricer",
    "analytic_bs_greeks",
    "delta",
    "gamma",
    "vega",
    "rho",
    "theta",
    "all_greeks",
    "fd_pricer",
    "fd_payoff",
    "crank_nicolson_transformed_am",
    "mc_pricer",
    "mc_payoff",
    "pinn_payoff",
    "build_model",
    "train",
    "pinn_pricer",
    "__version__",
]