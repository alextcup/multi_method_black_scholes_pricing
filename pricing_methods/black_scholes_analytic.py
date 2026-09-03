from scipy.stats import norm
import math

from option import Option

def analytic_bs_pricer(option: Option):
    S0 = option.S0
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    sigma = option.sigma

    d_1 = (math.log(S0 / K) + (r - div_yield + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)
    
    if option.option_type == "call":
        price = math.exp(-div_yield * T) * S0 * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)

    elif option.option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d_2) - math.exp(-div_yield * T) * S0 * norm.cdf(-d_1)

    return price