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

    if option.exercise == "american":
        raise ValueError("option exercise must be european")

    d_1 = (math.log(S0 / K) + (r - div_yield + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)
    
    if option.option_type == "call":
        price = math.exp(-div_yield * T) * S0 * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)

    elif option.option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d_2) - math.exp(-div_yield * T) * S0 * norm.cdf(-d_1)

    return price


def analytic_bs_greeks(option: Option):
    S0 = option.S0
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    sigma = option.sigma

    if option.exercise == "american":
        raise ValueError("option exercise must be european")

    d_1 = (math.log(S0 / K) + (r - div_yield + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-div_yield * T)
    pdf_d1 = norm.pdf(d_1)

    gamma = disc_q * pdf_d1 / (S0 * sigma * math.sqrt(T))
    vega = S0 * disc_q * pdf_d1 * math.sqrt(T)
    time_decay = -S0 * disc_q * pdf_d1 * sigma / (2 * math.sqrt(T))

    if option.option_type == "call":
        delta = disc_q * norm.cdf(d_1)
        theta = time_decay + div_yield * S0 * disc_q * norm.cdf(d_1) - r * K * disc_r * norm.cdf(d_2)
        rho = K * T * disc_r * norm.cdf(d_2)

    elif option.option_type == "put":
        delta = -disc_q * norm.cdf(-d_1)
        theta = time_decay - div_yield * S0 * disc_q * norm.cdf(-d_1) + r * K * disc_r * norm.cdf(-d_2)
        rho = -K * T * disc_r * norm.cdf(-d_2)

    return (analytic_bs_pricer(option), delta, gamma, vega, theta, rho)