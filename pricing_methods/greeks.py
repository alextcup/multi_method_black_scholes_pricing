from dataclasses import replace

from option import Option


def delta(pricer, option: Option, h=1e-2, **kwargs):
    dS = h * option.S0

    up = replace(option, S0=option.S0 + dS)
    down = replace(option, S0=option.S0 - dS)

    return (pricer(up, **kwargs) - pricer(down, **kwargs)) / (2 * dS)


def gamma(pricer, option: Option, h=5e-2, **kwargs):
    dS = h * option.S0

    up = replace(option, S0=option.S0 + dS)
    down = replace(option, S0=option.S0 - dS)

    return (pricer(up, **kwargs) - 2 * pricer(option, **kwargs) + pricer(down, **kwargs)) / dS ** 2


def vega(pricer, option: Option, h=1e-2, **kwargs):
    up = replace(option, sigma=option.sigma + h)
    down = replace(option, sigma=option.sigma - h)

    return (pricer(up, **kwargs) - pricer(down, **kwargs)) / (2 * h)


def rho(pricer, option: Option, h=1e-2, **kwargs):
    up = replace(option, r=option.r + h)
    down = replace(option, r=option.r - h)

    return (pricer(up, **kwargs) - pricer(down, **kwargs)) / (2 * h)


def theta(pricer, option: Option, h=1e-2, **kwargs):
    up = replace(option, T=option.T + h)
    down = replace(option, T=option.T - h)

    return -(pricer(up, **kwargs) - pricer(down, **kwargs)) / (2 * h)


def all_greeks(pricer, option: Option, **kwargs):
    price = pricer(option, **kwargs)
    d = delta(pricer, option, **kwargs)
    g = gamma(pricer, option, **kwargs)
    v = vega(pricer, option, **kwargs)
    th = theta(pricer, option, **kwargs)
    r = rho(pricer, option, **kwargs)

    return (price, d, g, v, th, r)
