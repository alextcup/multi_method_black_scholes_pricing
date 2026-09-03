import numpy as np

from option import Option


def integrate_sde(y0, T, n_steps, a, b):
    dt = T / n_steps
    y_values = [y0]
    for i in range(0, n_steps):
        ti = dt * (i+1)
        dW = np.random.normal(0, np.sqrt(dt))

        yi = y_values[i]

        ai = a(yi)
        bi = b(yi)

        z = yi + ai * dt + bi * np.sqrt(dt)
        y = yi + ai * dt + bi * dW + (1 / (2 * np.sqrt(dt))) * (dW**2 - dt) * (b(z) - bi)

        y_values.append(y)

    return y_values


def payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0)
    elif option_type == "put":
        return np.maximum(K - S, 0)


def simulate_paths(y0, T, n_paths, n_steps, a, b):
    dt = T / n_steps

    S = np.zeros((n_steps+1, n_paths))
    S[0,:] = y0
    for i in range(0, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), size=n_paths)

        yi = S[i,:]

        ai = a(yi)
        bi = b(yi)

        z = yi + ai * dt + bi * np.sqrt(dt)
        S[i+1,:] = yi + ai * dt + bi * dW + (1 / (2 * np.sqrt(dt))) * (dW**2 - dt) * (b(z) - bi)

    return S


def mc_pricer_eur(option: Option, n_paths, n_steps=10):
    S0 = option.S0
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    sigma = option.sigma
    option_type = option.option_type

    rd = r - div_yield

    dt = T / n_steps

    a = lambda S: rd * S
    b = lambda S: sigma * S

    S = simulate_paths(S0, T, n_paths, n_steps, a, b)[-1,:]

    V = payoff(S, K, option_type)
    expectation = np.mean(V)

    price = np.exp(-r * T) * expectation

    return price


def mc_pricer_am(option: Option, n_paths, n_steps=10):
    S0 = option.S0
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    sigma = option.sigma
    option_type = option.option_type

    rd = r - div_yield

    dt = T / n_steps

    drift = lambda S: rd * S
    diffusion = lambda S: sigma * S

    S = simulate_paths(S0, T, n_paths, n_steps, drift, diffusion)[1:,:]

    g = payoff(S[n_steps-1,:], K, option_type)
    tau = n_steps * np.ones((n_paths,))

    for i in range(n_steps-1, 0, -1):
        x_data = S[i-1,:]
        itm_mask = payoff(x_data, K, option_type) > 0
        y_data = np.exp(-r * (tau - i) * dt) * g
        A = np.vander(x_data[itm_mask], 4)

        coeffs, _, _, _ = np.linalg.lstsq(A, y_data[itm_mask], rcond=None)

        approx = A @ coeffs

        ex_mask = np.zeros(n_paths, dtype=bool)
        ex_mask[itm_mask] = payoff(S[i-1,itm_mask], K, option_type) >= approx

        g[ex_mask] = payoff(S[i-1,ex_mask], K, option_type)
        tau[ex_mask] = i

    mean = np.mean(np.exp(-r * tau * dt) * g)
    price = np.maximum(payoff(S0, K, option_type), mean)

    return price


def mc_pricer(option: Option, n_paths, n_steps=10):
    if option.exercise == "european":
        return mc_pricer_eur(option, n_paths, n_steps)
    elif option.exercise == "american":
        return mc_pricer_am(option, n_paths, n_steps)
