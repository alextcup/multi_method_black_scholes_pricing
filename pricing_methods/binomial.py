import numpy as np

from option import Option


def binomial_pricer(option: Option, n_steps=10):
    S0 = option.S0
    K = option.K
    T = option.T
    r = option.r
    div_yield = option.div_yield
    sigma = option.sigma
    option_type = option.option_type
    exercise = option.exercise

    rd = r - div_yield

    dt = T / n_steps
    beta = 0.5 * (np.exp(-rd * dt) + np.exp((rd + sigma**2) * dt))

    u = beta + np.sqrt(beta**2 - 1)
    d = 1 / u
    p = (np.exp(rd * dt) - d) / (u - d)

    S = np.zeros((n_steps+1, n_steps+1))
    V = np.zeros((n_steps+1, n_steps+1))

    S[0,0] = S0

    if exercise == "american":
        for i in range(1, n_steps):
            for j in range(0, i+1):
                S[j,i] = S0 * u**j * d**(i - j)

    for j in range(0, n_steps+1):
        S[j,n_steps] = S0 * u**j * d**(n_steps - j)

        if option_type == "call":
            V[j, n_steps] = max(S[j,n_steps] - K, 0)
        if option_type == "put":
            V[j, n_steps] = max(K - S[j,n_steps], 0)

    for i in range(n_steps - 1, -1, -1):
        for j in range(0, i+1):
            V[j,i] = np.exp(-r * dt) * (p * V[j+1, i+1] + (1 - p) * V[j, i+1])

            if exercise == "american":
                if option_type == "call":
                    V[j,i] = max(max(S[j,i] - K, 0), V[j,i])
                elif option_type == "put":
                    V[j,i] = max(max(K - S[j,i], 0), V[j,i])


    return V[0,0]
