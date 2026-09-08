import numpy as np
from scipy.linalg import solve_banded

from option import Option


def payoff(S, K, option_type):
    if option_type == "call":
        return np.maximum(S - K, 0)
    elif option_type == "put":
        return np.maximum(K - S, 0)


def crank_nicolson_matrices(dx, dt, n_space):
    l = dt / dx ** 2
    diag1 = np.array([0.5 * l for i in range(n_space-2)])
    diag2 = [1 + l for i in range(n_space-1)]
    diag3 = [1 - l for i in range(n_space-1)]

    A = np.zeros((3, n_space-1))
    A[0,1:] = -1 * diag1
    A[1,:] = diag2
    A[2,:-1] = -1 * diag1

    B = np.diag(diag3) + np.diag(diag1, k=-1) + np.diag(diag1, k=1)

    return A, B


def crank_nicolson_bcs(q, option_type):
    if option_type == "call":
        r1 = lambda x, t: 0
        r2 = lambda x, t: np.exp(0.5 * (q + 1) * x + 0.25 * (q + 1) ** 2 * t)
    elif option_type == "put":
        r1 = lambda x, t:  np.exp(0.5 * (q - 1) * x + 0.25 * (q - 1) ** 2 * t)
        r2 = lambda x, t: 0

    return r1, r2


def crank_nicolson_transformed_eur(option: Option, n_space, n_steps, xmin=-3, xmax=3):
    T = option.T
    r = option.r
    sigma = option.sigma
    div_yield = option.div_yield
    option_type = option.option_type

    dx = (xmax - xmin) / n_space

    tmax = 0.5 * sigma ** 2 * T
    dt = tmax / n_steps

    l = dt / dx ** 2

    q = 2 * (r - div_yield) / sigma ** 2

    x = np.array([xmin + i * dx for i in range(n_space+1)])

    w = []

    w.append(payoff(np.exp(0.5 * x * (q + 1)), np.exp(0.5 * x * (q - 1)), option_type))

    A, B = crank_nicolson_matrices(dx, dt, n_space)
    r1, r2 = crank_nicolson_bcs(q, option_type)
    for v in range(0, n_steps):
        g = np.zeros((n_space-1,))
        g[0] = r1(xmin, dt*(v+1)) + r1(xmin, dt*v)
        g[-1] = r2(xmax, dt*(v+1)) + r2(xmax, dt*v)
        d = 0.5 * l * g
        c = B @ w[v][1:-1] + d

        w1 = np.zeros((n_space+1,))

        w1[1:-1] = solve_banded((1, 1), A, c)
        w1[0] = r1(xmin, (v+1) * dt)
        w1[-1] = r2(xmax, (v+1) * dt)

        w.append(w1)

    return np.array(w)


def transform_to_original(option: Option, y, n_space, n_steps, xmin=-3, xmax=3):
    K = option.K
    T = option.T
    r = option.r
    sigma = option.sigma
    div_yield = option.div_yield

    dx = (xmax - xmin) / n_space

    tmax = 0.5 * sigma ** 2 * T
    dt = tmax / n_steps

    l = dt / dx ** 2

    q = 2 * (r - div_yield) / sigma ** 2
    p = 2 * r / sigma ** 2

    x = np.array([xmin + i * dx for i in range(n_space+1)])
    tau = tmax

    v = K * np.exp(-0.5 * (q - 1) * x - (0.25 * (q-1) ** 2 + p) * tau) * y

    S = K * np.exp(x)
    t = T - (2 / sigma ** 2) * tau

    return S, t, v


def fd_pricer_eur(option: Option, n_space, n_steps, xmin=-3, xmax=3):
    y = crank_nicolson_transformed_eur(option, n_space, n_steps, xmin, xmax)
    S, t, v = transform_to_original(option, y[-1], n_space, n_steps, xmin, xmax)

    return np.interp(option.S0, S, v)


def g(option: Option, x, t):
    r = option.r
    sigma = option.sigma
    div_yield = option.div_yield
    option_type = option.option_type

    q = 2 * (r - div_yield) / sigma ** 2
    p = 2 * r / sigma ** 2
    return np.exp(0.25 * t * ((q - 1)**2 + 4 * p)) * payoff(np.exp(0.5 * x * (q + 1)), np.exp(0.5 * x * (q - 1)), option_type)


def init_w(option: Option, n_space, v, xmin, dx, dt):
    l = dt / dx ** 2

    w = np.zeros((n_space+1,))
    w[0] = g(option, xmin, v * dt)
    w[n_space] = g(option, xmin + n_space * dx, v * dt)
    for i in range(1, n_space):
        w[i] = g(option, xmin + i * dx, v * dt)

    return w


def discretization(option: Option, w, n_space, v, xmin, dx, dt):
    l = dt / dx ** 2
    b = np.zeros((n_space-1,))

    for i in range(1, n_space):
        b[i-1] = w[i] + 0.5 * l * (w[i+1] - 2 * w[i] + w[i-1])

    b[0] = w[1] + 0.5 * l * (w[2] - 2 * w[1] + g(option, xmin, v * dt)) + 0.5 * l * g(option, xmin, (v + 1) * dt)
    b[n_space-2] = w[n_space-1] + 0.5 * l * (g(option, xmin + n_space * dx, v * dt) - 2 * w[n_space-1] + w[n_space-2]) + 0.5 * l * g(option, xmin + n_space * dx, (v + 1) * dt)

    return b


def crank_nicolson_transformed_am(option: Option, n_space, n_steps, xmin=-3, xmax=3, eps=1e-10, omega=1.0, max_itr=10000):
    T = option.T
    sigma = option.sigma

    dx = (xmax - xmin) / n_space

    tmax = 0.5 * sigma ** 2 * T
    dt = tmax / n_steps

    l = dt / dx ** 2
    alpha = l / 2

    w = init_w(option, n_space, 0.0, xmin, dx, dt)

    w_vals = [w]

    for v in range(n_steps):
        t = v * dt
        b = discretization(option, w, n_space, v, xmin, dx, dt)
        g_cur = g(option, xmin + np.array([i for i in range(n_space+1)]) * dx, v * dt)
        g_next = g(option, xmin + np.array([i for i in range(n_space+1)]) * dx, (v + 1) * dt)

        cur_vec = np.maximum(w, g_next)
        cur_vec[n_space] = 0
        new_vec = np.zeros_like(cur_vec)

        for i in range(1, n_space):
            rho = (b[i-1] + alpha * (new_vec[i-1] + cur_vec[i+1])) / (1 + 2 * alpha)
            new_vec[i] = np.maximum(g(option, xmin + i * dx, (v + 1) * dt), cur_vec[i] + omega * (rho - cur_vec[i]))
        itr = 0
        while np.linalg.norm(new_vec - cur_vec) > eps and itr <= max_itr:
            itr += 1
            cur_vec = np.copy(new_vec)
            cur_vec[n_space] = 0
            new_vec[0] = 0
            for i in range(1, n_space):
                rho = (b[i-1] + alpha * (new_vec[i-1] + cur_vec[i+1])) / (1 + 2 * alpha)
                new_vec[i] = np.maximum(g(option, xmin + i * dx, (v + 1) * dt), cur_vec[i] + omega * (rho - cur_vec[i]))
        new_vec[0] = g_next[0]
        new_vec[n_space] = g_next[n_space]
        w = np.copy(new_vec)
        w_vals.append(w)

    return np.array(w_vals)

def fd_pricer_am(option: Option, n_space, n_steps, xmin=-3, xmax=3, eps=1e-10, omega=1.0, max_itr=10000):
    y = crank_nicolson_transformed_am(option, n_space, n_steps, xmin, xmax, eps, omega, max_itr)[-1]
    S, t, v = transform_to_original(option, y, n_space, n_steps, xmin, xmax)

    return np.interp(option.S0, S, v)


def fd_pricer(option: Option, n_space, n_steps, xmin=-3, xmax=3, **kwargs):
    if option.exercise == "european":
        return fd_pricer_eur(option, n_space, n_steps, xmin, xmax, **kwargs)
    elif option.exercise == "american":
        return fd_pricer_am(option, n_space, n_steps, xmin, xmax, **kwargs)
