"""Population profile likelihood of the lag-1 persistence model (shared by m5c and m5d)."""
import numpy as np
from scipy.optimize import minimize_scalar

GAMMAS = np.linspace(-5, 5, 1001)


def s_match(e1, e2, p0, K):
    """P(target observed label == source observed label | source accuracy e1) under gamma = 0."""
    other = (1 - e2) / (K - 1)
    return (e1 * (p0 * e2 + (1 - p0) * other)
            + (1 - e1) * ((1 - p0) / (K - 1) * e2 + (p0 + (1 - p0) * (K - 2) / (K - 1)) * other))


def profile(grid, w, s, K, model):
    """Profile expected log-likelihood over GAMMAS (nuisance r' or beta maximised for each gamma)."""
    out = np.empty(len(GAMMAS))
    for i, g in enumerate(GAMMAS):
        def negll(t):
            if model == "M1":
                x = np.exp(t) * np.exp(g * grid)
                p = (1 + x) / (K + x)
            else:
                y = np.exp(t + g * grid)
                p = y / (y + K - 1)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return -np.sum(w * (s * np.log(p) + (1 - s) * np.log(1 - p)))
        out[i] = -minimize_scalar(negll, bounds=(-15, 15), method="bounded").fun
    return out


def summarize(prof):
    """Number of interior local maxima, argmax gamma and whether the argmax is at the grid edge."""
    d = np.diff(prof)
    n_max = int(np.sum((d[:-1] > 0) & (d[1:] <= 0)))
    g_star = float(GAMMAS[int(np.argmax(prof))])
    at_edge = bool(np.argmax(prof) in (0, len(GAMMAS) - 1))
    return n_max, g_star, at_edge
