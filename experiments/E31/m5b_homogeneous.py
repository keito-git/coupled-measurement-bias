"""
E31b: Proposition 1 with homogeneous annotators (accuracy unrelated to any latent ambiguity).

Even when every annotator has the same accuracy q, the observed vote entropy h of an item is evidence
about whether its plurality label is wrong: unanimous items are almost never mislabelled, split items
often are. Hence f(h) = P(plurality label wrong | observed entropy h) co-varies positively with h, which
is the only property the proof of Proposition 1 uses (Cov_g(h, f(h)) > 0). This script
  (1) simulates 5-annotator votes with constant accuracy q and uniform errors over K-1 categories,
      plurality label with random tie-break, and tabulates g(h) and f(h) over the observed entropy values,
  (2) computes the pseudo-true gamma* of the hard-mark lag-1 model under true gamma = 0 with these g and f
      (same population computation as m5_analytic_bias.py),
  (3) checks it against Monte-Carlo MLE on simulated sequences whose labels come from the simulated votes.
Output: E31b_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

OUT_DIR = config.results_dir("E31")
RNG = np.random.default_rng(20260929)
K, N_ANN, R_TRUE = 7, 5, 2.0


def vote_items(c, q):
    n = len(c)
    correct = RNG.random((n, N_ANN)) < q
    wrong = (c[:, None] + RNG.integers(1, K, size=(n, N_ANN))) % K
    votes = np.where(correct, c[:, None], wrong)
    counts = np.zeros((n, K), int)
    for j in range(N_ANN):
        np.add.at(counts, (np.arange(n), votes[:, j]), 1)
    p = counts / N_ANN
    h = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1)
    lab = np.argmax(counts + RNG.random((n, K)) * 1e-3, 1)
    return np.round(h, 6), lab


def tabulate(q, n=400000):
    c = RNG.integers(K, size=n)
    h, lab = vote_items(c, q)
    grid = np.unique(h)
    g = np.array([(h == v).mean() for v in grid])
    f = np.array([(lab[h == v] != c[h == v]).mean() for v in grid])
    return grid, g, f


def pseudo_true(grid, g, f):
    p0 = (1 + R_TRUE) / (K + R_TRUE)
    e1 = 1 - f
    e2 = 1 - np.sum(g * f)
    other = (1 - e2) / (K - 1)
    s = (e1 * (p0 * e2 + (1 - p0) * other)
         + (1 - e1) * ((1 - p0) / (K - 1) * e2 + (p0 + (1 - p0) * (K - 2) / (K - 1)) * other))

    def negll(th):
        x = np.exp(th[0]) * np.exp(th[1] * grid)
        p = np.clip((1 + x) / (K + x), 1e-12, 1 - 1e-12)
        return -np.sum(g * (s * np.log(p) + (1 - s) * np.log(1 - p)))
    res = minimize(negll, x0=[np.log(R_TRUE), 0.0], method="BFGS")
    cov = float(np.sum(g * (grid - np.sum(g * grid)) * (f - np.sum(g * f))))
    return float(res.x[1]), cov


def mc_mle(q, n=40000):
    p0 = (1 + R_TRUE) / (K + R_TRUE)
    c = np.empty(n + 1, dtype=int)
    c[0] = RNG.integers(K)
    for m in range(1, n + 1):
        c[m] = c[m - 1] if RNG.random() < p0 else (c[m - 1] + RNG.integers(1, K)) % K
    h, lab = vote_items(c, q)
    match = (lab[1:] == lab[:-1]).astype(float)
    hs = h[:-1]

    def negll(th):
        x = np.exp(th[0]) * np.exp(th[1] * hs)
        p = np.clip((1 + x) / (K + x), 1e-12, 1 - 1e-12)
        return -np.mean(match * np.log(p) + (1 - match) * np.log(1 - p))
    return float(minimize(negll, x0=[np.log(R_TRUE), 0.0], method="BFGS").x[1])


out = {"K": K, "n_annotators": N_ANN, "r_true": R_TRUE, "rows": []}
for q in (0.6, 0.7, 0.8):
    grid, g, f = tabulate(q)
    gs, cov = pseudo_true(grid, g, f)
    mc = [mc_mle(q) for _ in range(20)]
    row = {"q": q, "gamma_star": gs, "cov_h_f": cov, "overall_flip": float(np.sum(g * f)),
           "mc_gamma_mean": float(np.mean(mc)), "mc_gamma_sd": float(np.std(mc, ddof=1)),
           "f_by_h": {f"{v:.3f}": [float(gi), float(fi)] for v, gi, fi in zip(grid, g, f)}}
    out["rows"].append(row)
    print(f"q={q}: flip={row['overall_flip']:.3f} Cov(h,f)={cov:+.4f} gamma*={gs:+.3f} "
          f"MC={row['mc_gamma_mean']:+.3f}+/-{row['mc_gamma_sd']:.3f}")
(OUT_DIR / "E31b_results.json").write_text(json.dumps(out, indent=1))
print("wrote", OUT_DIR / "E31b_results.json")
