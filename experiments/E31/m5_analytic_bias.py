"""
E31: analytic (population) bias of the ambiguity modulation gamma under label noise (Proposition 1).

Minimal model (lag-1 kernel, K symmetric categories):
  true process      P(c_m = i | c_{m-1}) = (1 + r 1[i = c_{m-1}]) / (K + r)        (gamma = 0)
  ambiguity         h_m iid ~ g(h)   (g = empirical distribution of observed vote entropy)
  observation       c_obs = c with prob 1 - f(h), otherwise uniform over the other K-1 categories,
                    applied to both the source event (with its h) and the target event (with its own h)
  fitted model      P(d_m = i | c_obs_{m-1}, h_{m-1}) = (1 + r' e^{gamma h_{m-1}} 1[i = c_obs_{m-1}]) / (K + r' e^{gamma h_{m-1}})

By exchangeability of the non-matching categories, the likelihood depends on the data only through the
binary event "target observed category == source observed category" given h_{m-1}. Its population
probability s(h) is linear in the source accuracy 1 - f(h) (derived below), so if f increases in h and the
true persistence exceeds chance, s(h) decreases in h and the KL projection gives gamma* < 0 even though the
true gamma is 0 (Proposition 1). This script
  (1) computes the pseudo-true (r*, gamma*) exactly by maximising the expected log-likelihood over the
      discretised h distribution, for a grid of noise-link strengths b,
  (2) checks it against Monte-Carlo maximum likelihood on simulated sequences,
  (3) checks the sign of the score at gamma = 0 (the covariance form used in the proof).
Output: E31_results.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

OUT_DIR = config.results_dir("E31")
RNG = np.random.default_rng(20260929)
K = 7
R_TRUE = 2.0          # true persistence strength (P(persist) = (1+r)/(K+r))


def h_distribution():
    """Discretised entropy distribution on [0, log2 5] with the mean and SD of the EL-Friends vote entropy."""
    grid = np.linspace(0.0, np.log2(5), 41)
    w = np.exp(-0.5 * ((grid - 0.63) / 0.41) ** 2)   # mean / SD of the EL-Friends vote entropy
    return grid, w / w.sum()


def f_noise(h, a, b):
    """Label-flip probability increasing in h: 1 - sigmoid(a - b * z(h)), z = standardised h."""
    z = (h - 0.63) / 0.41
    return 1.0 - expit(a - b * z)


def s_match(h, a, b, grid, w, r=R_TRUE):
    """P(target observed == source observed | source ambiguity h) under gamma = 0."""
    p0 = (1 + r) / (K + r)                    # true persistence probability
    e1 = 1 - f_noise(h, a, b)                 # source label correct
    e2 = 1 - np.sum(w * f_noise(grid, a, b))  # target label correct (h of target ~ g, independent)
    other = (1 - e2) / (K - 1)
    s = (e1 * (p0 * e2 + (1 - p0) * other)
         + (1 - e1) * ((1 - p0) / (K - 1) * e2 + (p0 + (1 - p0) * (K - 2) / (K - 1)) * other))
    return s


def p_model(h, rp, gam):
    x = rp * np.exp(gam * h)
    return (1 + x) / (K + x)


def pseudo_true(a, b):
    grid, w = h_distribution()
    s = s_match(grid, a, b, grid, w)

    def negll(th):
        rp, gam = np.exp(th[0]), th[1]
        p = np.clip(p_model(grid, rp, gam), 1e-12, 1 - 1e-12)
        return -np.sum(w * (s * np.log(p) + (1 - s) * np.log(1 - p)))
    res = minimize(negll, x0=[np.log(R_TRUE), 0.0], method="BFGS")
    # score at gamma = 0 with r' profiled out (sign check of the proof)
    res0 = minimize(lambda t: negll([t[0], 0.0]), x0=[np.log(R_TRUE)], method="BFGS")
    rp0 = np.exp(res0.x[0])
    p0 = p_model(grid, rp0, 0.0)
    dp_dgam = grid * rp0 * (K - 1) / (K + rp0) ** 2          # d p / d gamma at gamma = 0
    score = np.sum(w * (s - p0) / (p0 * (1 - p0)) * dp_dgam)
    cov_h_s = np.sum(w * (grid - np.sum(w * grid)) * (s - np.sum(w * s)))
    return {"r_star": float(np.exp(res.x[0])), "gamma_star": float(res.x[1]), "score_at_0": float(score),
            "cov_h_s": float(cov_h_s), "s_range": [float(s.min()), float(s.max())]}


def simulate_mle(a, b, n=40000):
    grid, w = h_distribution()
    h = RNG.choice(grid, size=n + 1, p=w)
    c = np.empty(n + 1, dtype=int)
    c[0] = RNG.integers(K)
    p0 = (1 + R_TRUE) / (K + R_TRUE)
    for m in range(1, n + 1):
        c[m] = c[m - 1] if RNG.random() < p0 else RNG.choice([k for k in range(K) if k != c[m - 1]])
    flip = RNG.random(n + 1) < f_noise(h, a, b)
    shift = RNG.integers(1, K, size=n + 1)
    c_obs = np.where(flip, (c + shift) % K, c)
    match = (c_obs[1:] == c_obs[:-1]).astype(float)
    hs = h[:-1]

    def negll(th):
        p = np.clip(p_model(hs, np.exp(th[0]), th[1]), 1e-12, 1 - 1e-12)
        return -np.mean(match * np.log(p) + (1 - match) * np.log(1 - p))
    res = minimize(negll, x0=[np.log(R_TRUE), 0.0], method="BFGS")
    return float(res.x[1])


out = {"K": K, "r_true": R_TRUE, "rows": []}
a = 1.2   # baseline accuracy logit (fitted EL range: 0.75-1.28)
for b in (0.0, 0.25, 0.5, 0.8, 1.2, 1.6):
    pt = pseudo_true(a, b)
    mc = [simulate_mle(a, b) for _ in range(20)]
    row = {"a": a, "b": b, **pt, "mc_gamma_mean": float(np.mean(mc)), "mc_gamma_sd": float(np.std(mc, ddof=1))}
    out["rows"].append(row)
    print(f"b={b:4.2f}  gamma*={pt['gamma_star']:+.3f}  MC={row['mc_gamma_mean']:+.3f}+/-{row['mc_gamma_sd']:.3f}  "
          f"score(0)={pt['score_at_0']:+.4f}  cov(h,s)={pt['cov_h_s']:+.5f}  s in [{pt['s_range'][0]:.3f},{pt['s_range'][1]:.3f}]")
(OUT_DIR / "E31_results.json").write_text(json.dumps(out, indent=1))
print("wrote", OUT_DIR / "E31_results.json")
