"""
E31c: numerical check that the population profile likelihood is unimodal, for the DT-AMHP and for a
second downstream model.

Population setting as in m5_analytic_bias.py (lag-1, K symmetric categories, true gamma = 0); the observed
entropy distribution g(h) and label-error curve f(h) are either
  (A) parametric: f(h) = 1 - sigmoid(a - b z(h)), h ~ discretised normal (EL-Friends mean/SD), grid over b, or
  (B) homogeneous annotators: A annotators with constant accuracy q, plurality label; g and f tabulated from
      exact enumeration of vote patterns.
For each setting the profile expected log-likelihood l(gamma) = max_r' E[log p] is evaluated on a fine gamma
grid; we count interior local maxima, record gamma*, and check sign(gamma*) < 0.

Fitted models:
  M1 (DT-AMHP):  p(h) = (1 + r' e^{g h}) / (K + r' e^{g h})
  M2 (first-order multinomial logit with a disagreement interaction on persistence):
      p(h) = e^{beta + g h} / (e^{beta + g h} + K - 1)
Output: E31c_results.json
"""
import itertools
import json
import sys
from math import comb, log2
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from profile_lik import s_match, profile, summarize  # noqa: E402

OUT_DIR = config.results_dir("E31")


def parametric_settings():
    grid = np.linspace(0.0, np.log2(5), 41)
    w = np.exp(-0.5 * ((grid - 0.63) / 0.41) ** 2)
    w /= w.sum()
    z = (grid - 0.63) / 0.41
    for K, r, b in itertools.product((3, 4, 7, 14), (1.0, 2.0, 4.0), np.round(np.arange(0.0, 2.001, 0.05), 2)):
        f = 1 - expit(1.2 - b * z)
        e2 = 1 - np.sum(w * f)
        p0 = (1 + r) / (K + r)
        yield {"kind": "parametric", "K": K, "r": r, "b": float(b)}, grid, w, s_match(1 - f, e2, p0, K)


def vote_table(K, A, q):
    """Exact g(h) and f(h) for A annotators with accuracy q, uniform errors, plurality label with random tie-break."""
    probs = {}
    # enumerate counts: n_true votes for the true category, and a multiset of wrong votes over K-1 others
    for n_true in range(A + 1):
        n_wrong = A - n_true
        p_n = comb(A, n_true) * q ** n_true * (1 - q) ** n_wrong
        # distribute wrong votes over K-1 categories uniformly: multinomial over compositions
        for comp in compositions(n_wrong, K - 1):
            mult = multinomial(n_wrong, comp) / (K - 1) ** n_wrong
            counts = (n_true,) + comp
            pr = p_n * mult
            h = -sum((c / A) * log2(c / A) for c in counts if c > 0)
            top = max(counts)
            n_top = sum(c == top for c in counts)
            p_wrong = 1 - (1.0 / n_top if counts[0] == top else 0.0)
            key = round(h, 9)
            gp, gw = probs.get(key, (0.0, 0.0))
            probs[key] = (gp + pr, gw + pr * p_wrong)
    hs = np.array(sorted(probs))
    g = np.array([probs[h][0] for h in hs])
    f = np.array([probs[h][1] / probs[h][0] for h in hs])
    return hs, g / g.sum(), f


def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in compositions(n - i, k - 1):
            yield (i,) + rest


def multinomial(n, comp):
    out = 1
    rem = n
    for c in comp:
        out *= comb(rem, c)
        rem -= c
    return out


def homogeneous_settings():
    for K, A, q, r in itertools.product((3, 4, 7, 14), (3, 5), (0.5, 0.6, 0.7, 0.8, 0.9), (1.0, 2.0, 4.0)):
        if K ** A > 2e6:
            continue
        hs, g, f = vote_table(K, A, q)
        e2 = 1 - np.sum(g * f)
        p0 = (1 + r) / (K + r)
        yield {"kind": "homogeneous", "K": K, "A": A, "q": q, "r": r}, hs, g, s_match(1 - f, e2, p0, K)


rows = []
for maker in (parametric_settings, homogeneous_settings):
    for meta, grid, w, s in maker():
        for model in ("M1", "M2"):
            n_max, g_star, edge = summarize(profile(grid, w, s, meta["K"], model))
            cov = float(np.sum(w * (grid - np.sum(w * grid)) * (s - np.sum(w * s))))
            rows.append({**meta, "model": model, "n_local_max": n_max, "gamma_star": g_star, "at_edge": edge,
                         "cov_h_s": cov})
res = {"n_settings": len(rows), "rows": rows}
agg = {}
for r_ in rows:
    k = (r_["kind"], r_["model"])
    a = agg.setdefault(k, {"n": 0, "unimodal": 0, "neg_when_cov_neg": 0, "n_cov_neg": 0, "zero_when_cov0": 0, "n_cov0": 0})
    a["n"] += 1
    a["unimodal"] += int(r_["n_local_max"] <= 1)
    if r_["cov_h_s"] < -1e-12:
        a["n_cov_neg"] += 1
        a["neg_when_cov_neg"] += int(r_["gamma_star"] < 0)
    else:
        a["n_cov0"] += 1
        a["zero_when_cov0"] += int(abs(r_["gamma_star"]) < 0.02)
res["summary"] = {f"{k[0]}/{k[1]}": v for k, v in agg.items()}
(OUT_DIR / "E31c_results.json").write_text(json.dumps(res, indent=1))
for k, v in res["summary"].items():
    print(k, v)
