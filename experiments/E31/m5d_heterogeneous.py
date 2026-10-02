"""
E31d: annotator heterogeneity. Same population setting and profile-likelihood computation as
m5c_unimodality.py (M1 = DT-AMHP, M2 = multinomial logit), but the A = 5 annotators have different, fixed
accuracies q_a = mean_q + sd_q * z_a (z = standardised evenly spaced pattern, clipped to [0.05, 0.99]); errors are
uniform over the other K-1 categories; plurality label with random tie-break. g(h) and f(h) are obtained by exact
enumeration of all K^A vote vectors. Output: E31d_results.json
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from profile_lik import s_match, profile, summarize  # noqa: E402

OUT_DIR = config.results_dir("E31")
A = 5
Z = np.linspace(-1, 1, A)
Z = (Z - Z.mean()) / Z.std()


def hetero_table(K, qs):
    probs = {}
    for votes in itertools.product(range(K), repeat=A):           # true category = 0
        pr = 1.0
        for a, v in enumerate(votes):
            pr *= qs[a] if v == 0 else (1 - qs[a]) / (K - 1)
        counts = np.bincount(votes, minlength=K)
        p = counts[counts > 0] / A
        h = float(-(p * np.log2(p)).sum())
        top = counts.max()
        n_top = int((counts == top).sum())
        p_wrong = 1 - (1.0 / n_top if counts[0] == top else 0.0)
        key = round(h, 9)
        gp, gw = probs.get(key, (0.0, 0.0))
        probs[key] = (gp + pr, gw + pr * p_wrong)
    hs = np.array(sorted(probs))
    g = np.array([probs[h][0] for h in hs])
    f = np.array([probs[h][1] / probs[h][0] for h in hs])
    return hs, g / g.sum(), f


rows = []
for K, mean_q, sd_q, r in itertools.product((3, 7), (0.6, 0.7), (0.0, 0.05, 0.10, 0.15, 0.20), (2.0,)):
    qs = np.clip(mean_q + sd_q * Z, 0.05, 0.99)
    hs, g, f = hetero_table(K, qs)
    e2 = 1 - np.sum(g * f)
    p0 = (1 + r) / (K + r)
    s = s_match(1 - f, e2, p0, K)
    cov = float(np.sum(g * (hs - np.sum(g * hs)) * (s - np.sum(g * s))))
    for model in ("M1", "M2"):
        n_max, g_star, edge = summarize(profile(hs, g, s, K, model))
        rows.append({"K": K, "mean_q": mean_q, "sd_q": sd_q, "q": [float(x) for x in qs], "model": model,
                     "gamma_star": g_star, "n_local_max": n_max, "at_edge": edge, "cov_h_s": cov,
                     "label_error": float(np.sum(g * f))})
(OUT_DIR / "E31d_results.json").write_text(json.dumps({"rows": rows}, indent=1))
for r_ in rows:
    print(r_["K"], r_["mean_q"], r_["sd_q"], r_["model"], round(r_["gamma_star"], 3), r_["n_local_max"], round(r_["cov_h_s"], 5))
