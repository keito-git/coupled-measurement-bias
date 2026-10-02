"""
E30b: AR(1) annotator model for M3ED by direct maximisation of the truncated likelihood.

With the truncated (majority-only) likelihood the (a, b) M-step of the EM in run_E30.py ignores the
truncation normaliser, so EM is not guaranteed to ascend. Here the EM confusion matrix and prior are kept
and the same truncated marginal log-likelihood (sum over dialogues of the forward-algorithm log Z) is
maximised directly over (a, b, rho) with Nelder-Mead from several starting points; the best run is kept.
The posterior predictive simulation rejection-samples each item's votes until a majority exists.
Output: E30b_m3ed.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_E30 as R  # noqa: E402

OUT_DIR = R.OUT_DIR
dlgs = R.load_m3ed_votes()
fit0 = json.loads((OUT_DIR / "E30_fit.json").read_text())["m3ed"]
conf = np.array(fit0["confusion"])
prior = np.array(fit0["prior"])
n_ann = 3
dlg_votes = [np.stack(d) for d in dlgs]
log_conf_nd = np.log(np.maximum(conf, 1e-300))
np.fill_diagonal(log_conf_nd, 0.0)
log_prior = np.log(prior + 1e-300)


def loglik(theta):
    a, b, rho = theta
    if not (0.0 <= rho <= 0.99) or b < 0:
        return -np.inf
    log_T = np.log(R.make_transition(rho) + 1e-300)
    log_p_valid = R.compute_log_p_valid(n_ann, a, b, conf)
    total = 0.0
    for votes in dlg_votes:
        log_em, _ = R.compute_log_emission(votes, n_ann, a, b, log_conf_nd, log_prior,
                                           truncate=True, log_p_valid=log_p_valid)
        _, _, log_z = R.forward_backward_dlg(log_em, log_T)
        total += log_z
    return total


starts = [(fit0["a"], fit0["b"], fit0["rho"]), (1.8, 1.0, 0.8), (1.5, 1.5, 0.6)]
runs = []
for s in starts:
    ll0 = loglik(s)
    res = minimize(lambda t: -loglik(t), x0=np.array(s), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-3, "maxiter": 150})
    runs.append({"start": list(s), "start_logL": float(ll0), "a": float(res.x[0]), "b": float(res.x[1]),
                 "rho": float(res.x[2]), "logL": float(-res.fun), "nit": int(res.nit), "success": bool(res.success)})
    print(runs[-1], flush=True)
best = max(runs, key=lambda r: r["logL"])


def simulate_truncated(a, b, rho, seed):
    rng = np.random.default_rng(seed)
    T = R.make_transition(rho)
    p_norm = prior / prior.sum()
    out = []
    for dlg in dlgs:
        M = len(dlg)
        u = np.empty(M, dtype=int)
        u[0] = int(rng.choice(R.N_U, p=R.PI0))
        for m in range(1, M):
            u[m] = int(rng.choice(R.N_U, p=T[u[m - 1]]))
        sim = []
        for m in range(M):
            q = min(max(1.0 / (1.0 + math.exp(-(a - b * R.U_NODES[u[m]]))), 1e-4), 1 - 1e-4)
            while True:  # keep only vote patterns with a majority
                c = int(rng.choice(R.K, p=p_norm))
                row = conf[c].copy()
                row[c] = 0.0
                row /= row.sum()
                v = np.zeros(R.K, dtype=np.int64)
                for _ in range(n_ann):
                    v[c if rng.random() < q else int(rng.choice(R.K, p=row))] += 1
                if v.max() >= 2:
                    break
            sim.append(v)
        out.append(sim)
    return out


real = R.compute_vote_stats(dlgs, n_ann)
sims = [R.compute_vote_stats(simulate_truncated(best["a"], best["b"], best["rho"], 1000 + i), n_ann) for i in range(20)]
agg = R.aggregate_sim_stats(sims)
out = {"runs": runs, "best": best, "real": real, "ppc": agg}
(OUT_DIR / "E30b_m3ed.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print("best:", best)
print("real stats:", {k: real[k] for k in ("consec_H_corr", "H_obs_mean", "H_obs_sd")}, real.get("plurality_dist"))
print("ppc keys:", list(agg)[:12])
