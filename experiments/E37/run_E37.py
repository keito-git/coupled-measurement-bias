"""
E37: MC-SIMEX for the modulation parameter under the matched data-generating process of E34e.

The naive estimator is the hard-mark within-cell fit. Each event's label-misclassification matrix
P(majority label = k | true = j) is evaluated at the accuracy implied by the posterior mean difficulty
E[u | votes] (pi_table.npy, tabulated by Monte Carlo on an accuracy grid). Labels are re-drawn B_RELABEL
times from the lambda-th matrix power (lambda = 1, 2, 3; integer powers keep the matrices stochastic), the
naive estimator is refitted, and the estimates are extrapolated to lambda = -1 (quadratic and linear).
gamma in {0, -0.4}, 30 replications each. Requires RESULTS_ROOT/E29/E29_fit.json.

Outputs (RESULTS_ROOT/E37): pi_table.npy, E37_results.json
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

import latent_mark as d  # noqa: E402
from latent_mark import simulate_matched as simulate, fit_within_cell  # noqa: E402

OUT = config.results_dir("E37")
LAMBDAS = (1.0, 2.0, 3.0)
B_RELABEL = 3
REPS = 30
GAMMAS = (0.0, -0.4)
Q_GRID = np.linspace(0.02, 0.999, 60)
GH_U, GH_W = np.polynomial.hermite_e.hermegauss(41)
GH_W = GH_W / GH_W.sum()


def row_conf(conf, j):
    r = conf[j].copy()
    r[j] = 0.0
    return r / r.sum() if r.sum() > 1e-12 else np.where(np.arange(d.K_EL) == j, 0.0, 1.0 / (d.K_EL - 1))


def build_pi_table(conf, n_ann, n_mc=20000, seed=123):
    """T[g, j, k] = P(majority label = k | true = j, q = Q_GRID[g]) by Monte Carlo (random tie-break)."""
    rng = np.random.default_rng(seed)
    K = d.K_EL
    T = np.zeros((len(Q_GRID), K, K))
    for j in range(K):
        row = row_conf(conf, j)
        for g, q in enumerate(Q_GRID):
            correct = rng.random((n_mc, n_ann)) < q
            wrong = rng.choice(K, size=(n_mc, n_ann), p=row)
            lab = np.where(correct, j, wrong)
            counts = np.zeros((n_mc, K))
            for a in range(n_ann):
                counts[np.arange(n_mc), lab[:, a]] += 1
            mx = counts.max(1, keepdims=True)
            tied = (counts == mx) / (counts == mx).sum(1, keepdims=True)
            T[g, j] = tied.mean(0)
    return T


def pi_for_q(T, q):
    q = float(np.clip(q, Q_GRID[0], Q_GRID[-1]))
    g = np.searchsorted(Q_GRID, q) - 1
    g = int(np.clip(g, 0, len(Q_GRID) - 2))
    w = (q - Q_GRID[g]) / (Q_GRID[g + 1] - Q_GRID[g])
    return (1 - w) * T[g] + w * T[g + 1]


def post_mean_u(votes, a, b, conf, prior, n_ann):
    """E[u | votes] under the i.i.d. annotator model (true category marginalised)."""
    K = d.K_EL
    q = 1.0 / (1.0 + np.exp(-(a - b * GH_U)))           # (G,)
    lik = np.zeros(len(GH_U))
    for c in range(K):
        row = row_conf(conf, c)
        lp = np.zeros(len(GH_U))
        for k in range(K):
            n = votes[k]
            if n < 0.5:
                continue
            pk = q if k == c else (1 - q) * row[k]
            lp += n * np.log(np.maximum(pk, 1e-300))
        lik += prior[c] * np.exp(lp)
    w = GH_W * lik
    return float((w * GH_U).sum() / w.sum())


def matrix_power(P, lam):
    """Integer power of a row-stochastic matrix; bad = number of entries < -1e-9 (always 0 here)."""
    M = np.linalg.matrix_power(P, int(round(lam)))
    bad = int((M < -1e-9).sum())
    return M / M.sum(1, keepdims=True), bad


def fractional_matrix_power(P, lam):
    """Eigen-decomposition power for non-integer lambda; bad = number of negative entries before clipping."""
    vals, vecs = np.linalg.eig(P)
    M = np.real(vecs @ np.diag(np.power(vals.astype(complex), lam)) @ np.linalg.inv(vecs))
    bad = int((M < -1e-9).sum())
    M = np.clip(M, 0.0, None)
    M = M / M.sum(1, keepdims=True)
    return M, bad


def naive(dlgs, seed):
    ev = [[{"cat": x["cat"], "H": x["H"], "plurality": x["plurality"]} for x in dlg] for dlg in dlgs]
    r = fit_within_cell(ev, seed=seed)
    return float(r["gamma_hat"]) if r["converged"] else float("nan")


def simex(dlgs, T, ann, rep_seed):
    a, b = ann["a"], ann["b"]
    conf = np.asarray(ann["confusion"], dtype=float)
    prior = np.asarray(ann["prior"], dtype=float)
    pis = [[pi_for_q(T, 1 / (1 + np.exp(-(a - b * post_mean_u(x["votes"], a, b, conf, prior, d.N_ANN_EL)))))
            for x in dlg] for dlg in dlgs]
    rng = np.random.default_rng(rep_seed)
    est = {0.0: naive(dlgs, rep_seed)}
    n_bad = 0
    cache = {}
    for lam in LAMBDAS:
        vals = []
        for bb in range(B_RELABEL):
            new = []
            for dlg, pd in zip(dlgs, pis):
                nd = []
                for x, P in zip(dlg, pd):
                    key = (id(P), lam)
                    if key not in cache:
                        cache[key] = matrix_power(P, lam)
                    M, bad = cache[key]
                    n_bad += bad
                    nd.append({"cat": int(rng.choice(d.K_EL, p=M[x["cat"]])), "H": x["H"], "plurality": x["plurality"]})
                new.append(nd)
            r = fit_within_cell(new, seed=rep_seed + 7 * bb + int(lam * 100))
            if r["converged"]:
                vals.append(float(r["gamma_hat"]))
        est[lam] = float(np.mean(vals)) if vals else float("nan")
    lam = np.array([0.0] + list(LAMBDAS))
    y = np.array([est[0.0]] + [est[l] for l in LAMBDAS])
    quad = float(np.polyval(np.polyfit(lam, y, 2), -1.0))
    lin = float(np.polyval(np.polyfit(lam, y, 1), -1.0))
    return {"naive": est[0.0], "path": {str(k): v for k, v in est.items()}, "simex_quad": quad, "simex_lin": lin,
            "n_negative_entries": n_bad}


def worker(args):
    rep, gamma, lens, mu, alpha, beta, ann, T = args
    dlgs = simulate(rep, gamma, lens, mu, alpha, beta, ann)
    out = simex(dlgs, T, ann, rep_seed=rep * 1009 + int(abs(gamma) * 1000))
    out.update({"rep": rep, "gamma": gamma})
    return out


def main():
    lens, mu, alpha, beta = d.fit_el_wide_generator()
    ann = d.load_annotator_params()
    T = build_pi_table(np.asarray(ann["confusion"], dtype=float), d.N_ANN_EL)
    np.save(OUT / "pi_table.npy", T)
    print("pi table built; P(correct) at q=0.5:", [round(T[np.searchsorted(Q_GRID, 0.5), j, j], 3) for j in range(d.K_EL)], flush=True)
    d._warmup_numba()
    args = [(r, g, lens, mu, alpha, float(beta), ann, T) for g in GAMMAS for r in range(REPS)]
    if os.environ.get("E37_PILOT"):
        args = args[:1]
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(config.n_workers(3)) as pool:
        for r in pool.imap_unordered(worker, args):
            rows.append(r)
            (OUT / "E37_results.json").write_text(json.dumps({"rows": rows}, indent=1))
            print(len(rows), r["gamma"], round(r["naive"], 3), round(r["simex_quad"], 3), round(r["simex_lin"], 3),
                  "neg", r["n_negative_entries"], flush=True)
    summ = {}
    for g in GAMMAS:
        for k in ("naive", "simex_quad", "simex_lin"):
            v = np.array([r[k] for r in rows if r["gamma"] == g and np.isfinite(r[k])])
            if v.size:
                summ[f"g{g:+.1f}_{k}"] = {"n": int(v.size), "mean": float(v.mean()),
                                         "se": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")}
    (OUT / "E37_results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
