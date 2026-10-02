"""
E36b (post hoc): generator without within-cell refinement (n_refine = 0, the modifier is the raw latent H);
checks whether the raw-H oracle estimator recovers gamma. Same generator and estimators as E36.
Output: E36b_results.json
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from dtsim_core import (K, load_el_raw, empirical_q_pool, compute_empirical_confusion, empirical_H_pool,  # noqa: E402
                        residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded, unpack_v_dt)
from dtsim_fits import Q_NOISE_STD, fit_within_cell, fit_raw_H  # noqa: E402
from dtsim_kernels import _process_one_dlg_scheme_c_nb, _seed_nb, _warmup_numba  # noqa: E402

OUT_DIR = config.results_dir("E36")
REPS = 30
GAMMAS = (-0.4, -0.8)


def worker(args):
    rep, gamma, lens, mu, alpha, beta, H_dist, H_gm, H_sorted, q_sorted, conf = args
    _seed_nb(rep * 52361 + int(abs(gamma) * 1000) + 3)
    dlgs = []
    for L in lens:
        if L < 2:
            continue
        oc, oh, op, sc, sh, sp, H_raw = _process_one_dlg_scheme_c_nb(
            int(L), mu, alpha, beta, gamma, H_dist, H_gm, H_sorted, q_sorted, conf, Q_NOISE_STD, 0, K, 5)
        dlgs.append([{"cat": int(oc[m]), "H": float(oh[m]), "plurality": int(op[m])} for m in range(int(L))])
    rh = fit_raw_H(dlgs, seed=rep + 100000)
    wc = fit_within_cell(dlgs, seed=rep)
    return {"rep": rep, "gamma": gamma, "rh": rh["gamma_hat"], "rh_conv": bool(rh["converged"]),
            "wc": wc["gamma_hat"], "wc_conv": bool(wc["converged"])}


def main():
    raw = load_el_raw()
    q_pool, conf, H_dist = empirical_q_pool(raw), compute_empirical_confusion(raw), empirical_H_pool(raw)
    raw_dt = [{"cats": np.array([e["cat"] for e in d], dtype=np.int64),
               "Hs_raw": np.array([e["H"] for e in d])} for d in raw]
    dr = residualize_within_cell(raw_dt)
    hb = global_H_mean(dr)
    fit = fit_dt_bounded(prepare_dt_dlgs(dr, hb), hb, n_restarts=3, seed=0)
    mu, alpha, beta, _ = unpack_v_dt(np.array(fit["v_hat"]), K)
    _warmup_numba(K)
    lens = [len(d) for d in raw]
    common = (lens, mu, np.ascontiguousarray(alpha.reshape(K, K)), float(beta), H_dist, float(H_dist.mean()),
              np.sort(H_dist), np.sort(q_pool)[::-1], conf)
    args = [(r, g) + common for g in GAMMAS for r in range(REPS)]
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(config.n_workers(3)) as pool:
        for res in pool.imap_unordered(worker, args):
            out.append(res)
            (OUT_DIR / "E36b_results.json").write_text(json.dumps({"rows": out}, indent=1))
    summ = {}
    for g in GAMMAS:
        for k in ("rh", "wc"):
            v = np.array([r[k] for r in out if r["gamma"] == g and r[f"{k}_conv"]])
            summ[f"g{g:+.1f}_{k}"] = {"n": int(v.size), "mean": float(v.mean()), "bias": float(v.mean() - g),
                                      "mc_se": float(v.std(ddof=1) / np.sqrt(v.size))}
    (OUT_DIR / "E36b_results.json").write_text(json.dumps({"summary": summ, "rows": out}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
