"""
E34f (exploratory): decomposition of the over-shoot of the latent-mark correction at gamma = 0,
same data-generating process as E34e (30 replications). Variants fitted on the same simulated data:
  A corr_post_H  correction as in E34e (history = posterior expected marks, modifier = vote entropy H)
  B corr_true_H  history uses the true category (one-hot), current event marginalised, modifier H
  C corr_post_u  as A but modifier = latent difficulty u (within observed cells)
  D hard_u       hard-mark estimator with modifier u
  E hard_H       hard-mark estimator with modifier H (= E34e hard)
If B removes the over-shoot, the plug-in posterior history causes it; if C removes it, the H-versus-u
modifier mismatch causes it. Requires RESULTS_ROOT/E29/E29_fit.json.

Output (RESULTS_ROOT/E34f): E34f_results.json
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

import latent_mark as lm  # noqa: E402
from latent_mark import (fit_ds_fixed_c, fit_within_cell, _compute_ds_posterior_nb, _within_cell_H,  # noqa: E402
                         U_GH, W_GH, N_GH)

OUT = config.results_dir("E34f")
REPS = 30


def corr_data(dlgs, ann, history, modifier):
    conf = np.ascontiguousarray(ann["confusion"], dtype=np.float64)
    prior = np.ascontiguousarray(ann["prior"], dtype=np.float64)
    mods = []
    for dlg in dlgs:
        cats = np.array([x["cat"] for x in dlg], dtype=np.int64)
        val = np.array([x["H"] if modifier == "H" else x["u"] for x in dlg], dtype=np.float64)
        mods.append(_within_cell_H(cats, val))
    hbar = float(np.concatenate(mods).mean())
    data = []
    for dlg, mod in zip(dlgs, mods):
        M = len(dlg)
        post = np.zeros((M, lm.K_EL)); ll = np.zeros((M, lm.K_EL))
        for m, x in enumerate(dlg):
            pm, lmarg = _compute_ds_posterior_nb(np.ascontiguousarray(x["votes"], dtype=np.float64),
                                                 ann["a"], ann["b"], conf, prior, lm.N_ANN_EL, lm.K_EL,
                                                 U_GH, W_GH, N_GH)
            ll[m] = lmarg
            if history == "post":
                post[m] = pm
            else:
                post[m, x["true_cat"]] = 1.0
        data.append((post, ll, (mod - hbar).astype(np.float64)))
    return data


def worker(args):
    rep, dlg_lens, mu, alpha, beta, ann = args
    dlgs = lm.simulate_matched(rep, 0.0, dlg_lens, mu, alpha, beta, ann)
    res = {"rep": rep}
    for name, hist, mod in (("A_corr_post_H", "post", "H"), ("B_corr_true_H", "true", "H"), ("C_corr_post_u", "post", "u")):
        r = fit_ds_fixed_c(corr_data(dlgs, ann, hist, mod), seed=rep + 5000)
        res[name] = (float(r["gamma_hat"]), bool(r["converged"]))
    for name, key in (("D_hard_u", "u"), ("E_hard_H", "H")):
        ev = [[{"cat": x["cat"], "H": x[key], "plurality": x["plurality"]} for x in dlg] for dlg in dlgs]
        r = fit_within_cell(ev, seed=rep)
        res[name] = (float(r["gamma_hat"]), bool(r["converged"]))
    return res


def main():
    lens, mu, alpha, beta = lm.fit_el_wide_generator()
    ann = lm.load_annotator_params()
    lm._warmup_numba()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(config.n_workers(3)) as pool:
        for r in pool.imap_unordered(worker, [(rep, lens, mu, alpha, beta, ann) for rep in range(REPS)]):
            rows.append(r)
            (OUT / "E34f_results.json").write_text(json.dumps({"rows": rows}, indent=1))
            print(len(rows), {k: round(v[0], 3) for k, v in r.items() if k != "rep"}, flush=True)
    summ = {}
    for k in ("A_corr_post_H", "B_corr_true_H", "C_corr_post_u", "D_hard_u", "E_hard_H"):
        v = np.array([r[k][0] for r in rows if r[k][1]])
        summ[k] = {"n": int(v.size), "mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size))}
    (OUT / "E34f_results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
