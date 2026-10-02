"""
E34e: latent-mark correction under the matched data-generating process (src/latent_mark.py).

One latent difficulty u drives both the excitation modifier and the annotator accuracy; both estimators see
only the votes (observed majority label and vote entropy):
  hard     hard-mark within-cell estimator
  ds_corr  latent-mark corrected estimator with the true annotator parameters
gamma in {0, -0.4}; 40 replications at gamma = 0 and 30 at gamma = -0.4.
Requires RESULTS_ROOT/E29/E29_fit.json.

Outputs (RESULTS_ROOT/E34e): E34e_results.json, E34e_summary.json
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

import latent_mark as lm  # noqa: E402

OUT = config.results_dir("E34e")
N_REPS = {0.0: 40, -0.4: 30}


def worker(args):
    rep, gamma, dlg_lens, mu, alpha, beta, ann = args
    dlgs = lm.simulate_matched(rep, gamma, dlg_lens, mu, alpha, beta, ann)
    return lm._fit_both(rep, dlgs, lm.N_ANN_EL, ann, f"matched_g{gamma:+.1f}", gamma)


def main():
    dlg_lens, mu, alpha, beta = lm.fit_el_wide_generator()
    ann = lm.load_annotator_params()
    print(f"generator beta={beta:.4f}; annotator a={ann['a']:.3f} b={ann['b']:.3f}", flush=True)
    lm._warmup_numba()
    ctx = mp.get_context("spawn")
    results = []
    t0 = time.time()
    args = [(r, g, dlg_lens, mu, alpha, beta, ann) for g, n in N_REPS.items() for r in range(n)]
    with ctx.Pool(config.n_workers(3)) as pool:
        for res in pool.imap_unordered(worker, args):
            results.append(res)
            (OUT / "E34e_results.json").write_text(json.dumps(results, default=float, indent=1))
            print(f"{len(results)}/{len(args)} done  {time.time() - t0:.0f}s  "
                  f"{res['label']} hard={res['hard']['gamma_hat']:+.3f} corr={res['ds_corr']['gamma_hat']:+.3f}",
                  flush=True)
    summary = {}
    for g in N_REPS:
        for est in ("hard", "ds_corr"):
            v = np.array([r[est]["gamma_hat"] for r in results if r["gamma_inj"] == g and r[est]["converged"]])
            summary[f"g{g:+.1f}_{est}"] = {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                                           "se": float(v.std(ddof=1) / np.sqrt(v.size))}
    (OUT / "E34e_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
