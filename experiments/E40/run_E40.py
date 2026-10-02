"""
E40: per-corpus S2 null distributions (fitted AR(1) annotator model) with B = 2000 replications.

Same simulator and seeds as the S2 rows of E33c (rep ids 0..199 of this run are those rows).
p-value: median-centred two-sided, p = (1 + #{|g_b - med| >= |g_real - med|}) / (B + 1).
Requires E30_fit.json and E30b_m3ed.json (RESULTS_ROOT/E30).

Outputs (RESULTS_ROOT/E40): E40_ckpt_{friends,emotionpush,m3ed}_S2.json, E40_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from corpus_nulls import (  # noqa: E402
    load_friends_raw, load_emotionpush_raw, load_m3ed_raw, compute_gamma_hat,
    _s2_worker_el, _s2_worker_m3ed, _run_with_progress,
)

OUT_DIR = config.results_dir("E40")
E30_DIR = config.RESULTS_ROOT / "E30"
N_REPS = 2000
N_WORKERS = config.n_workers(3)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "E40_run.log"), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


def _flat(dlgs):
    flat_c = np.concatenate([np.array([e["cat"] for e in d], dtype=np.int64) for d in dlgs])
    starts = np.zeros(len(dlgs) + 1, dtype=np.int64)
    for i, d in enumerate(dlgs):
        starts[i + 1] = starts[i] + len(d)
    return flat_c.tolist(), starts.tolist()


def main() -> None:
    log.info("[1] Loading corpora ...")
    dlgs = {"friends": load_friends_raw(), "emotionpush": load_emotionpush_raw(), "m3ed": load_m3ed_raw()}

    log.info("[2] Real-data gamma_hat ...")
    real = {c: compute_gamma_hat(d, seed=0)["gamma_hat"] for c, d in dlgs.items()}
    for c, g in real.items():
        log.info(f"  {c}: {g:+.4f}")

    log.info("[3] AR(1) annotator parameters ...")
    e30_fit = json.loads((E30_DIR / "E30_fit.json").read_text())
    e30b = json.loads((E30_DIR / "E30b_m3ed.json").read_text())
    ar1 = {
        "friends": (e30_fit["el_friends"]["a"], e30_fit["el_friends"]["b"],
                    e30_fit["el_friends"]["rho"], e30_fit["el_friends"]["confusion"]),
        "emotionpush": (e30_fit["el_emotionpush"]["a"], e30_fit["el_emotionpush"]["b"],
                        e30_fit["el_emotionpush"]["rho"], e30_fit["el_emotionpush"]["confusion"]),
        # M3ED: (a, b, rho) from the direct maximisation, confusion from the EM fit
        "m3ed": (e30b["best"]["a"], e30b["best"]["b"], e30b["best"]["rho"], e30_fit["m3ed"]["confusion"]),
    }

    log.info(f"[4] S2 null distributions ({N_REPS} reps x 3 corpora, {N_WORKERS} workers) ...")
    results = {}
    for corpus in ("friends", "emotionpush", "m3ed"):
        fc, starts = _flat(dlgs[corpus])
        a, b, rho, conf = ar1[corpus]
        if corpus == "m3ed":
            args = [(rep, fc, starts, conf, a, b, rho, 3) for rep in range(N_REPS)]
            worker = _s2_worker_m3ed
        else:
            args = [(rep, corpus, fc, starts, conf, a, b, rho, 5) for rep in range(N_REPS)]
            worker = _s2_worker_el
        results[corpus] = _run_with_progress(worker, args, OUT_DIR / f"E40_ckpt_{corpus}_S2.json",
                                             "results", N_WORKERS)

    summ = {}
    for corpus, res in results.items():
        g_real = real[corpus]
        v = np.array([r["gamma_hat"] for r in res if r.get("converged", True) and np.isfinite(r["gamma_hat"])])
        med = float(np.median(v))
        k = int((np.abs(v - med) >= abs(g_real - med)).sum())
        pv = (1 + k) / (len(v) + 1)
        summ[corpus] = {"real": float(g_real), "B": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                        "q2_5": float(np.quantile(v, .025)), "q97_5": float(np.quantile(v, .975)), "p_two_sided": pv,
                        "reject": bool(not (np.quantile(v, .025) <= g_real <= np.quantile(v, .975)))}
    (OUT_DIR / "E40_summary.json").write_text(json.dumps(summ, indent=1))
    log.info(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
