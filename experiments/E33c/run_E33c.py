"""
E33c: per-corpus diagnostic test (Friends, EmotionPush, M3ED) with the S1 and S2 null simulators.

Real-data gamma_hat (hard-mark, within-cell, full data) per corpus; S1 null distributions with
N_REPS = 200 replications per corpus are simulated here; the S2 rows are the first 200 replications
(rep ids 0..199) of the S2 null of E40 (run E40 first). Decision: reject gamma = 0 if the real
gamma_hat lies outside the 2.5-97.5% band of the null.

Outputs (RESULTS_ROOT/E33c): E33c_ckpt_{friends,emotionpush,m3ed}_S1.json, E33c_results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from corpus_nulls import (  # noqa: E402
    CKPT_INTERVAL, Q_NOISE_STD, load_friends_raw, load_emotionpush_raw, load_m3ed_raw,
    compute_empirical_q_pool, compute_empirical_H_pool, compute_confusion, compute_consec_H_corr,
    compute_gamma_hat, _s1_worker_el, _s1_worker_m3ed, _run_with_progress, analyse_null, sanity_check_s2,
)

OUT_DIR = config.results_dir("E33c")
S2_DIR = config.RESULTS_ROOT / "E40"
N_REPS = 200
N_WORKERS = config.n_workers(4)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "E33c_run.log"), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)


def load_s2(corpus: str) -> list:
    """S2 replications 0..N_REPS-1 for one corpus from the E40 checkpoint."""
    ckpt = S2_DIR / f"E40_ckpt_{corpus}_S2.json"
    if not ckpt.exists():
        raise FileNotFoundError(f"S2 checkpoint not found: {ckpt} (run experiments/E40/run_E40.py)")
    results = json.loads(ckpt.read_text(encoding="utf-8"))["results"]
    results = sorted((r for r in results if r["rep"] < N_REPS), key=lambda r: r["rep"])
    log.info(f"  Loaded {len(results)} S2 reps for {corpus}")
    return results


def _corpus_stats(dlgs, n_ann):
    H_pool = compute_empirical_H_pool(dlgs)
    flat_c = np.concatenate([np.array([e["cat"] for e in d], dtype=np.int64) for d in dlgs])
    flat_H = np.concatenate([np.array([e["H"] for e in d], dtype=np.float64) for d in dlgs])
    starts = np.zeros(len(dlgs) + 1, dtype=np.int64)
    for i, d in enumerate(dlgs):
        starts[i + 1] = starts[i] + len(d)
    return (float(H_pool.mean()), compute_consec_H_corr(dlgs), compute_empirical_q_pool(dlgs, n_ann),
            H_pool, compute_confusion(dlgs), flat_c, flat_H, starts)


def main() -> None:
    t0 = time.time()

    log.info("[1] Loading corpora ...")
    dlgs_friends = load_friends_raw()
    dlgs_emotionpush = load_emotionpush_raw()
    dlgs_m3ed = load_m3ed_raw()
    dlgs_el_wide = dlgs_friends + dlgs_emotionpush
    for name, dlgs in [("Friends", dlgs_friends), ("EmotionPush", dlgs_emotionpush), ("M3ED", dlgs_m3ed)]:
        log.info(f"  {name}: {len(dlgs)} dialogues, {sum(len(d) for d in dlgs)} events")

    log.info("[2] Real-data gamma_hat ...")
    rg_el_wide = compute_gamma_hat(dlgs_el_wide, seed=0)
    rg_friends = compute_gamma_hat(dlgs_friends, seed=0)
    rg_emotionpush = compute_gamma_hat(dlgs_emotionpush, seed=0)
    rg_m3ed = compute_gamma_hat(dlgs_m3ed, seed=0)
    for name, rg in [("EL-wide", rg_el_wide), ("Friends", rg_friends), ("EmotionPush", rg_emotionpush),
                     ("M3ED", rg_m3ed)]:
        log.info(f"  {name}: gamma_hat = {rg['gamma_hat']:+.4f}")

    (H_mean_fr, consec_fr, _, _, conf_fr, fc_fr, fH_fr, starts_fr) = _corpus_stats(dlgs_friends, 5)
    (H_mean_ep, consec_ep, _, _, conf_ep, fc_ep, fH_ep, starts_ep) = _corpus_stats(dlgs_emotionpush, 5)
    (H_mean_m3, consec_m3, q_pool_m3, H_dist_m3,
     conf_m3, fc_m3, fH_m3, starts_m3) = _corpus_stats(dlgs_m3ed, 3)

    # EL-wide q and H pools for the S1 rank mapping of Friends and EmotionPush
    q_pool_el = compute_empirical_q_pool(dlgs_el_wide, n_ann=5)
    H_dist_el = compute_empirical_H_pool(dlgs_el_wide)

    def _l(a):
        return a.tolist() if hasattr(a, "tolist") else list(a)

    s1_fr_args = [(rep, "friends", _l(fc_fr), _l(fH_fr), _l(starts_fr), conf_fr.tolist(),
                   _l(q_pool_el), _l(H_dist_el), 5) for rep in range(N_REPS)]
    s1_ep_args = [(rep, "emotionpush", _l(fc_ep), _l(fH_ep), _l(starts_ep), conf_ep.tolist(),
                   _l(q_pool_el), _l(H_dist_el), 5) for rep in range(N_REPS)]
    s1_m3_args = [(rep, _l(fc_m3), _l(fH_m3), _l(starts_m3), conf_m3.tolist(),
                   _l(q_pool_m3), _l(H_dist_m3), 3) for rep in range(N_REPS)]

    log.info(f"[3] S1 null distributions ({N_REPS} reps x 3 corpora, {N_WORKERS} workers) ...")
    results_s1_fr = _run_with_progress(_s1_worker_el, s1_fr_args, OUT_DIR / "E33c_ckpt_friends_S1.json",
                                       "results", N_WORKERS)
    results_s1_ep = _run_with_progress(_s1_worker_el, s1_ep_args, OUT_DIR / "E33c_ckpt_emotionpush_S1.json",
                                       "results", N_WORKERS)
    results_s1_m3 = _run_with_progress(_s1_worker_m3ed, s1_m3_args, OUT_DIR / "E33c_ckpt_m3ed_S1.json",
                                       "results", N_WORKERS)

    log.info("[4] S2 null distributions (from E40) ...")
    results_s2 = load_s2("friends") + load_s2("emotionpush") + load_s2("m3ed")

    log.info("[5] Aggregating ...")
    all_results = results_s1_fr + results_s1_ep + results_s1_m3 + results_s2

    decisions = {}
    for corpus, rg in [("friends", rg_friends["gamma_hat"]),
                       ("emotionpush", rg_emotionpush["gamma_hat"]),
                       ("m3ed", rg_m3ed["gamma_hat"])]:
        for sim in ["S1", "S2"]:
            d = analyse_null(all_results, rg, corpus, sim)
            decisions[f"{corpus}_{sim}"] = d
            log.info(f"  {corpus} {sim}: real={d['real_gamma']:+.4f}  "
                     f"null=[{d['null_q2_5']:+.4f}, {d['null_q97_5']:+.4f}]  reject={d['reject_gamma0']}")

    sanity = {}
    for corpus, H_mean_real, consec_real in [("friends", H_mean_fr, consec_fr),
                                             ("emotionpush", H_mean_ep, consec_ep),
                                             ("m3ed", H_mean_m3, consec_m3)]:
        sanity[corpus] = sanity_check_s2(all_results, consec_real, H_mean_real, corpus)

    out = {
        "experiment": "E33c_corpus_diagnostic",
        "config": {
            "n_reps": N_REPS,
            "n_workers": N_WORKERS,
            "ckpt_interval": CKPT_INTERVAL,
            "q_noise_std": Q_NOISE_STD,
            "s1_el_q_h_source": "EL-wide (Friends + EmotionPush)",
            "s1_m3ed_q_h_source": "M3ED",
            "s2_source": "E40 checkpoints, rep ids 0-199",
        },
        "real_gammas": {
            "el_wide": rg_el_wide,
            "friends": rg_friends,
            "emotionpush": rg_emotionpush,
            "m3ed": rg_m3ed,
        },
        "real_corpus_stats": {
            "friends": {"H_mean": H_mean_fr, "consec_H_corr": consec_fr},
            "emotionpush": {"H_mean": H_mean_ep, "consec_H_corr": consec_ep},
            "m3ed": {"H_mean": H_mean_m3, "consec_H_corr": consec_m3},
        },
        "decisions": decisions,
        "sanity": sanity,
        "elapsed_sec": float(time.time() - t0),
    }
    (OUT_DIR / "E33c_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    log.info(f"Results -> {OUT_DIR / 'E33c_results.json'}")


if __name__ == "__main__":
    main()
