"""
E27e: hard-mark estimator under gamma = 0 nulls and observation-pipeline power.

Scheme A' (null): the real EL-wide category sequences are kept as truth; votes are re-simulated with
    independent accuracy (indep) or accuracy linked to the item's entropy (linked); the estimator sees the
    observed majority label and the observed vote entropy. Within-cell (wc) and raw-H (rh) fits, plurality
    filters >= 1, 3, 4 of 5. N_REPS_A replications.
Scheme C' (power): dialogues generated from the DT-AMHP fitted to EL-wide with injected gamma in
    {0, -0.4, -0.8}, latent H and linked votes; the oracle fit uses true categories and latent H, the
    observed fit uses majority labels and vote entropy. N_REPS_C replications per gamma.

Outputs (RESULTS_ROOT/E27e): E27e_partial_schemeA.json, E27e_partial_schemeC.json, E27e_results.json
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from dtsim_core import (  # noqa: E402
    K, load_el_raw, empirical_q_pool, compute_empirical_confusion, empirical_H_pool,
    residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded, unpack_v_dt,
)
from dtsim_fits import (  # noqa: E402
    N_REPS_A, N_REPS_C, N_RESTARTS, N_REFINE, GAMMA_CONDITIONS, MIN_PL_LIST, Q_NOISE_STD, REAL_GAMMA,
    fit_raw_H, fit_within_cell, _conv_vals, _conv_vals_c, _summarise, _emp_p,
)
from dtsim_kernels import (  # noqa: E402
    _seed_nb, _sample_q_indep_nb, _compute_q_linked_nb, _simulate_votes_all_events_nb,
    _process_one_dlg_scheme_c_nb, _warmup_numba,
)

OUT_DIR = config.results_dir("E27e")
N_WORKERS = config.n_workers(10)
PROGRESS_INTERVAL = 10

# Real-data raw-H gamma_hat of the hard-mark estimator per plurality filter
REAL_GAMMA_RAW_DT = {1: -1.0742, 3: -1.1117, 4: -1.1523}


# ============================================================================
# Workers
# ============================================================================

def _scheme_a_prime_worker_e(args: tuple) -> dict:
    """One Scheme A' replication (both accuracy models, all filters, wc and rh fits)."""
    (rep, q_pool_arr, confusion_arr, flat_cats, flat_H, dlg_starts,
     H_sorted_arr, q_sorted_arr, Q_NOISE_STD_val) = args

    K_val = 7
    N_ANN = 5
    N_ev = len(flat_cats)
    n_dlg = len(dlg_starts) - 1

    _seed_nb(rep * 41117 + 7)
    q_vals_i = _sample_q_indep_nb(N_ev, q_pool_arr)
    obs_cats_i, H_obs_i, plur_i, flip_i = _simulate_votes_all_events_nb(
        flat_cats, q_vals_i, confusion_arr, K_val, N_ANN)

    _seed_nb(rep * 41117 + 10_000_007)
    q_vals_l = _compute_q_linked_nb(flat_H, H_sorted_arr, q_sorted_arr,
                                    Q_NOISE_STD_val)
    obs_cats_l, H_obs_l, plur_l, flip_l = _simulate_votes_all_events_nb(
        flat_cats, q_vals_l, confusion_arr, K_val, N_ANN)

    indep_dlgs: List[List[dict]] = []
    linked_dlgs: List[List[dict]] = []
    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        indep_dlgs.append([
            {"cat": int(obs_cats_i[k]), "H": float(H_obs_i[k]),
             "plurality": int(plur_i[k])}
            for k in range(s, e)
        ])
        linked_dlgs.append([
            {"cat": int(obs_cats_l[k]), "H": float(H_obs_l[k]),
             "plurality": int(plur_l[k])}
            for k in range(s, e)
        ])

    results: dict = {}
    for model_tag, dlgs in [("indep", indep_dlgs), ("linked", linked_dlgs)]:
        results[model_tag] = {}
        for min_pl in MIN_PL_LIST:
            filtered = [
                [ev for ev in dlg if ev["plurality"] >= min_pl]
                for dlg in dlgs
            ]
            filtered = [d for d in filtered if len(d) >= 2]

            if len(filtered) < 5:
                results[model_tag][min_pl] = {
                    "wc": {"gamma_hat": float("nan"), "converged": False},
                    "rh": {"gamma_hat": float("nan"), "converged": False},
                    "n_dlg": len(filtered),
                }
                continue

            rh = fit_raw_H(filtered, seed=rep)
            wc = fit_within_cell(filtered, seed=rep + 50000)

            results[model_tag][min_pl] = {
                "wc": {"gamma_hat": wc["gamma_hat"],
                       "converged": wc["converged"],
                       "at_bound": wc["at_bound"]},
                "rh": {"gamma_hat": rh["gamma_hat"],
                       "converged": rh["converged"],
                       "at_bound": rh["at_bound"]},
                "n_dlg": len(filtered),
            }

    return {
        "rep": rep,
        "results": results,
        "flip_frac_i": float(flip_i) / N_ev,
        "flip_frac_l": float(flip_l) / N_ev,
    }


def _scheme_c_prime_worker_e(args: tuple) -> dict:
    """One Scheme C' replication at one injected gamma (oracle and observed fits, all filters)."""
    (rep, mu_arr, alpha_arr, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr, dlg_lengths,
     H_sorted_arr, q_sorted_arr, Q_NOISE_STD_val) = args

    K_val = 7
    N_ANN = 5
    alpha_mat = alpha_arr.reshape(K_val, K_val)

    g_seed = int(round((gamma_inj + 2.0) * 10000))
    _seed_nb(rep * 91723 + g_seed * 1000 + 77)

    oracle_dlgs: List[List[dict]] = []
    obs_dlgs: List[List[dict]] = []
    H_raw_all: List[float] = []
    H_obs_all: List[float] = []

    for dlg_len in dlg_lengths:
        if dlg_len < 2:
            continue

        (oc, oh, op, sc, sh, sp, H_raw) = _process_one_dlg_scheme_c_nb(
            dlg_len, mu_arr, alpha_mat, float(beta_val), gamma_inj,
            H_dist_arr, H_gm_val, H_sorted_arr, q_sorted_arr,
            confusion_arr, Q_NOISE_STD_val, N_REFINE, K_val, N_ANN,
        )

        oracle_dlgs.append([
            {"cat": int(oc[m]), "H": float(oh[m]), "plurality": int(op[m])}
            for m in range(dlg_len)
        ])
        obs_dlgs.append([
            {"cat": int(sc[m]), "H": float(sh[m]), "plurality": int(sp[m])}
            for m in range(dlg_len)
        ])
        for m in range(dlg_len):
            H_raw_all.append(float(H_raw[m]))
            H_obs_all.append(float(sh[m]))

    # correlation between latent and observed entropy
    corr_val = 0.0
    if len(H_raw_all) > 2:
        arr_r = np.array(H_raw_all)
        arr_o = np.array(H_obs_all)
        if arr_r.std() > 1e-9 and arr_o.std() > 1e-9:
            from scipy.stats import pearsonr
            corr_val, _ = pearsonr(arr_r, arr_o)

    res: dict = {}
    for model_tag, dlgs in [("oracle", oracle_dlgs), ("obs", obs_dlgs)]:
        res[model_tag] = {}
        for min_pl in MIN_PL_LIST:
            filtered = [
                [ev for ev in dlg if ev["plurality"] >= min_pl]
                for dlg in dlgs
            ]
            filtered = [d for d in filtered if len(d) >= 2]
            if len(filtered) < 5:
                res[model_tag][min_pl] = {
                    "gamma_hat": float("nan"), "converged": False, "n_dlg": 0}
                continue
            wc = fit_within_cell(filtered, seed=rep)
            res[model_tag][min_pl] = {
                "gamma_hat": wc["gamma_hat"],
                "converged": wc["converged"],
                "at_bound": wc["at_bound"],
                "n_dlg": len(filtered),
            }

    return {
        "rep": rep,
        "gamma_inj": gamma_inj,
        "corr": float(corr_val),
        "results": res,
    }


# ============================================================================
# Pool runner with resumable checkpoints
# ============================================================================

def _run_with_progress(
    worker_fn,
    args_list: list,
    checkpoint_path: Path,
    key_name: str,
    rep_key: str = "rep",
    n_workers: int = N_WORKERS,
) -> list:
    """Run worker_fn over args_list (args[0] = rep id); skip reps already in the checkpoint."""
    done_reps: set = set()
    results: list = []
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_reps = {r[rep_key] for r in results}
            print(f"  Checkpoint: {len(done_reps)} reps already done "
                  f"({checkpoint_path.name})", flush=True)
        except Exception as ex:
            print(f"  Warning: could not load checkpoint ({ex}); starting fresh",
                  flush=True)

    pending = [a for a in args_list if a[0] not in done_reps]
    n_total = len(args_list)

    if not pending:
        print(f"  All {n_total} reps found in checkpoint.", flush=True)
        return results

    print(f"  {len(pending)} reps pending on {n_workers} workers ...",
          flush=True)

    ctx = mp.get_context("spawn")
    unsaved_since_ckpt = 0

    def _save():
        checkpoint_path.write_text(
            json.dumps({key_name: results}, ensure_ascii=False,
                       default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
            encoding="utf-8",
        )

    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pending):
            results.append(result)
            unsaved_since_ckpt += 1
            if unsaved_since_ckpt >= PROGRESS_INTERVAL:
                print(f"  [{time.strftime('%H:%M:%S')}] {len(results)}/{n_total} reps done", flush=True)
                _save()
                unsaved_since_ckpt = 0

    _save()
    print(f"  Final checkpoint saved -> {checkpoint_path}", flush=True)
    return results


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    t0_main = time.time()

    print("[1] Loading EL-wide data ...", flush=True)
    raw_dlgs = load_el_raw()
    q_pool = empirical_q_pool(raw_dlgs)
    confusion = compute_empirical_confusion(raw_dlgs)
    H_dist = empirical_H_pool(raw_dlgs)
    dlg_lengths = [len(d) for d in raw_dlgs]
    n_dlg = len(raw_dlgs)
    print(f"    {n_dlg} dialogues, {sum(dlg_lengths)} events", flush=True)

    flat_cats = np.concatenate(
        [np.array([e["cat"] for e in d], dtype=np.int64) for d in raw_dlgs])
    flat_H = np.concatenate(
        [np.array([e["H"] for e in d], dtype=np.float64) for d in raw_dlgs])
    dlg_starts = np.zeros(n_dlg + 1, dtype=np.int64)
    for i, d in enumerate(raw_dlgs):
        dlg_starts[i + 1] = dlg_starts[i] + len(d)

    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]

    print("[2] Real-data within-cell DT fit (generator parameters) ...", flush=True)
    raw_dt = [{"cats": np.array([e["cat"] for e in d], dtype=np.int64),
               "Hs_raw": np.array([e["H"] for e in d], dtype=np.float64)}
              for d in raw_dlgs]
    dlgs_r = residualize_within_cell(raw_dt)
    H_bar_r = global_H_mean(dlgs_r)
    dlgs_dt_r = prepare_dt_dlgs(dlgs_r, H_bar_r)
    real_fit = fit_dt_bounded(dlgs_dt_r, H_bar_r, n_restarts=3, seed=0)
    print(f"    gamma_hat = {real_fit['gamma_hat']:+.4f}", flush=True)

    if real_fit["v_hat"]:
        v = np.array(real_fit["v_hat"])
        mu_r, alpha_r, beta_r, _ = unpack_v_dt(v, K)
    else:
        mu_r = np.ones(K) / K * 0.2
        alpha_r = np.ones((K, K)) * 0.05
        beta_r = 1.0

    H_gm_val = float(H_dist.mean())

    print("[3] numba JIT warm-up ...", flush=True)
    _warmup_numba(K)

    a_args = [
        (rep, q_pool, confusion, flat_cats, flat_H, dlg_starts,
         H_sorted, q_sorted, Q_NOISE_STD)
        for rep in range(N_REPS_A)
    ]

    c_args = [
        (rep, mu_r, alpha_r.ravel(), float(beta_r), float(g),
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD)
        for rep in range(N_REPS_C)
        for g in GAMMA_CONDITIONS
    ]

    print(f"[4] Scheme C': {N_REPS_C} x {len(GAMMA_CONDITIONS)} reps ...", flush=True)
    scheme_c_raw = _run_with_progress(
        _scheme_c_prime_worker_e, c_args, OUT_DIR / "E27e_partial_schemeC.json", "scheme_c_prime")

    print(f"[5] Scheme A': {N_REPS_A} reps ...", flush=True)
    scheme_a_raw = _run_with_progress(
        _scheme_a_prime_worker_e, a_args, OUT_DIR / "E27e_partial_schemeA.json", "scheme_a_prime")

    print("[6] Aggregating ...", flush=True)
    agg: dict = {}
    flip_is = [r["flip_frac_i"] for r in scheme_a_raw
               if not math.isnan(r["flip_frac_i"])]
    flip_ls = [r["flip_frac_l"] for r in scheme_a_raw
               if not math.isnan(r["flip_frac_l"])]
    flip_fracs = {
        "flip_frac_i": float(np.mean(flip_is)) if flip_is else float("nan"),
        "flip_frac_l": float(np.mean(flip_ls)) if flip_ls else float("nan"),
    }

    real_gamma_by_pl = {1: REAL_GAMMA["full"], 3: REAL_GAMMA["p3_5"],
                        4: REAL_GAMMA["p4_5"]}

    for fit_prefix, fit_type in [("wc", "wc"), ("rh", "rh")]:
        for model in ["indep", "linked"]:
            for min_pl in MIN_PL_LIST:
                vals = _conv_vals(scheme_a_raw, model, min_pl, fit_type)
                ref_g = real_gamma_by_pl[min_pl] if fit_type == "wc" else REAL_GAMMA_RAW_DT[min_pl]
                key = f"{fit_prefix}_full_{model}_pl{min_pl}_{fit_type}"
                agg[key] = _summarise(vals, ref_g)
                agg[f"emp_p_{fit_prefix}_full_{model}_pl{min_pl}_{fit_type}"] = \
                    _emp_p(vals, ref_g)

    corr_c_vals = [r["corr"] for r in scheme_c_raw
                   if not math.isnan(r.get("corr", float("nan")))]
    mean_corr_c = float(np.mean(corr_c_vals)) if corr_c_vals else float("nan")

    for model in ["oracle", "obs"]:
        for min_pl in MIN_PL_LIST:
            for g in GAMMA_CONDITIONS:
                vals = _conv_vals_c(scheme_c_raw, model, min_pl, g)
                gk = f"g{str(g).replace('-', 'm').replace('.', 'p')}"
                key = f"obs_{model}_pl{min_pl}_{gk}"
                agg[key] = _summarise(vals, g)

    elapsed = time.time() - t0_main
    out = {
        "experiment": "E27e",
        "config": {
            "N_REPS_A": N_REPS_A,
            "N_REPS_C": N_REPS_C,
            "N_RESTARTS": N_RESTARTS,
            "N_REFINE": N_REFINE,
            "Q_NOISE_STD": Q_NOISE_STD,
            "N_WORKERS_E": N_WORKERS,
        },
        "real_gamma": REAL_GAMMA,
        "scheme_A_prime": agg,
        "scheme_C_prime": {k: v for k, v in agg.items() if k.startswith("obs_")},
        "flip_fracs": flip_fracs,
        "mean_corr_C": mean_corr_c,
        "elapsed_sec": float(elapsed),
    }

    out_path = OUT_DIR / "E27e_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                   default=lambda o: o.item() if hasattr(o, 'item') else str(o)),
                        encoding="utf-8")
    print(f"Results -> {out_path}  ({elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
