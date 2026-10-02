"""
E28: soft-mark estimator under the Scheme A' nulls of E27e (gamma = 0).

Votes are re-simulated exactly as in E27e Scheme A' (same seeds, independent and linked accuracy), and
the simulated vote distribution p = votes / 5 is the excitation source of the soft-mark estimator; the
likelihood target is the observed majority label. Within-cell (wc) and raw-H (rh) modifiers, plurality
filters >= 1, 3, 4 of 5. Real-data soft-mark gamma_hat is computed for every setting.

Outputs (RESULTS_ROOT/E28): E28_validation.json, E28_partial.json, E28_results.json
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from dtsim_core import (  # noqa: E402
    K, load_el_raw, empirical_q_pool, compute_empirical_confusion, empirical_H_pool,
)
from dtsim_fits import MIN_PL_LIST, Q_NOISE_STD  # noqa: E402
from dtsim_kernels import _seed_nb, _sample_q_indep_nb, _compute_q_linked_nb  # noqa: E402
from estimator_softmark import SoftMarkEstimator, gradient_check_softmark  # noqa: E402

N_REPS = 300
N_WORKERS = config.n_workers(6)
N_ANN = 5
N_RESTARTS_SOFT = 1          # restarts for the simulated fits
N_RESTARTS_REAL = 5          # restarts for the real-data fits
MAXITER_NULL = 3000
MAXITER_REAL = 5000
CHECKPOINT_INT = 50

OUT_DIR = config.results_dir("E28")


def _setup_log():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(OUT_DIR / "E28_run.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


# =============================================================================
# Vote simulation that also returns the vote distribution
# =============================================================================

@njit(cache=True)
def _simulate_votes_with_pdist_nb(
    all_true_cats: np.ndarray,
    all_q_vals: np.ndarray,
    confusion: np.ndarray,
    K: int,
    n_ann: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    As dtsim_kernels._simulate_votes_all_events_nb, additionally returning p_dist (N, K), the
    normalised vote distribution of each event. Returns (obs_cats, H_obs, plur, p_dist, flip_count).
    """
    N = len(all_true_cats)
    obs_cats = np.empty(N, dtype=np.int64)
    H_obs_arr = np.empty(N, dtype=np.float64)
    plur_arr = np.empty(N, dtype=np.int64)
    p_dist_arr = np.empty((N, K), dtype=np.float64)
    flip_count = 0

    for idx in range(N):
        tc = all_true_cats[idx]
        q = all_q_vals[idx]

        votes = np.zeros(K, dtype=np.float64)
        for _ in range(n_ann):
            if np.random.random() < q:
                votes[tc] += 1.0
            else:
                r2 = np.random.random()
                cum = 0.0
                chosen = K - 1
                for j in range(K):
                    cum += confusion[tc, j]
                    if r2 <= cum:
                        chosen = j
                        break
                votes[chosen] += 1.0

        if votes.sum() == 0.0:
            votes[tc] = float(n_ann)

        total = votes.sum()

        for j in range(K):
            p_dist_arr[idx, j] = votes[j] / total

        max_v = 0.0
        for j in range(K):
            if votes[j] > max_v:
                max_v = votes[j]

        n_tied = 0
        for j in range(K):
            if votes[j] == max_v:
                n_tied += 1

        r3 = np.random.random()
        obs_cat = 0
        cum_p = 0.0
        for j in range(K):
            if votes[j] == max_v:
                cum_p += 1.0 / n_tied
                if r3 <= cum_p:
                    obs_cat = j
                    break
        obs_cats[idx] = obs_cat

        if obs_cat != tc:
            flip_count += 1

        h_val = 0.0
        for j in range(K):
            p_j = votes[j] / total
            if p_j > 1e-12:
                h_val -= p_j * math.log(p_j + 1e-12)
        H_obs_arr[idx] = h_val
        plur_arr[idx] = int(max_v)

    return obs_cats, H_obs_arr, plur_arr, p_dist_arr, flip_count


# =============================================================================
# Soft-mark inputs and fits
# =============================================================================

def _build_soft_wc_threads(
    dlgs_evs: List[List[dict]],
    K_val: int,
) -> Tuple[List[dict], float]:
    """Soft-mark threads with within-cell residualised H (size-1 cells set to 0), globally centred."""
    intermediates = []
    for evs in dlgs_evs:
        cats = np.array([e["cat"] for e in evs], dtype=np.int64)
        H_raw = np.array([e["H"] for e in evs], dtype=np.float64)
        p_d = np.array([e["p_dist"] for e in evs], dtype=np.float64)

        H_wc = H_raw.copy()
        for c in range(K_val):
            idx = np.where(cats == c)[0]
            if len(idx) > 1:
                H_wc[idx] -= H_raw[idx].mean()
            elif len(idx) == 1:
                H_wc[idx[0]] = 0.0

        intermediates.append((cats, H_wc, p_d))

    all_H_wc = np.concatenate([x[1] for x in intermediates])
    H_bar = float(all_H_wc.mean())

    threads = []
    for cats, H_wc, p_d in intermediates:
        n = len(cats)
        threads.append({
            "times_h": np.arange(n, dtype=np.float64),
            "cats": cats,
            "Hs_c": H_wc - H_bar,
            "p_dist": p_d,
            "s_vals": np.zeros(n),
            "T": float(n),
            "s_int": 0.0,
        })

    return threads, H_bar


def _build_soft_rh_threads(
    dlgs_evs: List[List[dict]],
) -> Tuple[List[dict], float]:
    """Soft-mark threads with globally centred H."""
    all_H = [e["H"] for dlg in dlgs_evs for e in dlg]
    H_bar = float(np.mean(all_H)) if all_H else 0.0

    threads = []
    for evs in dlgs_evs:
        n = len(evs)
        threads.append({
            "times_h": np.arange(n, dtype=np.float64),
            "cats": np.array([e["cat"] for e in evs], dtype=np.int64),
            "Hs_c": np.array([e["H"] for e in evs], dtype=np.float64) - H_bar,
            "p_dist": np.array([e["p_dist"] for e in evs], dtype=np.float64),
            "s_vals": np.zeros(n),
            "T": float(n),
            "s_int": 0.0,
        })

    return threads, H_bar


def _fit_soft(
    threads: List[dict],
    H_bar: float,
    seed: int,
    n_restarts: int = N_RESTARTS_SOFT,
    maxiter: int = MAXITER_NULL,
) -> dict:
    K_val = threads[0]["cats"].max().item() + 1 if threads else K
    K_val = max(K_val, K)
    est = SoftMarkEstimator(threads, K=K_val, H_bar=H_bar, l1_alpha=0.001)
    r = est.fit(n_restarts=n_restarts, maxiter=maxiter, seed=seed)
    gamma = float(r.gamma_hat)
    return {
        "gamma_hat": gamma,
        "converged": bool(r.success) and math.isfinite(gamma),
        "neg_loglik": float(r.neg_loglik),
    }


def fit_soft_wc(dlgs_evs: List[List[dict]], seed: int,
                n_restarts: int = N_RESTARTS_SOFT, maxiter: int = MAXITER_NULL) -> dict:
    threads, H_bar = _build_soft_wc_threads(dlgs_evs, K)
    return _fit_soft(threads, H_bar, seed, n_restarts, maxiter)


def fit_soft_rh(dlgs_evs: List[List[dict]], seed: int,
                n_restarts: int = N_RESTARTS_SOFT, maxiter: int = MAXITER_NULL) -> dict:
    threads, H_bar = _build_soft_rh_threads(dlgs_evs)
    return _fit_soft(threads, H_bar, seed, n_restarts, maxiter)


# =============================================================================
# Worker
# =============================================================================

def _e28_worker(args: tuple) -> dict:
    """One replication: simulate votes (indep and linked accuracy) and fit wc / rh per filter."""
    (rep, q_pool_arr, confusion_arr, flat_cats, flat_H, dlg_starts,
     H_sorted_arr, q_sorted_arr, Q_NOISE_STD_val) = args

    K_val = 7
    N_ev = len(flat_cats)
    n_dlg = len(dlg_starts) - 1

    # same seeds as E27e Scheme A'
    _seed_nb(rep * 41117 + 7)
    q_vals_i = _sample_q_indep_nb(N_ev, q_pool_arr)
    obs_cats_i, H_obs_i, plur_i, pdist_i, flip_i = _simulate_votes_with_pdist_nb(
        flat_cats, q_vals_i, confusion_arr, K_val, N_ANN)

    _seed_nb(rep * 41117 + 10_000_007)
    q_vals_l = _compute_q_linked_nb(flat_H, H_sorted_arr, q_sorted_arr,
                                    Q_NOISE_STD_val)
    obs_cats_l, H_obs_l, plur_l, pdist_l, flip_l = _simulate_votes_with_pdist_nb(
        flat_cats, q_vals_l, confusion_arr, K_val, N_ANN)

    indep_dlgs: List[List[dict]] = []
    linked_dlgs: List[List[dict]] = []
    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        indep_dlgs.append([
            {"cat": int(obs_cats_i[k]), "H": float(H_obs_i[k]),
             "plurality": int(plur_i[k]), "p_dist": pdist_i[k]}
            for k in range(s, e)
        ])
        linked_dlgs.append([
            {"cat": int(obs_cats_l[k]), "H": float(H_obs_l[k]),
             "plurality": int(plur_l[k]), "p_dist": pdist_l[k]}
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

            wc = fit_soft_wc(filtered, seed=rep)
            rh = fit_soft_rh(filtered, seed=rep + 50000)

            results[model_tag][min_pl] = {
                "wc": {"gamma_hat": wc["gamma_hat"], "converged": wc["converged"]},
                "rh": {"gamma_hat": rh["gamma_hat"], "converged": rh["converged"]},
                "n_dlg": len(filtered),
            }

    return {
        "rep": rep,
        "results": results,
        "flip_frac_i": float(flip_i) / N_ev,
        "flip_frac_l": float(flip_l) / N_ev,
    }


def _warmup_numba_e28(K_val: int = 7) -> None:
    dummy_cats = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    dummy_q = np.array([0.7, 0.8, 0.6, 0.7, 0.7])
    dummy_conf = np.zeros((K_val, K_val))
    for i in range(K_val):
        for j in range(K_val):
            dummy_conf[i, j] = 0.0 if i == j else 1.0 / (K_val - 1)
    dummy_pool = np.array([0.6, 0.7, 0.8, 0.9])
    dummy_H = np.array([0.3, 0.8, 0.1, 0.6, 0.4])

    _seed_nb(42)
    _sample_q_indep_nb(5, dummy_pool)
    _compute_q_linked_nb(dummy_H, np.sort(dummy_H), np.sort(dummy_pool)[::-1], 0.05)
    _simulate_votes_with_pdist_nb(dummy_cats, dummy_q, dummy_conf, K_val, 5)


# =============================================================================
# Pool runner with resumable checkpoints
# =============================================================================

def _run_with_progress(
    worker_fn,
    args_list: list,
    checkpoint_path: Path,
    key_name: str = "scheme_a_prime",
    n_workers: int = N_WORKERS,
) -> list:
    done_reps: set = set()
    results: list = []

    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_reps = {r["rep"] for r in results}
            logging.info(f"  Checkpoint: {len(done_reps)} reps already done.")
        except Exception as ex:
            logging.warning(f"  Could not load checkpoint ({ex}); starting fresh.")

    pending = [a for a in args_list if a[0] not in done_reps]
    n_total = len(args_list)

    if not pending:
        logging.info(f"  All {n_total} reps already done.")
        return results

    logging.info(f"  {len(pending)} reps pending on {n_workers} workers ...")

    def _save():
        checkpoint_path.write_text(
            json.dumps({key_name: results}, ensure_ascii=False,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8",
        )

    ctx = mp.get_context("spawn")
    unsaved = 0
    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pending):
            results.append(result)
            unsaved += 1
            if unsaved >= CHECKPOINT_INT:
                logging.info(f"  [{time.strftime('%H:%M:%S')}] {len(results)}/{n_total} reps done")
                _save()
                unsaved = 0

    _save()
    logging.info(f"  Final checkpoint saved -> {checkpoint_path}")
    return results


# =============================================================================
# Aggregation
# =============================================================================

def _conv_vals_e28(recs: list, model: str, min_pl: int, fit_type: str) -> list:
    """Converged gamma_hat values for one condition (filter keys may be int or str after JSON)."""
    vals = []
    for r in recs:
        m_dict = r.get("results", {}).get(model, {})
        d = m_dict.get(min_pl, m_dict.get(str(min_pl), {}))
        if not d:
            continue
        fit = d.get(fit_type, {})
        if fit.get("converged", False):
            v = fit.get("gamma_hat", float("nan"))
            if math.isfinite(v):
                vals.append(v)
    return vals


def _summarise_e28(vals: list, real_gamma: float) -> dict:
    """Mean, SD, number converged, one-sided empirical p and the gate-G1 verdict."""
    if not vals:
        return {
            "mean": float("nan"), "sd": float("nan"),
            "n_conv": 0, "emp_p": float("nan"),
            "g1_verdict": "INSUFFICIENT_DATA",
        }
    arr = np.array(vals)
    mean_v = float(arr.mean())
    sd_v = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    emp_p = float((arr <= real_gamma).mean())
    # G1: the null mean departs from 0 if |mean| > 0.5 * SD
    if math.isfinite(sd_v) and sd_v > 0:
        g1_departs = abs(mean_v) > 0.5 * sd_v
    else:
        g1_departs = False
    return {
        "mean": mean_v,
        "sd": sd_v,
        "n_conv": len(vals),
        "emp_p": emp_p,
        "g1_verdict": "DEPARTS" if g1_departs else "DOES_NOT_DEPART",
    }


def _real_rh_threads(raw_dlgs: list, min_pl: int = 1) -> Tuple[List[dict], float]:
    evs_list = [
        [{"cat": e["cat"], "H": e["H"], "p_dist": e["p"]} for e in dlg if e["plurality"] >= min_pl]
        for dlg in raw_dlgs
    ]
    evs_list = [x for x in evs_list if len(x) >= 2]
    return _build_soft_rh_threads(evs_list)


def _real_wc_threads(raw_dlgs: list, min_pl: int = 1) -> Tuple[List[dict], float]:
    evs_list = []
    for dlg in raw_dlgs:
        filtered = [e for e in dlg if e["plurality"] >= min_pl]
        if len(filtered) >= 2:
            evs_list.append([{"cat": e["cat"], "H": e["H"], "p_dist": e["p"]} for e in filtered])
    return _build_soft_wc_threads(evs_list, K)


def _real_gamma(threads: List[dict], H_bar: float) -> float:
    r = SoftMarkEstimator(threads, K=K, H_bar=H_bar, l1_alpha=0.001).fit(
        n_restarts=N_RESTARTS_REAL, maxiter=MAXITER_REAL, seed=0)
    return float(r.gamma_hat)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    _setup_log()
    t0 = time.time()

    logging.info("[1] Loading EL-wide data ...")
    raw_dlgs = load_el_raw()
    q_pool = empirical_q_pool(raw_dlgs)
    confusion = compute_empirical_confusion(raw_dlgs)
    H_dist = empirical_H_pool(raw_dlgs)
    n_dlg = len(raw_dlgs)
    logging.info(f"    {n_dlg} dialogues, {sum(len(d) for d in raw_dlgs)} events")

    flat_cats = np.concatenate(
        [np.array([e["cat"] for e in d], dtype=np.int64) for d in raw_dlgs])
    flat_H = np.concatenate(
        [np.array([e["H"] for e in d], dtype=np.float64) for d in raw_dlgs])
    dlg_starts = np.zeros(n_dlg + 1, dtype=np.int64)
    for i, d in enumerate(raw_dlgs):
        dlg_starts[i + 1] = dlg_starts[i] + len(d)

    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]

    logging.info("[2] numba JIT warm-up ...")
    _warmup_numba_e28(K)

    logging.info("[3] Gradient check and real-data soft-mark fits ...")
    gc = gradient_check_softmark(K=4, n_threads=6, l1_alpha=0.001, seed=42, eps=1e-5)
    logging.info(f"    gradient check: max_rel_err = {gc['max_rel_err']:.2e}")

    real_gammas = {
        "rh_pl1": _real_gamma(*_real_rh_threads(raw_dlgs, 1)),
        "wc_pl1": _real_gamma(*_real_wc_threads(raw_dlgs, 1)),
        "wc_pl3": _real_gamma(*_real_wc_threads(raw_dlgs, 3)),
        "wc_pl4": _real_gamma(*_real_wc_threads(raw_dlgs, 4)),
        "rh_pl3": _real_gamma(*_real_rh_threads(raw_dlgs, 3)),
        "rh_pl4": _real_gamma(*_real_rh_threads(raw_dlgs, 4)),
    }
    for k, v in real_gammas.items():
        logging.info(f"    real gamma ({k}) = {v:+.4f}")

    val_result = {
        "gradient_check": {
            "max_rel_err": gc["max_rel_err"],
            "mean_rel_err": gc["mean_rel_err"],
            "passed_strict": gc["passed_strict"],
            "passed": gc["passed"],
            "threshold": 1e-5,
            "result": "PASS" if gc["passed_strict"] else "FAIL",
        },
        "real_data_gammas": real_gammas,
    }
    (OUT_DIR / "E28_validation.json").write_text(
        json.dumps({"experiment": "E28_validation", "validation": val_result},
                   ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    if not gc["passed_strict"]:
        logging.error("Gradient check failed (max_rel_err >= 1e-5); stopping.")
        return

    logging.info(f"[4] Null replications: {N_REPS} ...")
    worker_args = [
        (rep, q_pool, confusion, flat_cats, flat_H, dlg_starts,
         H_sorted, q_sorted, Q_NOISE_STD)
        for rep in range(N_REPS)
    ]
    null_recs = _run_with_progress(_e28_worker, worker_args, OUT_DIR / "E28_partial.json",
                                   key_name="scheme_a_prime", n_workers=N_WORKERS)

    logging.info("[5] Aggregating ...")
    real_gamma_map_wc = {1: real_gammas["wc_pl1"], 3: real_gammas["wc_pl3"], 4: real_gammas["wc_pl4"]}
    real_gamma_map_rh = {1: real_gammas["rh_pl1"], 3: real_gammas["rh_pl3"], 4: real_gammas["rh_pl4"]}

    agg = {}
    for model in ["indep", "linked"]:
        for min_pl in MIN_PL_LIST:
            for fit_type in ["wc", "rh"]:
                key = f"{fit_type}_{model}_pl{min_pl}"
                real = real_gamma_map_wc[min_pl] if fit_type == "wc" else real_gamma_map_rh[min_pl]
                agg[key] = _summarise_e28(_conv_vals_e28(null_recs, model, min_pl, fit_type), real)
                agg[key]["real_gamma"] = real

    out = {
        "experiment": "E28",
        "config": {
            "N_REPS": N_REPS,
            "N_WORKERS": N_WORKERS,
            "N_RESTARTS_SOFT": N_RESTARTS_SOFT,
            "N_RESTARTS_REAL": N_RESTARTS_REAL,
            "MIN_PL_LIST": MIN_PL_LIST,
            "Q_NOISE_STD": Q_NOISE_STD,
        },
        "validation": val_result,
        "real_gammas": real_gammas,
        "null_distribution": agg,
        "elapsed_sec": float(time.time() - t0),
    }
    (OUT_DIR / "E28_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    logging.info(f"Results -> {OUT_DIR / 'E28_results.json'}")


if __name__ == "__main__":
    main()
