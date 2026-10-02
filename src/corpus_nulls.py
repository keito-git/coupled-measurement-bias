"""
Per-corpus null distributions of the hard-mark within-cell gamma_hat under gamma = 0
(Friends, EmotionPush, M3ED), shared by experiments E33c and E40.

Real category sequences (majority labels of the real votes) are kept as truth and votes are re-simulated:
  S1  accuracy-linked pipeline: q = q_sorted[lower-bound rank of the real H] + N(0, 0.05), clipped to
      [0.2, 1]; for Friends and EmotionPush the rank mapping uses the EL-wide (Friends + EmotionPush)
      q and H pools, for M3ED its own pools. M3ED (3 annotators) is rejection-sampled until a majority.
  S2  fitted AR(1) annotator model (E30_fit.json; M3ED: E30b_m3ed.json "best"):
      u follows the discretised AR(1) chain, q = sigmoid(a - b*u), errors from the fitted confusion.
The estimator sees the observed majority label and the observed vote entropy.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.special import roots_hermite as _roots_hermite
from scipy.stats import pearsonr

import config
from dtsim_fits import fit_within_cell

log = logging.getLogger(__name__)

K = 7
CKPT_INTERVAL = 25
Q_NOISE_STD = 0.05


# ============================================================================
# Data loading
# ============================================================================

def _parse_el_file(path: Path, n_ann: int = 5) -> List[List[dict]]:
    raw = json.loads(path.read_text())
    dlgs = []
    for dialog in raw:
        evs = []
        for utt in dialog:
            ann = utt.get("annotation", "")
            if len(ann) != K or not ann.isdigit():
                continue
            votes = np.array([int(c) for c in ann], dtype=np.float64)
            if votes.sum() == 0:
                continue
            p = votes / votes.sum()
            H = float(-(p * np.log(p + 1e-12)).sum())
            evs.append({
                "votes":     votes,
                "H":         H,
                "cat":       int(p.argmax()),
                "plurality": int(votes.max()),
            })
        if len(evs) >= 2:
            dlgs.append(evs)
    return dlgs


def load_friends_raw() -> List[List[dict]]:
    return _parse_el_file(config.FRIENDS_JSON)


def load_emotionpush_raw() -> List[List[dict]]:
    return _parse_el_file(config.EMOTIONPUSH_JSON)


def load_m3ed_raw() -> List[List[dict]]:
    import pandas as pd
    df = pd.read_parquet(config.M3ED_PARQUET)
    df = df[df["dataset_source"] == "m3ed"]
    df = df[df["n_raters"] == 3].copy()
    df = df.sort_values(["dialog_id", "turn_id"])
    dlgs: List[List[dict]] = []
    for _did, grp in df.groupby("dialog_id"):
        evs = []
        for _, row in grp.iterrows():
            votes = np.round(row["p_dist"] * 3).astype(np.int64)
            if votes.sum() != 3:
                diff = 3 - int(votes.sum())
                fracs = row["p_dist"] * 3 - votes
                idx = np.argsort(fracs)[::-1]
                for i in range(abs(diff)):
                    votes[idx[i]] += (1 if diff > 0 else -1)
            votes = votes.astype(np.float64)
            if votes.sum() == 0:
                continue
            p = votes / votes.sum()
            H = float(-(p * np.log(p + 1e-12)).sum())
            evs.append({
                "votes":     votes,
                "H":         H,
                "cat":       int(p.argmax()),
                "plurality": int(votes.max()),
            })
        if len(evs) >= 2:
            dlgs.append(evs)
    return dlgs


def compute_empirical_q_pool(dlgs: List[List[dict]], n_ann: int) -> np.ndarray:
    return np.array([e["plurality"] / float(n_ann) for d in dlgs for e in d])


def compute_empirical_H_pool(dlgs: List[List[dict]]) -> np.ndarray:
    return np.array([e["H"] for d in dlgs for e in d])


def compute_confusion(dlgs: List[List[dict]]) -> np.ndarray:
    C = np.zeros((K, K))
    for d in dlgs:
        for e in d:
            i = e["cat"]
            for j in range(K):
                if j != i:
                    C[i, j] += e["votes"][j]
    for i in range(K):
        rs = C[i].sum()
        if rs > 0:
            C[i] /= rs
        else:
            C[i] = np.ones(K) / (K - 1)
            C[i, i] = 0.0
    return C


def compute_consec_H_corr(dlgs: List[List[dict]]) -> float:
    h1_list: List[float] = []
    h2_list: List[float] = []
    for dlg in dlgs:
        for i in range(len(dlg) - 1):
            h1_list.append(dlg[i]["H"])
            h2_list.append(dlg[i + 1]["H"])
    h1 = np.array(h1_list)
    h2 = np.array(h2_list)
    if len(h1) < 3 or h1.std() < 1e-9 or h2.std() < 1e-9:
        return float("nan")
    return float(pearsonr(h1, h2)[0])


# ============================================================================
# Real-data estimate
# ============================================================================

def compute_gamma_hat(dlgs: List[List[dict]], seed: int = 0) -> dict:
    dlgs_filt = [d for d in dlgs if len(d) >= 2]
    if len(dlgs_filt) < 5:
        return {"gamma_hat": float("nan"), "converged": False,
                "at_bound": False, "n_dlg": len(dlgs_filt)}
    wc = fit_within_cell(dlgs_filt, seed=seed)
    return {
        "gamma_hat":  float(wc["gamma_hat"]),
        "converged":  bool(wc["converged"]),
        "at_bound":   bool(wc["at_bound"]),
        "n_dlg":      len(dlgs_filt),
    }


# ============================================================================
# S1: accuracy-linked pipeline
# ============================================================================

def _make_linked_q_fn(H_dist: np.ndarray, q_pool: np.ndarray):
    """Map H to a base accuracy by rank matching (decreasing in H)."""
    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]

    def q_for_h(h: float) -> float:
        rank = int(np.searchsorted(H_sorted, h, side="left"))  # lower-bound rank
        rank = min(rank, len(H_sorted) - 1)
        q_idx = int(rank / len(H_sorted) * len(q_sorted))
        q_idx = min(q_idx, len(q_sorted) - 1)
        return float(q_sorted[q_idx])

    return q_for_h


def _simulate_votes_el(rng, true_cat: int, q: float, confusion: np.ndarray,
                        n_ann: int) -> Tuple[int, float, int]:
    votes = np.zeros(K, dtype=np.int64)
    conf_row = confusion[true_cat].copy()
    row_sum = conf_row.sum()
    if row_sum > 1e-12:
        conf_row /= row_sum
    else:
        conf_row = np.ones(K) / (K - 1)
        conf_row[true_cat] = 0.0
        conf_row /= conf_row.sum()

    for _ in range(n_ann):
        if rng.random() < q:
            votes[true_cat] += 1
        else:
            votes[int(rng.choice(K, p=conf_row))] += 1

    max_v = int(votes.max())
    tied = np.where(votes == max_v)[0]
    obs_cat = int(rng.choice(tied))
    total = float(votes.sum())
    p = votes / total
    H = float(-(p * np.log(p + 1e-12)).sum())
    return obs_cat, H, max_v


def _simulate_votes_m3ed(rng, true_cat: int, q: float, confusion: np.ndarray,
                          n_ann: int = 3, max_tries: int = 100) -> Tuple[int, float, int]:
    conf_row = confusion[true_cat].copy()
    row_sum = conf_row.sum()
    if row_sum > 1e-12:
        conf_row /= row_sum
    else:
        conf_row = np.ones(K) / (K - 1)
        conf_row[true_cat] = 0.0
        conf_row /= conf_row.sum()

    for _try in range(max_tries):
        votes = np.zeros(K, dtype=np.int64)
        for _ in range(n_ann):
            if rng.random() < q:
                votes[true_cat] += 1
            else:
                votes[int(rng.choice(K, p=conf_row))] += 1
        max_v = int(votes.max())
        if max_v >= 2:
            break
    else:
        votes = np.zeros(K, dtype=np.int64)
        votes[true_cat] = 2
        max_v = 2

    tied = np.where(votes == max_v)[0]
    obs_cat = int(rng.choice(tied))
    total = float(votes.sum())
    p = votes / total
    H = float(-(p * np.log(p + 1e-12)).sum())
    return obs_cat, H, max_v


def _s1_worker_el(args: tuple) -> dict:
    """S1 replication for Friends or EmotionPush (5 annotators; q and H pools from EL-wide)."""
    (rep, corpus_name,
     flat_cats, flat_H_real, dlg_starts,
     confusion_list, q_pool_list, H_dist_list,
     n_ann) = args

    confusion  = np.array(confusion_list)
    q_pool     = np.array(q_pool_list)
    H_dist     = np.array(H_dist_list)
    flat_cats  = np.array(flat_cats, dtype=np.int64)
    flat_H     = np.array(flat_H_real, dtype=np.float64)
    dlg_starts = np.array(dlg_starts, dtype=np.int64)
    n_dlg = len(dlg_starts) - 1

    q_for_H = _make_linked_q_fn(H_dist, q_pool)
    rng = np.random.default_rng(rep * 41117 + 7)

    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []
    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        evs = []
        for k in range(s, e):
            tc = int(flat_cats[k])
            h  = float(flat_H[k])
            q_base = q_for_H(h)
            q = float(np.clip(q_base + rng.normal(0.0, Q_NOISE_STD), 0.2, 1.0))
            obs, H_obs, plur = _simulate_votes_el(rng, tc, q, confusion, n_ann)
            evs.append({"cat": obs, "H": H_obs, "plurality": plur})
            H_obs_all.append(H_obs)
        sim_dlgs.append(evs)

    h1_list: List[float] = []
    h2_list: List[float] = []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1_list.append(dlg[i]["H"])
            h2_list.append(dlg[i + 1]["H"])
    h1 = np.array(h1_list)
    h2 = np.array(h2_list)
    consec_corr = (float(pearsonr(h1, h2)[0])
                   if len(h1) >= 3 and h1.std() > 1e-9 and h2.std() > 1e-9
                   else float("nan"))

    fit = {"gamma_hat": float("nan"), "converged": False, "at_bound": False, "n_dlg": 0}
    filt = [d for d in sim_dlgs if len(d) >= 2]
    if len(filt) >= 5:
        wc = fit_within_cell(filt, seed=rep)
        fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(filt),
        }

    return {
        "rep":         rep,
        "corpus":      corpus_name,
        "sim":         "S1",
        "gamma_hat":   fit["gamma_hat"],
        "converged":   fit["converged"],
        "at_bound":    fit["at_bound"],
        "n_dlg":       fit["n_dlg"],
        "H_obs_mean":  float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec_corr,
    }


def _s1_worker_m3ed(args: tuple) -> dict:
    """S1 replication for M3ED (3 annotators, rejection until a majority; M3ED q and H pools)."""
    (rep,
     flat_cats, flat_H_real, dlg_starts,
     confusion_list, q_pool_list, H_dist_list,
     n_ann) = args

    confusion  = np.array(confusion_list)
    q_pool     = np.array(q_pool_list)
    H_dist     = np.array(H_dist_list)
    flat_cats  = np.array(flat_cats, dtype=np.int64)
    flat_H     = np.array(flat_H_real, dtype=np.float64)
    dlg_starts = np.array(dlg_starts, dtype=np.int64)
    n_dlg = len(dlg_starts) - 1

    q_for_H = _make_linked_q_fn(H_dist, q_pool)
    rng = np.random.default_rng(rep * 31337 + 13)

    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []
    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        evs = []
        for k in range(s, e):
            tc = int(flat_cats[k])
            h  = float(flat_H[k])
            q_base = q_for_H(h)
            q = float(np.clip(q_base + rng.normal(0.0, Q_NOISE_STD), 0.2, 1.0))
            obs, H_obs, plur = _simulate_votes_m3ed(rng, tc, q, confusion, n_ann)
            evs.append({"cat": obs, "H": H_obs, "plurality": plur})
            H_obs_all.append(H_obs)
        sim_dlgs.append(evs)

    h1_list: List[float] = []
    h2_list: List[float] = []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1_list.append(dlg[i]["H"])
            h2_list.append(dlg[i + 1]["H"])
    h1 = np.array(h1_list)
    h2 = np.array(h2_list)
    consec_corr = (float(pearsonr(h1, h2)[0])
                   if len(h1) >= 3 and h1.std() > 1e-9 and h2.std() > 1e-9
                   else float("nan"))

    fit = {"gamma_hat": float("nan"), "converged": False, "at_bound": False, "n_dlg": 0}
    filt = [d for d in sim_dlgs if len(d) >= 2]
    if len(filt) >= 5:
        wc = fit_within_cell(filt, seed=rep)
        fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(filt),
        }

    return {
        "rep":          rep,
        "corpus":       "m3ed",
        "sim":          "S1",
        "gamma_hat":    fit["gamma_hat"],
        "converged":    fit["converged"],
        "at_bound":     fit["at_bound"],
        "n_dlg":        fit["n_dlg"],
        "H_obs_mean":   float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec_corr,
    }


# ============================================================================
# S2: fitted AR(1) annotator model
# ============================================================================

_N_U = 19
_x_gh, _w_gh = _roots_hermite(_N_U)
U_NODES = np.sqrt(2) * _x_gh
W_NODES = _w_gh / np.sqrt(np.pi)
PI0 = W_NODES / W_NODES.sum()


def _make_ar1_transition(rho: float) -> np.ndarray:
    """Row-stochastic AR(1) transition matrix on the quadrature nodes (as in E30)."""
    from scipy.special import logsumexp
    if abs(rho) < 1e-8:
        return np.tile(PI0, (_N_U, 1))
    std = max(math.sqrt(1.0 - rho * rho), 1e-8)
    diff = U_NODES[np.newaxis, :] - rho * U_NODES[:, np.newaxis]
    log_phi = -0.5 * (diff / std) ** 2
    log_T = log_phi + np.log(W_NODES)[np.newaxis, :]
    from scipy.special import logsumexp as _lse
    log_T -= _lse(log_T, axis=1, keepdims=True)
    return np.exp(log_T)


def _s2_worker_el(args: tuple) -> dict:
    """S2 replication for Friends or EmotionPush (5 annotators)."""
    (rep, corpus_name,
     flat_cats, dlg_starts,
     confusion_list,
     ar1_a, ar1_b, ar1_rho,
     n_ann) = args

    confusion  = np.array(confusion_list)
    flat_cats  = np.array(flat_cats, dtype=np.int64)
    dlg_starts = np.array(dlg_starts, dtype=np.int64)
    n_dlg = len(dlg_starts) - 1

    T = _make_ar1_transition(ar1_rho)
    rng = np.random.default_rng(rep * 53117 + 21)

    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []

    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        dlg_len = e - s

        # latent difficulty path (indices of the quadrature nodes)
        u_idx = np.empty(dlg_len, dtype=np.int64)
        u_idx[0] = int(rng.choice(_N_U, p=PI0))
        for m in range(1, dlg_len):
            u_idx[m] = int(rng.choice(_N_U, p=T[u_idx[m - 1]]))

        evs = []
        for m in range(dlg_len):
            tc = int(flat_cats[s + m])
            u_val = float(U_NODES[u_idx[m]])
            q = min(max(1.0 / (1.0 + math.exp(-(ar1_a - ar1_b * u_val))), 1e-4), 1.0 - 1e-4)

            obs, H_obs, plur = _simulate_votes_el(rng, tc, q, confusion, n_ann)
            evs.append({"cat": obs, "H": H_obs, "plurality": plur})
            H_obs_all.append(H_obs)
        sim_dlgs.append(evs)

    h1_list: List[float] = []
    h2_list: List[float] = []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1_list.append(dlg[i]["H"])
            h2_list.append(dlg[i + 1]["H"])
    h1 = np.array(h1_list)
    h2 = np.array(h2_list)
    consec_corr = (float(pearsonr(h1, h2)[0])
                   if len(h1) >= 3 and h1.std() > 1e-9 and h2.std() > 1e-9
                   else float("nan"))

    fit = {"gamma_hat": float("nan"), "converged": False, "at_bound": False, "n_dlg": 0}
    filt = [d for d in sim_dlgs if len(d) >= 2]
    if len(filt) >= 5:
        wc = fit_within_cell(filt, seed=rep)
        fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(filt),
        }

    return {
        "rep":          rep,
        "corpus":       corpus_name,
        "sim":          "S2",
        "gamma_hat":    fit["gamma_hat"],
        "converged":    fit["converged"],
        "at_bound":     fit["at_bound"],
        "n_dlg":        fit["n_dlg"],
        "H_obs_mean":   float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec_corr,
    }


def _s2_worker_m3ed(args: tuple) -> dict:
    """S2 replication for M3ED (3 annotators, rejection until a majority)."""
    (rep,
     flat_cats, dlg_starts,
     confusion_list,
     ar1_a, ar1_b, ar1_rho,
     n_ann) = args

    confusion  = np.array(confusion_list)
    flat_cats  = np.array(flat_cats, dtype=np.int64)
    dlg_starts = np.array(dlg_starts, dtype=np.int64)
    n_dlg = len(dlg_starts) - 1

    T = _make_ar1_transition(ar1_rho)
    rng = np.random.default_rng(rep * 57239 + 99)

    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []
    rejection_total = 0
    attempt_total = 0

    for d in range(n_dlg):
        s = int(dlg_starts[d])
        e = int(dlg_starts[d + 1])
        dlg_len = e - s

        u_idx = np.empty(dlg_len, dtype=np.int64)
        u_idx[0] = int(rng.choice(_N_U, p=PI0))
        for m in range(1, dlg_len):
            u_idx[m] = int(rng.choice(_N_U, p=T[u_idx[m - 1]]))

        evs = []
        for m in range(dlg_len):
            tc = int(flat_cats[s + m])
            u_val = float(U_NODES[u_idx[m]])
            q = min(max(1.0 / (1.0 + math.exp(-(ar1_a - ar1_b * u_val))), 1e-4), 1.0 - 1e-4)

            conf_row = confusion[tc].copy()
            row_sum = conf_row.sum()
            if row_sum > 1e-12:
                conf_row /= row_sum
            else:
                conf_row = np.ones(K) / (K - 1)
                conf_row[tc] = 0.0
                conf_row /= conf_row.sum()

            for _try in range(100):
                attempt_total += 1
                votes = np.zeros(K, dtype=np.int64)
                for _ in range(n_ann):
                    if rng.random() < q:
                        votes[tc] += 1
                    else:
                        votes[int(rng.choice(K, p=conf_row))] += 1
                if int(votes.max()) >= 2:
                    break
                rejection_total += 1
            else:
                # Fallback
                votes = np.zeros(K, dtype=np.int64)
                votes[tc] = 2
                attempt_total += 1

            max_v = int(votes.max())
            tied = np.where(votes == max_v)[0]
            obs_cat = int(rng.choice(tied))
            total = float(votes.sum())
            p = votes / total
            H_obs = float(-(p * np.log(p + 1e-12)).sum())

            evs.append({"cat": obs_cat, "H": H_obs, "plurality": max_v})
            H_obs_all.append(H_obs)
        sim_dlgs.append(evs)

    h1_list: List[float] = []
    h2_list: List[float] = []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1_list.append(dlg[i]["H"])
            h2_list.append(dlg[i + 1]["H"])
    h1 = np.array(h1_list)
    h2 = np.array(h2_list)
    consec_corr = (float(pearsonr(h1, h2)[0])
                   if len(h1) >= 3 and h1.std() > 1e-9 and h2.std() > 1e-9
                   else float("nan"))

    fit = {"gamma_hat": float("nan"), "converged": False, "at_bound": False, "n_dlg": 0}
    filt = [d for d in sim_dlgs if len(d) >= 2]
    if len(filt) >= 5:
        wc = fit_within_cell(filt, seed=rep)
        fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(filt),
        }

    return {
        "rep":           rep,
        "corpus":        "m3ed",
        "sim":           "S2",
        "gamma_hat":     fit["gamma_hat"],
        "converged":     fit["converged"],
        "at_bound":      fit["at_bound"],
        "n_dlg":         fit["n_dlg"],
        "H_obs_mean":    float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec_corr,
        "rejection_frac": (float(rejection_total) / attempt_total
                           if attempt_total > 0 else float("nan")),
    }


# ============================================================================
# Pool runner and analysis
# ============================================================================

def _run_with_progress(
    worker_fn,
    args_list: list,
    ckpt_path: Path,
    key_name: str,
    n_workers: int = 4,
) -> list:
    done_reps: set = set()
    results: list = []
    if ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_reps = {r["rep"] for r in results}
            log.info(f"  Checkpoint {ckpt_path.name}: {len(done_reps)} reps done")
        except Exception as ex:
            log.warning(f"  Checkpoint load failed ({ex}); starting fresh")

    pending = [a for a in args_list if a[0] not in done_reps]
    if not pending:
        log.info(f"  All {len(args_list)} reps found in checkpoint.")
        return results

    log.info(f"  {len(pending)} reps pending on {n_workers} workers ...")

    ctx = mp.get_context("spawn")
    unsaved = 0

    def _save():
        ckpt_path.write_text(
            json.dumps({key_name: results}, ensure_ascii=False,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8",
        )

    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pending):
            results.append(result)
            unsaved += 1
            n_done = len(results)
            if unsaved >= CKPT_INTERVAL:
                ts = time.strftime("%H:%M:%S")
                log.info(f"  [{ts}] {n_done}/{len(args_list)} reps done")
                _save()
                unsaved = 0

    _save()
    log.info(f"  Final checkpoint saved -> {ckpt_path}")
    return results


def analyse_null(results: list, real_gamma: float, corpus: str, sim: str) -> dict:
    gamma_hats = [
        r["gamma_hat"] for r in results
        if r.get("corpus") == corpus and r.get("sim") == sim
           and not math.isnan(r["gamma_hat"])
    ]
    converged_n = sum(
        1 for r in results
        if r.get("corpus") == corpus and r.get("sim") == sim
           and r.get("converged", False)
    )
    total_n = sum(
        1 for r in results
        if r.get("corpus") == corpus and r.get("sim") == sim
    )

    if len(gamma_hats) < 2:
        return {
            "corpus": corpus, "sim": sim, "n_reps": total_n, "n_conv": converged_n,
            "frac_conv": (converged_n / total_n if total_n > 0 else float("nan")),
            "real_gamma": real_gamma, "null_mean": float("nan"),
            "null_std": float("nan"), "null_q2_5": float("nan"),
            "null_q97_5": float("nan"), "emp_p": float("nan"),
            "reject_gamma0": False, "note": "insufficient converged reps",
        }

    arr = np.array(gamma_hats)
    q2_5  = float(np.percentile(arr, 2.5))
    q97_5 = float(np.percentile(arr, 97.5))
    null_mean = float(np.mean(arr))
    null_std  = float(np.std(arr, ddof=1))
    # P(|null| >= |real|); the paper reports the median-centred two-sided p (analysis/two_sided_pvalues.py)
    emp_p = float(np.mean(np.abs(arr) >= abs(real_gamma)))
    reject = not (q2_5 <= real_gamma <= q97_5)

    return {
        "corpus":       corpus,
        "sim":          sim,
        "n_reps":       total_n,
        "n_conv":       converged_n,
        "frac_conv":    float(converged_n) / total_n if total_n > 0 else float("nan"),
        "real_gamma":   float(real_gamma),
        "null_mean":    null_mean,
        "null_std":     null_std,
        "null_q2_5":    q2_5,
        "null_q97_5":   q97_5,
        "emp_p":        emp_p,
        "reject_gamma0": reject,
    }


def sanity_check_s2(results: list, real_consec_H: float, real_H_mean: float,
                    corpus: str) -> dict:
    """Simulated lag-1 entropy correlation and mean entropy (S1 and S2) against the real values."""
    corr_vals = [
        r["consec_H_corr"] for r in results
        if r.get("corpus") == corpus and r.get("sim") == "S2"
           and not math.isnan(r.get("consec_H_corr", float("nan")))
    ]
    h_mean_vals = [
        r["H_obs_mean"] for r in results
        if r.get("corpus") == corpus and r.get("sim") == "S2"
           and not math.isnan(r.get("H_obs_mean", float("nan")))
    ]
    s1_corr_vals = [
        r["consec_H_corr"] for r in results
        if r.get("corpus") == corpus and r.get("sim") == "S1"
           and not math.isnan(r.get("consec_H_corr", float("nan")))
    ]
    s1_h_mean_vals = [
        r["H_obs_mean"] for r in results
        if r.get("corpus") == corpus and r.get("sim") == "S1"
           and not math.isnan(r.get("H_obs_mean", float("nan")))
    ]

    def _stats(lst):
        if not lst:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.array(lst)
        return {"mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr)}

    return {
        "real_consec_H_corr":  float(real_consec_H),
        "real_H_mean":         float(real_H_mean),
        "S2_consec_H_corr":    _stats(corr_vals),
        "S2_H_mean":           _stats(h_mean_vals),
        "S1_consec_H_corr":    _stats(s1_corr_vals),
        "S1_H_mean":           _stats(s1_h_mean_vals),
    }

