"""
EL-wide null simulators and the full decision table (experiments E35, E38, E39).

Data: EL-wide = EmotionLines Friends + EmotionPush. Four estimators of gamma:
    hard_wc  hard-mark, within-cell modifier      hard_rh  hard-mark, raw (globally centred) modifier
    soft_wc  soft-mark, within-cell modifier      soft_rh  soft-mark, raw modifier
Null simulators (gamma = 0):
    S1  accuracy-linked pipeline on the real category sequences (lower-bound rank of the real H)
    S2  fitted AR(1) annotator model (E30) on the real category sequences
    C0  Scheme C' generator (DT-AMHP fitted to EL-wide) with linked votes at gamma = 0
Mode "table" (E35): X3 decision table (4 estimators x 3 simulators, N_NULL_REPS each), X2 power
    (Scheme C' at true gamma in POWER_GRID, hard_wc) and cross-simulator type-I error, X6 sensitivity
    (S1' = S1 with an accuracy offset delta calibrated to the real mean entropy).
Mode "bigB" (E38): hard_wc nulls under S2 and C0 only, with a larger N_NULL_REPS.
Mode "ppc" (E39): vote-level summary statistics of 20 simulated data sets per simulator.
p-values: median-centred two-sided, p = (1 + #{|g_b - med| >= |g_real - med|}) / (B + 1).
Requires E30_fit.json (RESULTS_ROOT/E30).
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.special import roots_hermite
from scipy.stats import pearsonr

import config
from dtsim_core import (
    _generate_cats_from_H, _compute_within_cell_H,
    residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded,
)
from dtsim_fits import fit_within_cell, fit_raw_H
from dtsim_kernels import _seed_nb, _process_one_dlg_scheme_c_nb, _warmup_numba
from estimator_dt import unpack_v_dt
from estimator_softmark import SoftMarkEstimator

log = logging.getLogger(__name__)

# The mode is read by the worker processes through this environment variable.
MODE_ENV = "EL_NULLS_MODE"

K = 7
N_ANN_EL = 5
N_REFINE = 2                # Scheme C' refinement passes
Q_NOISE_STD = 0.05
N_POWER_REPS = 100          # X2 power reps per true gamma
N_CAL_PILOT = 3             # pilot reps for the runtime estimate
N_X6_CAL_REPS = 20          # X6 delta-calibration reps (rep ids 200..219)
N_X6_NULL_REPS = 200        # X6 S1' null reps
CKPT_INTERVAL = 20
REAL_GAMMA_EL_WIDE = -0.3915
POWER_GAMMA_GRID = [-1.2, -0.8, -0.4, -0.2, 0.4]

# AR(1) difficulty on 19 Gauss-Hermite nodes (as in E30)
_N_U         = 19
_x_gh, _w_gh = roots_hermite(_N_U)
U_NODES      = np.sqrt(2) * _x_gh
W_NODES      = _w_gh / np.sqrt(np.pi)
PI0          = W_NODES / W_NODES.sum()
LOG_PI0      = np.log(PI0 + 1e-300)


# ===============================================================================
# 1. Data loading
# ===============================================================================


def _parse_el_file(path: Path) -> List[List[dict]]:
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
            p     = votes / votes.sum()
            H     = float(-(p * np.log(p + 1e-12)).sum())
            evs.append({
                "votes":     votes,
                "H":         H,
                "cat":       int(p.argmax()),
                "plurality": int(votes.max()),
                "p_dist":    p.copy(),
            })
        if len(evs) >= 2:
            dlgs.append(evs)
    return dlgs


def load_el_data() -> Tuple[List[List[dict]], List[List[dict]]]:
    fr = _parse_el_file(config.FRIENDS_JSON)
    ep = _parse_el_file(config.EMOTIONPUSH_JSON)
    return fr, ep


def compute_H_mean(dlgs: List[List[dict]]) -> float:
    h = [e["H"] for d in dlgs for e in d]
    return float(np.mean(h))


def compute_consec_H_corr(dlgs: List[List[dict]]) -> float:
    h1, h2 = [], []
    for d in dlgs:
        for i in range(len(d) - 1):
            h1.append(d[i]["H"]); h2.append(d[i + 1]["H"])
    a, b = np.array(h1), np.array(h2)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(pearsonr(a, b)[0])


def make_flat_arrays(dlgs: List[List[dict]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (flat_cats, flat_H, dlg_starts)."""
    flat_c = np.concatenate([np.array([e["cat"] for e in d], dtype=np.int64) for d in dlgs])
    flat_H = np.concatenate([np.array([e["H"]   for e in d], dtype=np.float64) for d in dlgs])
    starts = np.zeros(len(dlgs) + 1, dtype=np.int64)
    for i, d in enumerate(dlgs):
        starts[i + 1] = starts[i] + len(d)
    return flat_c, flat_H, starts


def compute_empirical_q_pool(dlgs: List[List[dict]], n_ann: int = 5) -> np.ndarray:
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


def make_q_link_fn(H_dist: np.ndarray, q_pool: np.ndarray):
    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]
    def q_for_h(h: float) -> float:
        rank = int(np.searchsorted(H_sorted, h, side="left"))  # lower-bound rank
        rank = min(rank, len(H_sorted) - 1)
        q_idx = int(rank / len(H_sorted) * len(q_sorted))
        q_idx = min(q_idx, len(q_sorted) - 1)
        return float(q_sorted[q_idx])
    return q_for_h


# ===============================================================================
# 2. Vote simulation helpers (with p_dist)
# ===============================================================================

def _sim_vote_pdist(
    rng: np.random.Generator,
    true_cat: int,
    q: float,
    confusion: np.ndarray,
    n_ann: int,
    max_tries_reject: int = 0,   # 0 = no rejection (EL); set >0 for M3ED
) -> Tuple[int, float, int, np.ndarray]:
    """Simulate votes; return (obs_cat, H_obs, plurality, p_dist)."""
    conf_row = confusion[true_cat].copy()
    rs = conf_row.sum()
    if rs > 1e-12:
        conf_row /= rs
    else:
        conf_row = np.ones(K) / (K - 1)
        conf_row[true_cat] = 0.0
        conf_row /= conf_row.sum()

    for _try in range(max(1, max_tries_reject)):
        votes = np.zeros(K, dtype=np.float64)
        for _ in range(n_ann):
            if rng.random() < q:
                votes[true_cat] += 1.0
            else:
                votes[int(rng.choice(K, p=conf_row))] += 1.0
        if max_tries_reject <= 0 or votes.max() >= 2:
            break
    else:
        votes = np.zeros(K, dtype=np.float64)
        votes[true_cat] = n_ann / 2.0 + 1.0
    total = votes.sum()
    if total == 0:
        votes[true_cat] = float(n_ann)
        total = float(n_ann)

    p = votes / total
    H_obs = float(-(p * np.log(p + 1e-12)).sum())
    max_v = votes.max()
    tied = np.where(votes == max_v)[0]
    obs_cat = int(rng.choice(tied))
    return obs_cat, H_obs, int(max_v), p.copy()


# ===============================================================================
# 3. AR(1) transition matrix helper
# ===============================================================================

def _make_ar1_T(rho: float) -> np.ndarray:
    from scipy.special import logsumexp
    if abs(rho) < 1e-8:
        return np.tile(PI0, (_N_U, 1))
    std = max(math.sqrt(1.0 - rho * rho), 1e-8)
    diff = U_NODES[np.newaxis, :] - rho * U_NODES[:, np.newaxis]
    log_phi = -0.5 * (diff / std) ** 2
    log_T = log_phi + np.log(W_NODES)[np.newaxis, :]
    log_T -= logsumexp(log_T, axis=1, keepdims=True)
    return np.exp(log_T)


# ===============================================================================
# 4. Fit wrappers (all 4 estimators)
# ===============================================================================

def ppc_stats(dlgs, n_ann=5):
    """Vote-level summary statistics for posterior predictive checks."""
    H = np.array([e["H"] for d in dlgs for e in d])
    V = np.array([np.asarray(e["p_dist"]) * n_ann for d in dlgs for e in d])
    V = np.round(V)
    srt = -np.sort(-V, axis=1)
    cats = np.array([e["cat"] for d in dlgs for e in d])
    h1 = np.concatenate([[e["H"] for e in d[:-1]] for d in dlgs if len(d) > 1])
    h2 = np.concatenate([[e["H"] for e in d[1:]] for d in dlgs if len(d) > 1])
    pair = (V * (V - 1)).sum(1) / (n_ann * (n_ann - 1))
    out = {"H_mean": float(H.mean()), "H_sd": float(H.std()), "H_q25": float(np.quantile(H, .25)),
           "H_q50": float(np.quantile(H, .5)), "H_q75": float(np.quantile(H, .75)),
           "lag1_corr_H": float(np.corrcoef(h1, h2)[0, 1]),
           "unanimous_rate": float((srt[:, 0] == n_ann).mean()),
           "margin_mean": float((srt[:, 0] - srt[:, 1]).mean()),
           "pairwise_agreement": float(pair.mean())}
    for k in range(7):
        out[f"label_freq_{k}"] = float((cats == k).mean())
    return out



def _fit_all_four(
    dlgs: List[List[dict]],
    rep_seed: int,
    n_restarts_hard: int = 1,
    n_restarts_soft: int = 1,
) -> dict:
    """Fit all 4 estimator variants on dlgs (each event has cat, H, plurality, p_dist)."""
    mode = os.environ.get(MODE_ENV, "table")
    if mode == "ppc" and rep_seed != 0:   # rep_seed 0 is the real-data fit
        return {"ppc": ppc_stats(dlgs)}

    # Filter: at least 2 events per dialogue
    dlgs_f = [d for d in dlgs if len(d) >= 2]

    def _empty() -> dict:
        return {"gamma_hat": float("nan"), "converged": False, "at_bound": False}

    if len(dlgs_f) < 5:
        return {
            "hard_wc": _empty(), "hard_rh": _empty(),
            "soft_wc": _empty(), "soft_rh": _empty(),
            "n_dlg": len(dlgs_f),
        }

    hard_wc = fit_within_cell(dlgs_f, seed=rep_seed)
    if mode == "bigB":
        def _e():
            return {"gamma_hat": float("nan"), "converged": False, "at_bound": False}
        return {"hard_wc": {"gamma_hat": hard_wc["gamma_hat"], "converged": hard_wc["converged"],
                            "at_bound": hard_wc.get("at_bound", False)},
                "hard_rh": _e(), "soft_wc": _e(), "soft_rh": _e()}
    hard_rh = fit_raw_H(dlgs_f, seed=rep_seed + 100000)

    # Soft mark
    has_pdist = all(
        "p_dist" in e
        for d in dlgs_f
        for e in d
    )
    if has_pdist:
        soft_wc = _fit_soft_wc(dlgs_f, rep_seed + 200000, n_restarts_soft)
        soft_rh = _fit_soft_rh(dlgs_f, rep_seed + 300000, n_restarts_soft)
    else:
        soft_wc = _empty()
        soft_rh = _empty()

    return {
        "hard_wc":  {"gamma_hat": float(hard_wc["gamma_hat"]),
                     "converged": bool(hard_wc["converged"]),
                     "at_bound": bool(hard_wc["at_bound"])},
        "hard_rh":  {"gamma_hat": float(hard_rh["gamma_hat"]),
                     "converged": bool(hard_rh["converged"]),
                     "at_bound": bool(hard_rh["at_bound"])},
        "soft_wc":  soft_wc,
        "soft_rh":  soft_rh,
        "n_dlg":    len(dlgs_f),
    }


def _fit_soft_wc(dlgs: List[List[dict]], seed: int, n_restarts: int = 1) -> dict:
    # Within-cell residualisation
    intermediates = []
    for evs in dlgs:
        cats  = np.array([e["cat"]    for e in evs], dtype=np.int64)
        H_raw = np.array([e["H"]      for e in evs], dtype=np.float64)
        p_d   = np.array([e["p_dist"] for e in evs], dtype=np.float64)
        H_wc  = H_raw.copy()
        for c in range(K):
            idx = np.where(cats == c)[0]
            if len(idx) > 1:
                H_wc[idx] -= H_raw[idx].mean()
            elif len(idx) == 1:
                H_wc[idx[0]] = 0.0
        intermediates.append((cats, H_wc, p_d))
    all_H_wc = np.concatenate([x[1] for x in intermediates])
    H_bar    = float(all_H_wc.mean())
    threads  = []
    for (cats, H_wc, p_d) in intermediates:
        n = len(cats)
        threads.append({
            "times_h": np.arange(n, dtype=np.float64),
            "cats":    cats,
            "Hs_c":    H_wc - H_bar,
            "p_dist":  p_d,
            "s_vals":  np.zeros(n),
            "T":       float(n),
            "s_int":   0.0,
        })
    return _call_softmark(threads, H_bar, seed, n_restarts)


def _fit_soft_rh(dlgs: List[List[dict]], seed: int, n_restarts: int = 1) -> dict:
    all_H = [e["H"] for d in dlgs for e in d]
    H_bar = float(np.mean(all_H)) if all_H else 0.0
    threads = []
    for evs in dlgs:
        n      = len(evs)
        cats   = np.array([e["cat"]    for e in evs], dtype=np.int64)
        H      = np.array([e["H"]      for e in evs], dtype=np.float64)
        p_d    = np.array([e["p_dist"] for e in evs], dtype=np.float64)
        threads.append({
            "times_h": np.arange(n, dtype=np.float64),
            "cats":    cats,
            "Hs_c":    H - H_bar,
            "p_dist":  p_d,
            "s_vals":  np.zeros(n),
            "T":       float(n),
            "s_int":   0.0,
        })
    return _call_softmark(threads, H_bar, seed, n_restarts)


def _call_softmark(threads, H_bar, seed, n_restarts) -> dict:
    K_val = max((int(threads[i]["cats"].max()) + 1 for i in range(len(threads))),
                default=K)
    K_val = max(K_val, K)
    est = SoftMarkEstimator(threads, K=K_val, H_bar=H_bar, l1_alpha=0.001)
    r   = est.fit(n_restarts=n_restarts, maxiter=3000, seed=seed)
    gamma = float(r.gamma_hat)
    conv  = bool(r.success) and math.isfinite(gamma)
    at_b  = abs(gamma) >= 4.5
    return {"gamma_hat": gamma, "converged": conv, "at_bound": at_b}


# ===============================================================================
# 5. S1 null worker
# ===============================================================================

def _s1_null_worker(args: tuple) -> dict:
    """
    S1 null replication (EL-wide, gamma = 0, real categories, linked accuracy).
    Returns all 4 estimator fits + diagnostics.
    """
    (rep,
     fr_cats, fr_H, fr_starts,
     ep_cats, ep_H, ep_starts,
     confusion_list, q_pool_list, H_dist_list,
     delta) = args

    confusion  = np.array(confusion_list)
    q_pool     = np.array(q_pool_list)
    H_dist     = np.array(H_dist_list)
    fr_cats    = np.array(fr_cats, dtype=np.int64)
    fr_H       = np.array(fr_H,    dtype=np.float64)
    fr_starts  = np.array(fr_starts, dtype=np.int64)
    ep_cats    = np.array(ep_cats, dtype=np.int64)
    ep_H       = np.array(ep_H,    dtype=np.float64)
    ep_starts  = np.array(ep_starts, dtype=np.int64)

    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]

    def q_for_h(h: float) -> float:
        rank = int(np.searchsorted(H_sorted, h, side="left"))  # lower-bound rank
        rank = min(rank, len(H_sorted) - 1)
        q_idx = int(rank / len(H_sorted) * len(q_sorted))
        q_idx = min(q_idx, len(q_sorted) - 1)
        return float(q_sorted[q_idx]) + delta

    rng = np.random.default_rng(rep * 41117 + 3)
    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []

    def _sim_corpus(cats_flat, H_flat, starts_arr):
        n_dlg_c = len(starts_arr) - 1
        dlgs_c  = []
        for d in range(n_dlg_c):
            s, e = int(starts_arr[d]), int(starts_arr[d + 1])
            evs = []
            for k in range(s, e):
                q_base = q_for_h(float(H_flat[k]))
                q = float(np.clip(q_base + rng.normal(0.0, Q_NOISE_STD), 0.2, 1.0))
                obs, H_obs, plur, p_d = _sim_vote_pdist(
                    rng, int(cats_flat[k]), q, confusion, N_ANN_EL)
                evs.append({
                    "cat": obs, "H": H_obs, "plurality": plur, "p_dist": p_d})
                H_obs_all.append(H_obs)
            dlgs_c.append(evs)
        return dlgs_c

    sim_dlgs  = _sim_corpus(fr_cats, fr_H, fr_starts)
    sim_dlgs += _sim_corpus(ep_cats, ep_H, ep_starts)

    h1, h2 = [], []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1.append(dlg[i]["H"]); h2.append(dlg[i + 1]["H"])
    ha, hb = np.array(h1), np.array(h2)
    consec = (float(pearsonr(ha, hb)[0])
              if len(ha) >= 3 and ha.std() > 1e-9 and hb.std() > 1e-9
              else float("nan"))

    fits = _fit_all_four(sim_dlgs, rep_seed=rep)
    return {
        "rep":         rep,
        "sim":         "S1",
        "H_obs_mean":  float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec,
        "fits":        fits,
    }


# ===============================================================================
# 6. S2 EL-wide null worker
# ===============================================================================

def _s2_el_wide_null_worker(args: tuple) -> dict:
    """S2 null simulation (EL-wide = Friends+EmotionPush, AR(1) models)."""
    (rep,
     fr_cats, fr_starts, fr_confusion,
     fr_a, fr_b, fr_rho,
     ep_cats, ep_starts, ep_confusion,
     ep_a, ep_b, ep_rho, n_ann) = args

    fr_cats    = np.array(fr_cats, dtype=np.int64)
    fr_starts  = np.array(fr_starts, dtype=np.int64)
    fr_conf    = np.array(fr_confusion)
    ep_cats    = np.array(ep_cats, dtype=np.int64)
    ep_starts  = np.array(ep_starts, dtype=np.int64)
    ep_conf    = np.array(ep_confusion)

    from scipy.special import logsumexp as _lse
    def _T(rho):
        if abs(rho) < 1e-8:
            return np.tile(PI0, (_N_U, 1))
        std = max(math.sqrt(1.0 - rho * rho), 1e-8)
        diff = U_NODES[np.newaxis, :] - rho * U_NODES[:, np.newaxis]
        log_phi = -0.5 * (diff / std) ** 2
        log_T = log_phi + np.log(W_NODES)[np.newaxis, :]
        log_T -= _lse(log_T, axis=1, keepdims=True)
        return np.exp(log_T)

    T_fr = _T(fr_rho)
    T_ep = _T(ep_rho)

    def _sim_ar1(cats_flat, starts_arr, confusion_c, a_c, b_c, T_c, rng_seed_base):
        n_dlg_c = len(starts_arr) - 1
        rng = np.random.default_rng(rng_seed_base)
        dlgs_c = []
        H_obs_all = []
        for d in range(n_dlg_c):
            s, e = int(starts_arr[d]), int(starts_arr[d + 1])
            dlg_len = e - s
            u_idx = np.empty(dlg_len, dtype=np.int64)
            u_idx[0] = int(rng.choice(_N_U, p=PI0))
            for m in range(1, dlg_len):
                u_idx[m] = int(rng.choice(_N_U, p=T_c[u_idx[m - 1]]))
            evs = []
            for m in range(dlg_len):
                tc    = int(cats_flat[s + m])
                u_val = float(U_NODES[u_idx[m]])
                q     = min(max(1.0 / (1.0 + math.exp(-(a_c - b_c * u_val))), 1e-4), 1.0 - 1e-4)
                obs, H_obs, plur, p_d = _sim_vote_pdist(rng, tc, q, confusion_c, n_ann)
                evs.append({"cat": obs, "H": H_obs, "plurality": plur, "p_dist": p_d})
                H_obs_all.append(H_obs)
            dlgs_c.append(evs)
        return dlgs_c, H_obs_all

    fr_dlgs, fr_H_all = _sim_ar1(fr_cats, fr_starts, fr_conf, fr_a, fr_b, T_fr,
                                   rep * 53117 + 22)
    ep_dlgs, ep_H_all = _sim_ar1(ep_cats, ep_starts, ep_conf, ep_a, ep_b, T_ep,
                                   rep * 57239 + 100)

    sim_dlgs  = fr_dlgs + ep_dlgs
    H_obs_all = fr_H_all + ep_H_all

    h1, h2 = [], []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1.append(dlg[i]["H"]); h2.append(dlg[i + 1]["H"])
    ha, hb = np.array(h1), np.array(h2)
    consec = (float(pearsonr(ha, hb)[0])
              if len(ha) >= 3 and ha.std() > 1e-9 and hb.std() > 1e-9
              else float("nan"))

    fits = _fit_all_four(sim_dlgs, rep_seed=rep)
    return {
        "rep":          rep,
        "sim":          "S2",
        "H_obs_mean":   float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec,
        "fits":          fits,
    }


# ===============================================================================
# 7. C0 null worker (Scheme C' generator at gamma = 0, all four estimators)
# ===============================================================================

def _c0_null_worker(args: tuple) -> dict:
    """
    C0 null replication (Scheme C' generator at gamma = 0).
    Generates consistent (obs_cat, p_dist) from the same vote simulation.
    """
    (rep, mu_arr, alpha_flat, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr,
     dlg_lengths_list, H_sorted_arr, q_sorted_arr) = args

    mu         = np.array(mu_arr)
    alpha      = np.array(alpha_flat).reshape(K, K)
    beta       = float(beta_val)
    gamma_use  = float(gamma_inj)   # 0.0 for C0 null
    H_dist     = np.array(H_dist_arr)
    H_gm       = float(H_gm_val)
    q_pool     = np.array(q_pool_arr)
    confusion  = np.array(confusion_arr)
    dlg_lengths = list(dlg_lengths_list)
    H_sorted   = np.array(H_sorted_arr)
    q_sorted   = np.array(q_sorted_arr)

    def _q_link(h: float) -> float:
        rank = int(np.searchsorted(H_sorted, h, side="left"))  # lower-bound rank
        rank = min(rank, len(H_sorted) - 1)
        q_idx = int(rank / len(H_sorted) * len(q_sorted))
        q_idx = min(q_idx, len(q_sorted) - 1)
        return float(q_sorted[q_idx])

    rng = np.random.default_rng(rep * 71831 + 11)

    sim_dlgs: List[List[dict]] = []
    H_obs_all: List[float] = []

    for dlg_len in dlg_lengths:
        if dlg_len < 2:
            continue

        # Draw H_raw from H_dist
        idx = (rng.integers(0, len(H_dist), size=dlg_len)).tolist()
        H_raw = np.array([H_dist[i] for i in idx], dtype=np.float64)

        # Pass 0: globally-centered H covariate
        H_cov = H_raw - H_gm
        cats  = _generate_cats_from_H(dlg_len, mu, alpha, beta, gamma_use, H_cov, rng)

        # Refinement passes (within-cell aligned)
        for _ in range(N_REFINE):
            H_cov = _compute_within_cell_H(cats, H_raw)
            cats  = _generate_cats_from_H(dlg_len, mu, alpha, beta, gamma_use, H_cov, rng)

        # Vote simulation (consistent obs_cat + p_dist from same votes)
        evs = []
        for m in range(dlg_len):
            q_base = _q_link(float(H_raw[m]))
            q      = float(np.clip(q_base + rng.normal(0.0, Q_NOISE_STD), 0.2, 1.0))
            obs, H_obs, plur, p_d = _sim_vote_pdist(
                rng, int(cats[m]), q, confusion, N_ANN_EL)
            evs.append({"cat": obs, "H": H_obs, "plurality": plur, "p_dist": p_d})
            H_obs_all.append(H_obs)
        sim_dlgs.append(evs)

    h1, h2 = [], []
    for dlg in sim_dlgs:
        for i in range(len(dlg) - 1):
            h1.append(dlg[i]["H"]); h2.append(dlg[i + 1]["H"])
    ha, hb = np.array(h1), np.array(h2)
    consec = (float(pearsonr(ha, hb)[0])
              if len(ha) >= 3 and ha.std() > 1e-9 and hb.std() > 1e-9
              else float("nan"))

    fits = _fit_all_four(sim_dlgs, rep_seed=rep)
    return {
        "rep":          rep,
        "sim":          "C0",
        "gamma_inj":    gamma_use,
        "H_obs_mean":   float(np.mean(H_obs_all)) if H_obs_all else float("nan"),
        "consec_H_corr": consec,
        "fits":          fits,
    }


# ===============================================================================
# 8. Power worker (Scheme C' generator at gamma != 0, hard_wc only)
# ===============================================================================

def _power_worker(args: tuple) -> dict:
    """
    Power replication: Scheme C' data with true gamma != 0, hard_wc fit.
    Uses numba kernel _process_one_dlg_scheme_c_nb for speed.
    """
    (rep, mu_arr, alpha_flat, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr,
     dlg_lengths_list, H_sorted_arr, q_sorted_arr) = args

    mu     = np.array(mu_arr)
    alpha  = np.array(alpha_flat)
    beta   = float(beta_val)
    H_dist = np.array(H_dist_arr)
    H_gm   = float(H_gm_val)
    q_pool = np.array(q_pool_arr)
    conf   = np.array(confusion_arr)
    H_sort = np.array(H_sorted_arr)
    q_sort = np.array(q_sorted_arr)
    dlg_lengths = list(dlg_lengths_list)

    g_seed = int(round((gamma_inj + 2.0) * 10000))
    _seed_nb(rep * 73919 + g_seed)

    obs_dlgs: List[List[dict]] = []
    for dlg_len in dlg_lengths:
        if dlg_len < 2:
            continue
        (oc, oh, op, sc, sh, sp, _) = _process_one_dlg_scheme_c_nb(
            dlg_len, mu, alpha.reshape(K, K), beta, gamma_inj,
            H_dist, H_gm, H_sort, q_sort, conf, Q_NOISE_STD, N_REFINE, K, N_ANN_EL,
        )
        obs_dlgs.append([
            {"cat": int(sc[m]), "H": float(sh[m]), "plurality": int(sp[m])}
            for m in range(dlg_len)
        ])

    obs_f = [d for d in obs_dlgs if len(d) >= 2]
    hard_wc = {"gamma_hat": float("nan"), "converged": False, "at_bound": False}
    if len(obs_f) >= 5:
        wc = fit_within_cell(obs_f, seed=rep)
        hard_wc = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
        }
    return {
        "rep":        rep,
        "gamma_inj":  float(gamma_inj),
        "hard_wc":    hard_wc,
    }


# ===============================================================================
# 9. Runner with checkpointing
# ===============================================================================

def _run_pool(
    worker_fn,
    args_list: list,
    ckpt_path: Path,
    key_name: str,
    done_key_fn,     # function(result) -> hashable key
    args_key_fn,     # function(arg_tuple) -> hashable key
    n_workers: int = 6,
) -> list:
    done_keys: set = set()
    results: list = []
    if ckpt_path.exists():
        try:
            data = json.loads(ckpt_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_keys = {done_key_fn(r) for r in results}
            log.info(f"  Checkpoint {ckpt_path.name}: {len(done_keys)} done")
        except Exception as ex:
            log.warning(f"  Checkpoint load failed ({ex}); fresh start")

    pending = [a for a in args_list if args_key_fn(a) not in done_keys]
    if not pending:
        log.info(f"  All {len(args_list)} tasks in checkpoint")
        return results

    log.info(f"  {len(pending)}/{len(args_list)} pending on {n_workers} workers...")

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
                log.info(f"  [{ts}] {n_done}/{len(args_list)} done")
                _save()
                unsaved = 0
    _save()
    log.info(f"  Checkpoint -> {ckpt_path}")
    return results


# ===============================================================================
# 10. Statistical analysis
# ===============================================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    half   = (z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
              ) / (1 + z * z / n)
    return max(0.0, center - half), min(1.0, center + half)


def emp_p_median_centered(null_hats: np.ndarray, real_hat: float) -> dict:
    """
    Two-sided empirical p-value centred at the null median.
    p = (1 + #{|null_b - null_median| >= |real - null_median|}) / (B+1)
    """
    B = len(null_hats)
    if B == 0:
        return {
            "p": float("nan"), "mc_se": float("nan"),
            "null_median": float("nan"),
            "n_lt": 0, "n_gt": 0, "B": 0,
        }
    null_med = float(np.median(null_hats))
    stat_b   = np.abs(null_hats - null_med)
    stat_r   = abs(real_hat - null_med)
    count    = int((stat_b >= stat_r).sum())
    p        = (1 + count) / (B + 1)
    mc_se    = math.sqrt(p * (1 - p) / B) if B > 0 else float("nan")
    n_lt     = int((null_hats < real_hat).sum())
    n_gt     = int((null_hats > real_hat).sum())
    return {
        "p": p, "mc_se": mc_se,
        "null_median": null_med,
        "n_lt": n_lt, "n_gt": n_gt, "B": B,
    }


def holm_adjust(p_values: List[float]) -> List[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adj = [0.0] * n
    running_max = 0.0
    for rank, i in enumerate(order):
        adj_p = min(1.0, p_values[i] * (n - rank))
        running_max = max(running_max, adj_p)
        adj[i] = running_max
    return adj


def analyse_cell(
    null_results: list,
    sim_tag: str,
    estimator_tag: str,
    real_gamma: float,
) -> dict:
    """Extract statistics for one (estimator, simulator) cell."""
    gamma_hats = []
    n_conv = 0
    n_at_bound = 0
    for r in null_results:
        if r.get("sim") != sim_tag:
            continue
        fits = r.get("fits", {})
        fg   = fits.get(estimator_tag, {})
        gh   = fg.get("gamma_hat", float("nan"))
        if not math.isnan(gh):
            gamma_hats.append(gh)
        if fg.get("converged", False):
            n_conv += 1
        if fg.get("at_bound", False):
            n_at_bound += 1

    total_n = sum(1 for r in null_results if r.get("sim") == sim_tag)

    if len(gamma_hats) < 2:
        return {
            "estimator": estimator_tag, "sim": sim_tag,
            "real_gamma": real_gamma,
            "null_mean": float("nan"), "null_std": float("nan"),
            "null_q2_5": float("nan"), "null_q97_5": float("nan"),
            "null_median": float("nan"),
            "emp_p": float("nan"), "mc_se": float("nan"),
            "n_lt": 0, "n_gt": 0, "B": len(gamma_hats),
            "reject": False,
            "n_conv": n_conv, "n_at_bound": n_at_bound, "n_total": total_n,
            "note": "insufficient converged reps",
        }

    arr     = np.array(gamma_hats)
    q2_5    = float(np.percentile(arr, 2.5))
    q97_5   = float(np.percentile(arr, 97.5))
    null_mean = float(arr.mean())
    null_std  = float(arr.std(ddof=1))
    reject    = not (q2_5 <= real_gamma <= q97_5)
    pstats    = emp_p_median_centered(arr, real_gamma)

    return {
        "estimator":   estimator_tag, "sim": sim_tag,
        "real_gamma":  real_gamma,
        "null_mean":   null_mean, "null_std": null_std,
        "null_q2_5":   q2_5, "null_q97_5": q97_5,
        "null_median": pstats["null_median"],
        "emp_p":       pstats["p"], "mc_se": pstats["mc_se"],
        "n_lt":        pstats["n_lt"], "n_gt": pstats["n_gt"],
        "B":           len(gamma_hats),
        "reject":      reject,
        "n_conv":      n_conv, "n_at_bound": n_at_bound, "n_total": total_n,
    }


def analyse_x3(null_s1, null_s2, null_c0, real_gammas_dict) -> dict:
    """X3 decision table for the 4 estimators x 3 simulators."""
    estimators = ["hard_wc", "hard_rh", "soft_wc", "soft_rh"]
    sims       = [("S1", null_s1), ("S2", null_s2), ("C0", null_c0)]
    table: dict = {}

    for est in estimators:
        for sim_tag, null_res in sims:
            rg = real_gammas_dict.get(est, float("nan"))
            cell = analyse_cell(null_res, sim_tag, est, rg)
            table[f"{est}_{sim_tag}"] = cell

    # Holm adjustment, main family (hard_wc and hard_rh x 3 simulators)
    mt_keys  = [f"{e}_{s}" for e in ["hard_wc", "hard_rh"] for s in ["S1", "S2", "C0"]]
    mt_pvals = [table[k]["emp_p"] for k in mt_keys
                if not math.isnan(table[k]["emp_p"])]
    mt_keys_valid = [k for k in mt_keys if not math.isnan(table[k]["emp_p"])]
    if mt_pvals:
        adj_mt = holm_adjust(mt_pvals)
        for k, adj_p in zip(mt_keys_valid, adj_mt):
            table[k]["holm_p_main"] = adj_p
            table[k]["reject_holm_main"] = (adj_p < 0.05)
    for k in mt_keys:
        if "holm_p_main" not in table.get(k, {}):
            table[k]["holm_p_main"] = float("nan")
            table[k]["reject_holm_main"] = False

    # Holm adjustment, full family (all 12 tests)
    all_keys   = [f"{e}_{s}" for e in estimators for s in ["S1", "S2", "C0"]]
    all_pvals  = [table[k]["emp_p"] for k in all_keys
                  if not math.isnan(table[k]["emp_p"])]
    all_keys_v = [k for k in all_keys if not math.isnan(table[k]["emp_p"])]
    if all_pvals:
        adj_all = holm_adjust(all_pvals)
        for k, adj_p in zip(all_keys_v, adj_all):
            table[k]["holm_p_full"] = adj_p
            table[k]["reject_holm_full"] = (adj_p < 0.05)
    for k in all_keys:
        if "holm_p_full" not in table.get(k, {}):
            table[k]["holm_p_full"] = float("nan")
            table[k]["reject_holm_full"] = False

    return table


def analyse_power_x2(
    power_results: list,
    s1_null_q2_5: float, s1_null_q97_5: float,
    s2_null_q2_5: float, s2_null_q97_5: float,
) -> dict:
    """Power (rejection rate with Wilson CI) per (true gamma, null band)."""
    out = {}
    for gamma_val in POWER_GAMMA_GRID:
        hats = []
        for r in power_results:
            if abs(r["gamma_inj"] - gamma_val) < 1e-9:
                gh = r["hard_wc"]["gamma_hat"]
                if not math.isnan(gh):
                    hats.append(gh)
        n = len(hats)
        arr = np.array(hats)

        for sim_tag, q25, q975 in [("S1", s1_null_q2_5, s1_null_q97_5),
                                    ("S2", s2_null_q2_5, s2_null_q97_5)]:
            if n == 0 or math.isnan(q25):
                out[f"power_{gamma_val}_{sim_tag}"] = {
                    "gamma_true": gamma_val, "sim_band": sim_tag,
                    "n_reps": n, "n_reject": 0,
                    "reject_rate": float("nan"),
                    "wilson_lo": float("nan"), "wilson_hi": float("nan"),
                }
                continue
            n_rej = int(((arr < q25) | (arr > q975)).sum())
            lo, hi = wilson_ci(n_rej, n)
            out[f"power_{gamma_val}_{sim_tag}"] = {
                "gamma_true":  gamma_val, "sim_band": sim_tag,
                "n_reps":      n, "n_reject": n_rej,
                "reject_rate": n_rej / n,
                "wilson_lo": lo, "wilson_hi": hi,
            }
    return out


def analyse_cross_type1_x2(
    x3_table: dict,
    null_s1: list, null_s2: list, null_c0: list,
    s1_null_q2_5: float, s1_null_q97_5: float,
    s2_null_q2_5: float, s2_null_q97_5: float,
) -> dict:
    """Cross-simulator type-I error rates."""
    out = {}

    def _count_reject_from_list(results, sim_tag, q25, q975, label):
        hats = [r["fits"]["hard_wc"]["gamma_hat"] for r in results
                if r.get("sim") == sim_tag
                and not math.isnan(r.get("fits", {}).get("hard_wc", {}).get("gamma_hat", float("nan")))]
        n    = len(hats)
        if n == 0 or math.isnan(q25):
            return {"label": label, "n": n, "n_reject": 0,
                    "rate": float("nan"), "wilson_lo": float("nan"), "wilson_hi": float("nan")}
        arr  = np.array(hats)
        n_rej = int(((arr < q25) | (arr > q975)).sum())
        lo, hi = wilson_ci(n_rej, n)
        return {"label": label, "n": n, "n_reject": n_rej,
                "rate": n_rej / n, "wilson_lo": lo, "wilson_hi": hi}

    # S1 gamma=0 data tested against S2 null band
    out["S1_vs_S2"] = _count_reject_from_list(null_s1, "S1", s2_null_q2_5, s2_null_q97_5, "S1->S2")
    # S2 gamma=0 data tested against S1 null band
    out["S2_vs_S1"] = _count_reject_from_list(null_s2, "S2", s1_null_q2_5, s1_null_q97_5, "S2->S1")
    # C0 gamma=0 data tested against S1 null band
    out["C0_vs_S1"] = _count_reject_from_list(null_c0, "C0", s1_null_q2_5, s1_null_q97_5, "C0->S1")
    # C0 gamma=0 data tested against S2 null band
    out["C0_vs_S2"] = _count_reject_from_list(null_c0, "C0", s2_null_q2_5, s2_null_q97_5, "C0->S2")
    return out


# ===============================================================================
# 11. X6: S1' calibration + null band
# ===============================================================================

def _run_s1_prime_cal_reps(
    args_base_fn,   # function(rep) -> args (with delta in last position)
    delta: float,
    cal_reps: int,
    cal_rep_offset: int,   # starting rep index for cal reps
    n_workers: int,
) -> float:
    """Run calibration reps and return mean H_obs_mean."""
    args_list = [args_base_fn(rep + cal_rep_offset, delta) for rep in range(cal_reps)]
    ctx = mp.get_context("spawn")
    h_means = []
    with ctx.Pool(n_workers) as pool:
        for r in pool.imap_unordered(_s1_null_worker, args_list):
            h = r.get("H_obs_mean", float("nan"))
            if not math.isnan(h):
                h_means.append(h)
    return float(np.mean(h_means)) if h_means else float("nan")


def calibrate_delta_x6(
    real_H_mean: float,
    args_base_fn,
    n_workers: int,
    n_cal_reps: int = N_X6_CAL_REPS,
    cal_rep_offset: int = 200,
    tol: float = 0.005,
    max_iter: int = 25,
) -> dict:
    """Bisect delta so that the simulated EL-wide mean entropy matches the real one (within tol)."""
    lo, hi = -0.5, 0.5
    best_delta = 0.0
    best_diff  = float("inf")
    converged  = False

    log.info(f"  [X6] Bisecting delta: real H_mean = {real_H_mean:.4f}")
    log.info(f"  [X6] Cal reps: {n_cal_reps}, reps {cal_rep_offset}-{cal_rep_offset+n_cal_reps-1}")

    for it in range(max_iter):
        delta_mid = (lo + hi) / 2.0
        sim_h = _run_s1_prime_cal_reps(
            args_base_fn, delta_mid, n_cal_reps, cal_rep_offset, n_workers)
        diff  = sim_h - real_H_mean
        log.info(f"  [X6] iter {it+1}: delta={delta_mid:+.4f}  sim_H={sim_h:.4f}  diff={diff:+.4f}")
        if abs(diff) < best_diff:
            best_diff  = abs(diff)
            best_delta = delta_mid
        if abs(diff) < tol:
            converged = True
            break
        # a larger delta raises the accuracy and lowers the simulated entropy
        if diff > 0:
            lo = delta_mid
        else:
            hi = delta_mid

    log.info(f"  [X6] Calibration: best delta={best_delta:+.4f}  "
             f"|diff|={best_diff:.4f}  converged={converged}")
    return {"delta": best_delta, "diff": best_diff, "converged": converged,
            "n_cal_reps": n_cal_reps, "cal_rep_offset": cal_rep_offset}


# ===============================================================================
# 12. Main
# ===============================================================================

def _hard_wc_summary(res: list, real: float) -> dict:
    """Null summary and median-centred two-sided p for hard_wc (converged reps only)."""
    v = np.array([r["fits"]["hard_wc"]["gamma_hat"] for r in res
                  if r["fits"]["hard_wc"].get("converged") and np.isfinite(r["fits"]["hard_wc"]["gamma_hat"])])
    med = float(np.median(v))
    k = int((np.abs(v - med) >= abs(real - med)).sum())
    p = (1 + k) / (len(v) + 1)
    return {"B": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "q2_5": float(np.quantile(v, 0.025)), "q97_5": float(np.quantile(v, 0.975)),
            "p_two_sided": p, "mc_se": float(np.sqrt(p * (1 - p) / len(v))),
            "reject": bool(not (np.quantile(v, 0.025) <= real <= np.quantile(v, 0.975)))}


def main(mode: str, out_dir: Path, n_null_reps: int, n_workers: int) -> None:
    """mode: "table" (E35), "bigB" (E38) or "ppc" (E39)."""
    assert mode in ("table", "bigB", "ppc"), mode
    os.environ[MODE_ENV] = mode
    t0 = time.time()

    log.info("[1] Loading EL data ...")
    dlgs_fr, dlgs_ep = load_el_data()
    dlgs_el = dlgs_fr + dlgs_ep
    log.info(f"  Friends: {len(dlgs_fr)} dlgs, EmotionPush: {len(dlgs_ep)} dlgs, "
             f"EL-wide: {len(dlgs_el)} dlgs, {sum(len(d) for d in dlgs_el)} events")

    real_H_mean_el = compute_H_mean(dlgs_el)
    real_consec_el = compute_consec_H_corr(dlgs_el)

    q_pool_el = compute_empirical_q_pool(dlgs_el, n_ann=N_ANN_EL)
    H_dist_el = compute_empirical_H_pool(dlgs_el)
    H_sorted = np.sort(H_dist_el)
    q_sorted = np.sort(q_pool_el)[::-1]
    H_gm_el = float(H_dist_el.mean())

    fc_fr, fH_fr, starts_fr = make_flat_arrays(dlgs_fr)
    fc_ep, fH_ep, starts_ep = make_flat_arrays(dlgs_ep)

    conf_el = compute_confusion(dlgs_el)

    log.info("[2] AR(1) parameters (E30_fit.json) ...")
    e30 = json.loads((config.RESULTS_ROOT / "E30" / "E30_fit.json").read_text())
    ar_fr = (e30["el_friends"]["a"], e30["el_friends"]["b"],
             e30["el_friends"]["rho"], e30["el_friends"]["confusion"])
    ar_ep = (e30["el_emotionpush"]["a"], e30["el_emotionpush"]["b"],
             e30["el_emotionpush"]["rho"], e30["el_emotionpush"]["confusion"])

    log.info("[3] DT-AMHP fit (generator of C0 and power data) ...")
    raw_dt = [
        {"cats": np.array([e["cat"] for e in d], dtype=np.int64),
         "Hs_raw": np.array([e["H"] for e in d], dtype=np.float64)}
        for d in dlgs_el
    ]
    dlgs_r = residualize_within_cell(raw_dt)
    H_bar_r = global_H_mean(dlgs_r)
    dlgs_dtr = prepare_dt_dlgs(dlgs_r, H_bar_r)
    real_fit = fit_dt_bounded(dlgs_dtr, H_bar_r, n_restarts=3, seed=0)
    log.info(f"  gamma_hat (hard_wc) = {real_fit['gamma_hat']:+.4f}")

    if real_fit["v_hat"]:
        mu_r, alpha_r, beta_r, _ = unpack_v_dt(np.array(real_fit["v_hat"]), K)
    else:
        mu_r = np.ones(K) / K * 0.2
        alpha_r = np.ones((K, K)) * 0.05
        beta_r = 1.0

    alpha_flat = alpha_r.ravel()
    dlg_lengths_el = [len(d) for d in dlgs_el]

    log.info("[4] numba JIT warm-up ...")
    _warmup_numba(K)

    log.info("[5] Real-data gamma_hat for the estimators ...")
    real_gammas = _fit_all_four(dlgs_el, rep_seed=0, n_restarts_hard=3, n_restarts_soft=3)
    real_g_dict = {k: real_gammas[k]["gamma_hat"] for k in ["hard_wc", "hard_rh", "soft_wc", "soft_rh"]}
    log.info("  " + "  ".join(f"{k}: {v:+.4f}" for k, v in real_g_dict.items()))

    def _arr(a):
        return a.tolist() if hasattr(a, "tolist") else list(a)

    def _make_s1_args(rep: int, delta: float = 0.0):
        return (rep,
                _arr(fc_fr), _arr(fH_fr), _arr(starts_fr),
                _arr(fc_ep), _arr(fH_ep), _arr(starts_ep),
                conf_el.tolist(),
                _arr(q_pool_el), _arr(H_dist_el),
                delta)

    def _make_s2_args(rep: int):
        return (rep,
                _arr(fc_fr), _arr(starts_fr), ar_fr[3],
                ar_fr[0], ar_fr[1], ar_fr[2],
                _arr(fc_ep), _arr(starts_ep), ar_ep[3],
                ar_ep[0], ar_ep[1], ar_ep[2], N_ANN_EL)

    def _make_c0_args(rep: int, gamma_inj: float = 0.0):
        return (rep, _arr(mu_r), _arr(alpha_flat), float(beta_r), gamma_inj,
                _arr(H_dist_el), H_gm_el, _arr(q_pool_el), conf_el.tolist(),
                dlg_lengths_el, _arr(H_sorted), _arr(q_sorted))

    s1_args = [_make_s1_args(rep) for rep in range(n_null_reps)]
    s2_args = [_make_s2_args(rep) for rep in range(n_null_reps)]
    c0_args = [_make_c0_args(rep) for rep in range(n_null_reps)]
    power_args = [_make_c0_args(rep, g) for g in POWER_GAMMA_GRID for rep in range(N_POWER_REPS)]
    by_rep = dict(done_key_fn=lambda r: r["rep"], args_key_fn=lambda a: a[0], n_workers=n_workers)

    if mode == "ppc":
        real = ppc_stats([[{"H": e["H"], "p_dist": e["p_dist"], "cat": e["cat"]} for e in d] for d in dlgs_el])
        res = {"real": real}
        ctx_q = mp.get_context("spawn")
        for name, fn, mk in (("S1", _s1_null_worker, _make_s1_args), ("S2", _s2_el_wide_null_worker, _make_s2_args),
                             ("C0", _c0_null_worker, _make_c0_args)):
            with ctx_q.Pool(n_workers) as pool:
                rows = list(pool.imap_unordered(fn, [mk(r) for r in range(1, 21)]))
            stats = [r["fits"]["ppc"] for r in rows]
            res[name] = {k: {"mean": float(np.mean([x[k] for x in stats])), "sd": float(np.std([x[k] for x in stats]))}
                         for k in stats[0]}
        (out_dir / "E39_ppc.json").write_text(json.dumps(res, indent=1))
        return

    if mode == "bigB":
        null_s2 = _run_pool(_s2_el_wide_null_worker, s2_args, out_dir / "E38_ckpt_s2.json", "results", **by_rep)
        null_c0 = _run_pool(_c0_null_worker, c0_args, out_dir / "E38_ckpt_c0.json", "results", **by_rep)
        real = real_g_dict["hard_wc"]
        summ = {"real_hard_wc": real}
        for name, res in (("S2", null_s2), ("C0", null_c0)):
            summ[name] = _hard_wc_summary(res, real)
        (out_dir / "E38_summary.json").write_text(json.dumps(summ, indent=1))
        log.info(json.dumps(summ, indent=1))
        return

    log.info("[6] Pilot (3 reps per simulator) for a runtime estimate ...")
    ctx_p = mp.get_context("spawn")
    t_p = time.time()
    with ctx_p.Pool(n_workers) as pool:
        list(pool.imap_unordered(_s1_null_worker, [_make_s1_args(rep + 9000) for rep in range(N_CAL_PILOT)]))
    t_s1_per_rep = (time.time() - t_p) * n_workers / N_CAL_PILOT
    t_p = time.time()
    with ctx_p.Pool(n_workers) as pool:
        list(pool.imap_unordered(_s2_el_wide_null_worker, [_make_s2_args(rep + 9000) for rep in range(N_CAL_PILOT)]))
    t_s2_per_rep = (time.time() - t_p) * n_workers / N_CAL_PILOT
    t_p = time.time()
    with ctx_p.Pool(n_workers) as pool:
        list(pool.imap_unordered(_c0_null_worker, [_make_c0_args(rep + 9000) for rep in range(N_CAL_PILOT)]))
    t_c0_per_rep = (time.time() - t_p) * n_workers / N_CAL_PILOT
    log.info(f"  per-rep seconds: S1 {t_s1_per_rep:.1f}, S2 {t_s2_per_rep:.1f}, C0 {t_c0_per_rep:.1f}")

    log.info(f"[7] X3 null bands ({n_null_reps} reps x S1, S2, C0) ...")
    null_s1 = _run_pool(_s1_null_worker, s1_args, out_dir / "E35_ckpt_s1_null.json", "results", **by_rep)
    null_s2 = _run_pool(_s2_el_wide_null_worker, s2_args, out_dir / "E35_ckpt_s2_null.json", "results", **by_rep)
    null_c0 = _run_pool(_c0_null_worker, c0_args, out_dir / "E35_ckpt_c0_null.json", "results", **by_rep)

    log.info(f"[8] X2 power ({N_POWER_REPS} reps x {len(POWER_GAMMA_GRID)} gamma values) ...")
    power_results = _run_pool(
        _power_worker, power_args, out_dir / "E35_ckpt_power.json", "results",
        done_key_fn=lambda r: (r["rep"], r["gamma_inj"]),
        args_key_fn=lambda a: (a[0], float(a[4])),
        n_workers=n_workers,
    )

    log.info("[9] X6 calibration of the S1' accuracy offset ...")
    x6_cal = calibrate_delta_x6(
        real_H_mean=real_H_mean_el,
        args_base_fn=_make_s1_args,
        n_workers=n_workers,
        n_cal_reps=N_X6_CAL_REPS,
        cal_rep_offset=200,
    )
    delta_x6 = x6_cal["delta"]

    log.info(f"[10] X6 S1' null band ({N_X6_NULL_REPS} reps, delta={delta_x6:+.4f}) ...")
    s1prime_args = [_make_s1_args(rep, delta_x6) for rep in range(N_X6_NULL_REPS)]
    null_s1prime = _run_pool(_s1_null_worker, s1prime_args, out_dir / "E35_ckpt_s1prime_null.json", "results",
                             **by_rep)

    log.info("[11] Aggregating ...")
    x3_table = analyse_x3(null_s1, null_s2, null_c0, real_g_dict)

    s1_cell_hwc = x3_table.get("hard_wc_S1", {})
    s2_cell_hwc = x3_table.get("hard_wc_S2", {})
    bands = dict(
        s1_null_q2_5=s1_cell_hwc.get("null_q2_5", float("nan")),
        s1_null_q97_5=s1_cell_hwc.get("null_q97_5", float("nan")),
        s2_null_q2_5=s2_cell_hwc.get("null_q2_5", float("nan")),
        s2_null_q97_5=s2_cell_hwc.get("null_q97_5", float("nan")),
    )
    x2_power = analyse_power_x2(power_results, **bands)
    x2_cross = analyse_cross_type1_x2(x3_table, null_s1, null_s2, null_c0, **bands)

    s1p_hwc_hats = [
        r["fits"]["hard_wc"]["gamma_hat"]
        for r in null_s1prime
        if not math.isnan(r.get("fits", {}).get("hard_wc", {}).get("gamma_hat", float("nan")))
    ]
    s1p_H_means = [r["H_obs_mean"] for r in null_s1prime
                   if not math.isnan(r.get("H_obs_mean", float("nan")))]
    s1p_consec = [r["consec_H_corr"] for r in null_s1prime
                  if not math.isnan(r.get("consec_H_corr", float("nan")))]

    if s1p_hwc_hats:
        sp_arr = np.array(s1p_hwc_hats)
        sp_q25 = float(np.percentile(sp_arr, 2.5))
        sp_q975 = float(np.percentile(sp_arr, 97.5))
        sp_rej = not (sp_q25 <= real_g_dict["hard_wc"] <= sp_q975)
        sp_hmean = float(np.mean(s1p_H_means)) if s1p_H_means else float("nan")
        sp_consec_mean = float(np.mean(s1p_consec)) if s1p_consec else float("nan")
    else:
        sp_q25 = sp_q975 = sp_hmean = sp_consec_mean = float("nan")
        sp_rej = False

    x6_decision = {
        "q2_5": sp_q25, "q97_5": sp_q975,
        "real_gamma": real_g_dict["hard_wc"],
        "reject": sp_rej,
        "H_mean": sp_hmean,
        "consec_H_corr_mean": sp_consec_mean,
    }

    def _mean_stat(res_list, sim_tag, stat_key):
        vals = [r[stat_key] for r in res_list
                if r.get("sim") == sim_tag and not math.isnan(r.get(stat_key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    sanity = {
        "real": {"H_mean": real_H_mean_el, "consec_H_corr": real_consec_el},
        "S1": {"H_mean": _mean_stat(null_s1, "S1", "H_obs_mean"),
               "consec_H_corr": _mean_stat(null_s1, "S1", "consec_H_corr")},
        "S2": {"H_mean": _mean_stat(null_s2, "S2", "H_obs_mean"),
               "consec_H_corr": _mean_stat(null_s2, "S2", "consec_H_corr")},
        "C0": {"H_mean": _mean_stat(null_c0, "C0", "H_obs_mean"),
               "consec_H_corr": _mean_stat(null_c0, "C0", "consec_H_corr")},
        "S1'": {"H_mean": sp_hmean, "consec_H_corr": sp_consec_mean},
    }

    out = {
        "experiment": "E35_power_table",
        "config": {
            "N_NULL_REPS": n_null_reps, "N_POWER_REPS": N_POWER_REPS,
            "N_WORKERS": n_workers, "K": K, "N_ANN_EL": N_ANN_EL,
            "POWER_GAMMA_GRID": POWER_GAMMA_GRID,
            "REAL_GAMMA_EL_WIDE_REF": REAL_GAMMA_EL_WIDE,
            "n_reps_pilot": N_CAL_PILOT,
            "seeds": {
                "s1": "rep * 41117 + 3",
                "s2_fr": "rep * 53117 + 22",
                "s2_ep": "rep * 57239 + 100",
                "c0": "rep * 71831 + 11",
                "power": "rep * 73919 + g_seed",
                "x6_cal": "rep * 41117 + 3, rep in 200-219",
            },
        },
        "ar1_params": {
            "friends": {"a": ar_fr[0], "b": ar_fr[1], "rho": ar_fr[2]},
            "emotionpush": {"a": ar_ep[0], "b": ar_ep[1], "rho": ar_ep[2]},
        },
        "real_gammas": real_g_dict,
        "real_H_mean": real_H_mean_el,
        "real_consec": real_consec_el,
        "x3_table": x3_table,
        "x2_power": x2_power,
        "x2_cross_type1": x2_cross,
        "x6": {
            "delta": delta_x6,
            "calibration": x6_cal,
            "decision": x6_decision,
        },
        "sanity": sanity,
        "pilot_times": {
            "s1_per_rep": t_s1_per_rep, "s2_per_rep": t_s2_per_rep,
            "c0_per_rep": t_c0_per_rep,
        },
        "elapsed_sec": float(time.time() - t0),
    }

    results_path = out_dir / "E35_results.json"
    results_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: (o.item() if hasattr(o, "item") else str(o))),
        encoding="utf-8",
    )
    log.info(f"Results -> {results_path}")
