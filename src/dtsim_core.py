"""
EmotionLines loading, within-cell residualisation and the bounded DT-AMHP fit used throughout.

An event is a dict with
    votes (K,), p (K,) = votes / sum, H = vote entropy (nats), cat = argmax(votes), plurality = max(votes).
EL-wide = Friends + EmotionPush (EmotionX 2019 release), 5 annotators per utterance.
"""

from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

import config
from estimator_dt import DTEstimator, unpack_v_dt, n_params_dt, _dt_single_nll_grad  # noqa: F401

K = 7                   # emotion categories: neutral, joy, sadness, fear, anger, surprise, disgust
DT_L1 = 0.001           # L1 penalty on alpha
GAMMA_BOUND = 5.0       # gamma is bounded to [-GAMMA_BOUND, GAMMA_BOUND]
GAMMA_CONV_THR = 4.5    # |gamma_hat| >= this is flagged as at the bound (not converged)
N_RESTARTS = 3          # restarts for the real-data fit

# EL-wide within-cell gamma_hat of the hard-mark estimator (reproduced by the real-data fits below)
REAL_GAMMA_WITHINCELL_FULL = -0.3915


# ============================================================================
# Data loading
# ============================================================================

def load_el_raw() -> List[List[dict]]:
    """Load EL-wide dialogues (Friends then EmotionPush); dialogues with < 2 events are dropped."""
    dlgs = []
    for path in (config.FRIENDS_JSON, config.EMOTIONPUSH_JSON):
        for dialog in json.load(open(path)):
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
                    "votes": votes,
                    "p": p,
                    "H": H,
                    "cat": int(p.argmax()),
                    "plurality": int(votes.max()),
                })
            if len(evs) >= 2:
                dlgs.append(evs)
    return dlgs


def compute_empirical_confusion(dlgs: List[List[dict]]) -> np.ndarray:
    """Row-normalised confusion matrix (zero diagonal) from the off-plurality votes."""
    C = np.zeros((K, K))
    for d in dlgs:
        for e in d:
            i = e["cat"]
            for j in range(K):
                if j != i:
                    C[i, j] += e["votes"][j]
    for i in range(K):
        row_sum = C[i].sum()
        if row_sum > 0:
            C[i] /= row_sum
        else:
            C[i] = np.ones(K) / (K - 1)
            C[i, i] = 0.0
    return C


def empirical_q_pool(dlgs: List[List[dict]]) -> np.ndarray:
    """Pool of q = plurality / 5 over all utterances."""
    return np.array([e["plurality"] / 5.0 for d in dlgs for e in d])


def empirical_H_pool(dlgs: List[List[dict]]) -> np.ndarray:
    """Pool of vote entropies over all utterances."""
    return np.array([e["H"] for d in dlgs for e in d])


# ============================================================================
# Within-cell residualisation
# ============================================================================

def residualize_within_cell(dlgs: List[dict]) -> List[dict]:
    """
    H_resid = H - mean(H | category, dialogue).
    Input: list of {'cats': ndarray, 'Hs_raw': ndarray}; returns a new list.
    """
    new_dlgs = []
    for d in dlgs:
        cats = d["cats"]
        H = d["Hs_raw"]
        H_resid = H.copy()
        for c in range(K):
            idx = np.where(cats == c)[0]
            if len(idx) > 0:
                H_resid[idx] -= H[idx].mean()
        nd = dict(d)
        nd["Hs_raw"] = H_resid
        new_dlgs.append(nd)
    return new_dlgs


def prepare_dt_dlgs(dlgs_resid: List[dict], H_bar: float) -> List[dict]:
    """Subtract the global mean and return estimator-ready dicts {'cats', 'Hs_c'}."""
    return [{"cats": d["cats"], "Hs_c": d["Hs_raw"] - H_bar} for d in dlgs_resid]


def global_H_mean(dlgs: List[dict]) -> float:
    return float(np.concatenate([d["Hs_raw"] for d in dlgs]).mean())


# ============================================================================
# Bounded DT-AMHP fit
# ============================================================================

def fit_dt_bounded(
    dlgs_dt: List[dict],
    H_bar: float,
    n_restarts: int = N_RESTARTS,
    seed: int = 0,
    init_from: Optional[np.ndarray] = None,
    maxiter: int = 5000,
    gamma_bound: float = GAMMA_BOUND,
) -> dict:
    """
    Fit the DT-AMHP with gamma bounded to [-gamma_bound, gamma_bound].
    Returns gamma_hat, nll, converged (|gamma_hat| < GAMMA_CONV_THR and optimiser success),
    at_bound, success and v_hat.
    """
    n_p = n_params_dt(K)
    est = DTEstimator(dlgs_dt, K=K, H_bar=H_bar, l1_alpha=DT_L1)

    bounds = [(None, None)] * n_p
    bounds[K + K * K] = (np.log(0.05), np.log(5.0))        # beta
    bounds[K + K * K + 1] = (-gamma_bound, gamma_bound)     # gamma

    rng = np.random.default_rng(seed)
    starts = []
    if init_from is not None:
        starts.append(np.clip(init_from.copy(), -20, 20))
    while len(starts) < n_restarts:
        v0 = np.zeros(n_p)
        v0[:K] = rng.uniform(-0.5, 0.5, K)
        v0[K:K + K * K] = rng.uniform(-2.0, -0.5, K * K)
        v0[K + K * K] = rng.uniform(np.log(0.3), np.log(1.5))
        v0[K + K * K + 1] = rng.uniform(-0.5, 0.5)
        starts.append(v0)

    best_f = np.inf
    best_x = None
    best_success = False

    for v0 in starts:
        try:
            res = minimize(
                est._obj, v0, method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                best_success = bool(res.success)
        except Exception:
            pass

    if best_x is None:
        return {
            "gamma_hat": float("nan"), "nll": float("nan"),
            "converged": False, "at_bound": False, "v_hat": None,
        }

    _, _, _, gamma_hat = unpack_v_dt(best_x, K)
    at_bound = abs(gamma_hat) >= GAMMA_CONV_THR
    converged = (not at_bound) and best_success

    return {
        "gamma_hat": float(gamma_hat),
        "nll": float(best_f),
        "converged": converged,
        "at_bound": at_bound,
        "success": best_success,
        "v_hat": best_x.tolist(),
    }


# ============================================================================
# Pure-Python DT-AMHP category generator (numba versions in dtsim_kernels.py)
# ============================================================================

def _generate_cats_from_H(
    dlg_len: int,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma_inj: float,
    H_cov: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sequential DT-AMHP generation with an arbitrary modifier H_cov (length dlg_len)."""
    cats = np.empty(dlg_len, dtype=np.int64)
    R = np.zeros(K)
    ef = math.exp(-beta)
    for m in range(dlg_len):
        Lam = mu.copy()
        for j in range(K):
            for i in range(K):
                Lam[i] += alpha[i, j] * R[j]
        Lam = np.maximum(Lam, 1e-300)
        prob = Lam / Lam.sum()
        cats[m] = int(rng.choice(K, p=prob))
        gain = math.exp(float(np.clip(gamma_inj * H_cov[m], -30, 30)))
        ik = cats[m]
        R = ef * (R + np.where(np.arange(K) == ik, gain, 0.0))
    return cats


def _compute_within_cell_H(
    cats: np.ndarray,
    H_raw: np.ndarray,
) -> np.ndarray:
    """Within-cell residuals of H_raw (size-1 cells set to 0), then globally re-centred."""
    H_wc = H_raw.copy()
    for c in range(K):
        idx = np.where(cats == c)[0]
        if len(idx) > 1:
            H_wc[idx] -= H_raw[idx].mean()
        elif len(idx) == 1:
            H_wc[idx[0]] = 0.0
    H_wc -= H_wc.mean()
    return H_wc
