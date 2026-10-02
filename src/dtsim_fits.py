"""
Hard-mark estimators applied to (simulated or real) observed data, and aggregation helpers.

Each event passed to the fits is a dict with 'cat' (observed majority label), 'H' (modifier) and
'plurality'. Within-cell (wc): H is residualised within (dialogue, observed category) cells and globally
centred. Raw-H (rh): H is only globally centred.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from dtsim_core import residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded

N_REPS_A = 1000                 # Scheme A' replications
N_REPS_C = 100                  # Scheme C' replications per gamma
N_RESTARTS = 1                  # restarts per simulated fit
N_REFINE = 2                    # within-cell refinement passes of the generator
GAMMA_CONDITIONS = [0.0, -0.4, -0.8]
MIN_PL_LIST = [1, 3, 4]         # plurality filters: all, >= 3/5, >= 4/5
Q_NOISE_STD = 0.05              # SD of the Gaussian noise on the linked accuracy

# Real-data within-cell gamma_hat of the hard-mark estimator per plurality filter
REAL_GAMMA = {"full": -0.3915, "p3_5": -0.3481, "p4_5": -0.1274}


def fit_raw_H(dlgs_evs: List[List[dict]], seed: int) -> dict:
    """Hard-mark fit with globally centred H (no within-cell residualisation)."""
    all_H = [e["H"] for dlg in dlgs_evs for e in dlg]
    H_bar = float(np.mean(all_H)) if all_H else 0.0
    dlgs_dt = [
        {"cats": np.array([e["cat"] for e in evs], dtype=np.int64),
         "Hs_c": np.array([e["H"] for e in evs], dtype=np.float64) - H_bar}
        for evs in dlgs_evs
    ]
    return fit_dt_bounded(dlgs_dt, H_bar, n_restarts=N_RESTARTS, seed=seed)


def fit_within_cell(dlgs_evs: List[List[dict]], seed: int) -> dict:
    """Hard-mark fit with within-cell residualised H."""
    dlgs_dt_raw = [
        {"cats": np.array([e["cat"] for e in evs], dtype=np.int64),
         "Hs_raw": np.array([e["H"] for e in evs], dtype=np.float64)}
        for evs in dlgs_evs
    ]
    dlgs_r = residualize_within_cell(dlgs_dt_raw)
    H_bar = global_H_mean(dlgs_r)
    dlgs_dt = prepare_dt_dlgs(dlgs_r, H_bar)
    return fit_dt_bounded(dlgs_dt, H_bar, n_restarts=N_RESTARTS, seed=seed)


# ============================================================================
# Aggregation
# ============================================================================

def _conv_vals(recs, model, min_pl, fit_type="wc"):
    """Converged gamma_hat values for (model, min_pl, fit_type) from Scheme A' records."""
    out = []
    for r in recs:
        d = r["results"].get(model, {}).get(min_pl, {}).get(fit_type, {})
        gh = d.get("gamma_hat", float("nan"))
        if d.get("converged") and not d.get("at_bound") and not math.isnan(gh):
            out.append(gh)
    return out


def _conv_vals_c(recs, model, min_pl, gamma_inj):
    """Converged gamma_hat values from Scheme C' records."""
    out = []
    for r in recs:
        if abs(r.get("gamma_inj", 999) - gamma_inj) > 1e-6:
            continue
        d = r["results"].get(model, {}).get(min_pl, {})
        gh = d.get("gamma_hat", float("nan"))
        if d.get("converged") and not d.get("at_bound") and not math.isnan(gh):
            out.append(gh)
    return out


def _summarise(vals, true_gamma=None):
    if len(vals) < 2:
        return {}
    arr = np.array(vals)
    bias = float(arr.mean() - true_gamma) if true_gamma is not None else float("nan")
    att = float(arr.mean() / true_gamma) if (true_gamma and abs(true_gamma) > 1e-6) else float("nan")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "bias": bias,
        "attenuation": att,
        "n": len(vals),
    }


def _emp_p(vals, threshold):
    """One-sided empirical P(null <= threshold) with a Clopper-Pearson interval."""
    from scipy.stats import beta as beta_d
    n = len(vals)
    k = sum(1 for v in vals if v <= threshold)
    p = k / n if n > 0 else float("nan")
    lo = float(beta_d.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_d.ppf(0.975, k + 1, n - k)) if k < n else 1.0
    return {"n": n, "k": k, "p_hat": p, "ci95_lo": lo, "ci95_hi": hi}
