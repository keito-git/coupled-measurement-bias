"""
E32: simulation-calibrated estimator with a Neyman confidence belt (hard-mark, within-cell, full data).

Grid of true gamma: {+0.4, 0.0, -0.4, -0.8, -1.2}; Scheme C' generator with linked votes (as in E27e).
  Phase 1 (calibration map):  200 reps per gamma, rep ids 0..199
  Phase 2 (gate G2, held out): 100 reps per gamma, rep ids 200..299
  Phase 3 (oracle check):      50 reps per gamma, rep ids 300..349
Gate G2: |bias of the calibrated estimate| < 0.1 and Neyman coverage in [0.90, 0.99] at true gamma in
{+0.4, 0, -0.4, -0.8}.

Outputs (RESULTS_ROOT/E32): E32_cal_ckpt.json, E32_g2_ckpt.json, E32_orc_ckpt.json, E32_results.json
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from dtsim_core import (  # noqa: E402
    K, REAL_GAMMA_WITHINCELL_FULL, load_el_raw, empirical_q_pool, compute_empirical_confusion,
    empirical_H_pool, residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded, unpack_v_dt,
)
from dtsim_fits import N_REFINE, Q_NOISE_STD, fit_within_cell  # noqa: E402
from dtsim_kernels import _seed_nb, _process_one_dlg_scheme_c_nb, _warmup_numba  # noqa: E402

# -- Constants -----------------------------------------------------------------
OUT_DIR      = config.results_dir("E32")
GAMMA_GRID   = [0.4, 0.0, -0.4, -0.8, -1.2]   # calibration grid
N_CAL_REPS   = 200   # Phase 1: calibration
N_G2_REPS    = 100   # Phase 2: G2 held-out (seeds disjoint from Phase 1)
N_ORC_REPS   = 50    # Phase 3: oracle check
N_WORKERS_E32 = config.n_workers(3)
CKPT_INTERVAL = 25   # checkpoint every N completed reps

# gate G2: |bias| < 0.1 and coverage in [0.90, 0.99] at true gamma in {+0.4, 0, -0.4, -0.8}
G2_GAMMA_SET   = {0.4, 0.0, -0.4, -0.8}
G2_BIAS_THR    = 0.1
G2_COV_LO      = 0.90
G2_COV_HI      = 0.99

# real-data gamma_hat (hard-mark, within-cell, EL-wide)
REAL_GAMMA_OBS = REAL_GAMMA_WITHINCELL_FULL


# ============================================================================
# Worker function (module-level for pickling)
# ============================================================================

def _e32_worker(args: tuple) -> dict:
    """
    Lightweight Scheme C' worker for E32.

    Runs the full Scheme C' DGP (linked-q, within-cell H, hard-mark),
    but fits ONLY obs/pl=1 and oracle/pl=1 (2 fits instead of 6 in E27e).

    args
    ----
    rep              : int     (seed index)
    mu_arr           : (K,)    float64
    alpha_flat       : (K*K,)  float64   (will be reshaped to (K,K))
    beta_val         : float
    gamma_inj        : float
    H_dist_arr       : (N_h,)  float64   empirical H pool
    H_gm_val         : float   global mean of H_dist
    q_pool_arr       : (N_q,)  float64   empirical q pool
    confusion_arr    : (K,K)   float64
    dlg_lengths      : (n_dlg,) int64    dialogue lengths
    H_sorted_arr     : (N_h,)  ascending sorted H_dist
    q_sorted_arr     : (N_q,)  descending sorted q_pool
    Q_NOISE_STD_val  : float
    phase            : str     "cal" | "g2" | "oracle"
    """
    (rep, mu_arr, alpha_flat, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr,
     dlg_lengths, H_sorted_arr, q_sorted_arr,
     Q_NOISE_STD_val, phase) = args

    K_val = 7
    N_ANN = 5
    alpha_mat = alpha_flat.reshape(K_val, K_val)

    # Seed: same formula as E27e _scheme_c_prime_worker_e
    g_seed = int(round((gamma_inj + 2.0) * 10000))
    _seed_nb(rep * 91723 + g_seed * 1000 + 77)

    oracle_dlgs: List[List[dict]] = []
    obs_dlgs:    List[List[dict]] = []

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

    # Fit obs/pl=1 (within-cell H, hard-mark DT - the target estimator)
    obs_dlgs_pl1 = [d for d in obs_dlgs if len(d) >= 2]
    obs_fit = {"gamma_hat": float("nan"), "converged": False,
               "at_bound": False, "n_dlg": 0}
    if len(obs_dlgs_pl1) >= 5:
        wc = fit_within_cell(obs_dlgs_pl1, seed=rep)
        obs_fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(obs_dlgs_pl1),
        }

    # Fit oracle/pl=1 (true categories + latent H)
    orc_fit = {"gamma_hat": float("nan"), "converged": False,
               "at_bound": False, "n_dlg": 0}
    if phase in ("cal", "oracle"):
        orc_dlgs_pl1 = [d for d in oracle_dlgs if len(d) >= 2]
        if len(orc_dlgs_pl1) >= 5:
            wc_o = fit_within_cell(orc_dlgs_pl1, seed=rep + 50000)
            orc_fit = {
                "gamma_hat": float(wc_o["gamma_hat"]),
                "converged":  bool(wc_o["converged"]),
                "at_bound":   bool(wc_o["at_bound"]),
                "n_dlg":      len(orc_dlgs_pl1),
            }

    return {
        "rep":        rep,
        "gamma_inj":  float(gamma_inj),
        "phase":      phase,
        "obs":        obs_fit,
        "oracle":     orc_fit,
    }


# ============================================================================
# Runner with incremental checkpoints
# ============================================================================

def _run_with_progress_e32(
    args_list: list,
    checkpoint_path: Path,
    key_name: str,
    n_workers: int = N_WORKERS_E32,
) -> list:
    """
    Run _e32_worker over args_list using imap_unordered.
    Checkpoints every CKPT_INTERVAL reps. Resume-safe.

    The rep id in each result is args[0]; used as the done-key.
    To handle multiple gamma values in one pool, the done-key is (rep, gamma_inj).
    """
    # Load existing checkpoint
    done_keys: set = set()
    results: list = []
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_keys = {(r["rep"], r["gamma_inj"]) for r in results}
            print(f"  Checkpoint: {len(done_keys)} reps already done "
                  f"({checkpoint_path.name})", flush=True)
        except Exception as ex:
            print(f"  Warning: checkpoint load failed ({ex}); starting fresh",
                  flush=True)

    pending = [a for a in args_list
               if (a[0], float(a[4])) not in done_keys]   # (rep, gamma_inj)
    n_total = len(args_list)

    if not pending:
        print(f"  All {n_total} tasks found in checkpoint.", flush=True)
        return results

    print(f"  {len(pending)} tasks pending on {n_workers} workers ...",
          flush=True)

    ctx = mp.get_context("spawn")
    unsaved = 0

    def _save():
        checkpoint_path.write_text(
            json.dumps({key_name: results}, ensure_ascii=False,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8",
        )

    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_e32_worker, pending):
            results.append(result)
            unsaved += 1
            n_done = len(results)

            if unsaved >= CKPT_INTERVAL:
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] {n_done}/{n_total} tasks done", flush=True)
                _save()
                unsaved = 0

    _save()
    print(f"  Final checkpoint saved -> {checkpoint_path}", flush=True)
    return results


# ============================================================================
# Calibration analysis
# ============================================================================

def _extract_gamma_hats(
    results: list,
    gamma_val: float,
    path: str,   # "obs" or "oracle"
) -> np.ndarray:
    """Extract gamma_hat values for one true-gamma condition, dropping NaN."""
    vals = []
    for r in results:
        if abs(r["gamma_inj"] - gamma_val) < 1e-9:
            g = r[path]["gamma_hat"]
            if not math.isnan(g):
                vals.append(g)
    return np.array(vals, dtype=float)


def build_calibration_map(
    cal_results: list,
    gamma_grid: List[float],
) -> dict:
    """
    For each true gamma in gamma_grid, compute mean and 2.5%/97.5% quantiles
    of gamma_hat_obs. Return summary dict and interpolation arrays.

    Also checks if mean/lower/upper curves are monotone.
    """
    grid = sorted(gamma_grid)   # ascending
    means, lo, hi, ns = [], [], [], []
    all_hats = {}

    for g in grid:
        hats = _extract_gamma_hats(cal_results, g, "obs")
        all_hats[g] = hats
        ns.append(len(hats))
        means.append(float(np.mean(hats)) if len(hats) > 0 else float("nan"))
        lo.append(float(np.percentile(hats, 2.5)) if len(hats) > 0 else float("nan"))
        hi.append(float(np.percentile(hats, 97.5)) if len(hats) > 0 else float("nan"))

    g_arr  = np.array(grid)
    m_arr  = np.array(means)
    lo_arr = np.array(lo)
    hi_arr = np.array(hi)

    # Monotonicity check: each curve should be non-decreasing in gamma
    def _is_monotone(arr: np.ndarray) -> bool:
        diffs = np.diff(arr)
        return bool(np.all(diffs >= 0))

    mono_mean = _is_monotone(m_arr)
    mono_lo   = _is_monotone(lo_arr)
    mono_hi   = _is_monotone(hi_arr)

    grid_stats = {}
    for i, g in enumerate(grid):
        grid_stats[g] = {
            "n":     ns[i],
            "mean":  means[i],
            "q2_5":  lo[i],
            "q97_5": hi[i],
        }

    return {
        "grid":       grid,
        "g_arr":      g_arr,
        "m_arr":      m_arr,
        "lo_arr":     lo_arr,
        "hi_arr":     hi_arr,
        "grid_stats": grid_stats,
        "monotone_mean": mono_mean,
        "monotone_lo":   mono_lo,
        "monotone_hi":   mono_hi,
        "all_hats":   all_hats,
    }


def calibrate_point_estimate(cal_map: dict, gamma_hat_obs: float) -> Optional[float]:
    """
    Calibrated point estimate: invert the mean curve.
    Returns the true gamma whose expected gamma_hat_obs equals gamma_hat_obs.
    Uses linear interpolation.
    If gamma_hat_obs is outside the range of the mean curve, returns None.
    """
    g_arr = cal_map["g_arr"]
    m_arr = cal_map["m_arr"]

    # Check bounds
    if gamma_hat_obs < m_arr.min() or gamma_hat_obs > m_arr.max():
        return None

    # np.interp requires x-axis to be increasing; mean curve should be mono increasing
    # Since g_arr is ascending and m_arr should be ascending,
    # invert: treat m_arr as x-axis and g_arr as y-axis
    gamma_cal = float(np.interp(gamma_hat_obs, m_arr, g_arr))
    return gamma_cal


def neyman_interval(
    cal_map: dict,
    gamma_hat_obs: float,
) -> dict:
    """
    Neyman 95% confidence interval by belt inversion.

    Principle:
      The Neyman belt has, at each true gamma_t, the interval
      [Q2.5(gamma_hat|gamma_t), Q97.5(gamma_hat|gamma_t)].
      The 95% CI for true gamma given observed gamma_hat is all gamma_t
      for which gamma_hat_obs lies in that belt.

    Since both quantile curves are monotone increasing in true gamma:
      - Lower bound: smallest gamma_t where Q97.5(gamma_hat|gamma_t) >= gamma_hat_obs
        (the observed value first enters the belt from below as true gamma rises)
      - Upper bound: largest gamma_t where Q2.5(gamma_hat|gamma_t) <= gamma_hat_obs
        (the observed value exits the belt from above as true gamma rises)

    Returns:
      lo_bound, hi_bound : lower/upper of Neyman interval (float)
      open_lo, open_hi   : True if the interval extends to the grid edge
      gamma_included     : list of grid points satisfying the condition
      contains_zero      : whether 0 is inside the interval
    """
    g_arr  = cal_map["g_arr"]   # sorted ascending (gamma_true values)
    lo_arr = cal_map["lo_arr"]  # Q2.5(gamma_hat_obs | gamma_true), monotone inc
    hi_arr = cal_map["hi_arr"]  # Q97.5(gamma_hat_obs | gamma_true), monotone inc
    obs    = gamma_hat_obs

    n = len(g_arr)

    # At each grid point: is obs in [Q2.5, Q97.5]?
    included = [bool(lo_arr[i] <= obs <= hi_arr[i]) for i in range(n)]
    included_g = [float(g_arr[i]) for i in range(n) if included[i]]

    if not included_g:
        return {
            "lo_bound": None, "hi_bound": None,
            "open_lo": True, "open_hi": True,
            "gamma_included": [],
            "contains_zero": False,
            "note": "gamma_hat_obs outside belt at all grid points",
        }

    # ---- Lower bound: smallest gamma_t where Q97.5(gamma_t) >= obs ----
    # Search from i=0 (smallest gamma) upward.
    # Assumption: hi_arr is monotone non-decreasing.
    lo_bound: float
    open_lo: bool

    if hi_arr[0] >= obs:
        # Already satisfied at the lowest grid point -> open low end
        lo_bound = float(g_arr[0])
        open_lo  = True
    else:
        # Find first i where hi_arr[i] >= obs (crossing from below)
        lo_bound = float(g_arr[-1])  # fallback (should not happen if included_g is non-empty)
        open_lo  = False
        for i in range(1, n):
            if hi_arr[i] >= obs:
                # Interpolate between (i-1) and i
                dh = hi_arr[i] - hi_arr[i - 1]
                if abs(dh) < 1e-12:
                    lo_bound = float(g_arr[i - 1])
                else:
                    lo_bound = float(g_arr[i - 1]
                                     + (obs - hi_arr[i - 1]) / dh
                                     * (g_arr[i] - g_arr[i - 1]))
                break

    # ---- Upper bound: largest gamma_t where Q2.5(gamma_t) <= obs ----
    # Search from i=n-1 (largest gamma) downward.
    # Assumption: lo_arr is monotone non-decreasing.
    hi_bound: float
    open_hi: bool

    if lo_arr[-1] <= obs:
        # Still satisfied at the highest grid point -> open high end
        hi_bound = float(g_arr[-1])
        open_hi  = True
    else:
        # Find last i where lo_arr[i] <= obs (crossing from below going right)
        hi_bound = float(g_arr[0])  # fallback
        open_hi  = False
        for i in range(n - 2, -1, -1):
            if lo_arr[i] <= obs:
                # Interpolate between i and (i+1)
                dh = lo_arr[i + 1] - lo_arr[i]
                if abs(dh) < 1e-12:
                    hi_bound = float(g_arr[i + 1])
                else:
                    hi_bound = float(g_arr[i]
                                     + (obs - lo_arr[i]) / dh
                                     * (g_arr[i + 1] - g_arr[i]))
                break

    contains_zero = (lo_bound <= 0.0 <= hi_bound)

    note_parts = []
    if open_lo:
        note_parts.append(
            f"interval is open-ended at the lower grid edge ({float(g_arr[0]):.1f})")
    if open_hi:
        note_parts.append(
            f"interval is open-ended at the upper grid edge ({float(g_arr[-1]):.1f})")
    note = "; ".join(note_parts) if note_parts else "interval contained within grid"

    return {
        "lo_bound":       lo_bound,
        "hi_bound":       hi_bound,
        "open_lo":        open_lo,
        "open_hi":        open_hi,
        "gamma_included": included_g,
        "contains_zero":  contains_zero,
        "note":           note,
    }


# ============================================================================
# G2 gate evaluation
# ============================================================================

def compute_g2_stats(
    g2_results: list,
    cal_map: dict,
    gamma_grid: List[float],
) -> dict:
    """
    For each true gamma in G2_GAMMA_SET, compute:
      - mean gamma_cal - gamma (bias of calibrated estimate)
      - coverage of Neyman interval
      - median interval width

    Uses the calibration from Phase 1 (cal_map).
    """
    stats = {}
    for g in sorted(gamma_grid):
        if g not in G2_GAMMA_SET:
            continue

        hats = _extract_gamma_hats(g2_results, g, "obs")
        n = len(hats)
        if n == 0:
            stats[g] = {"n": 0, "bias": float("nan"), "coverage": float("nan"),
                        "median_width": float("nan"), "g2_pass": False}
            continue

        gamma_cals = []
        covered = []
        widths = []

        for gamma_hat in hats:
            # Calibrated point estimate
            g_cal = calibrate_point_estimate(cal_map, gamma_hat)
            if g_cal is None:
                gamma_cals.append(float("nan"))
            else:
                gamma_cals.append(g_cal)

            # Neyman interval
            nint = neyman_interval(cal_map, gamma_hat)
            lo_b = nint["lo_bound"]
            hi_b = nint["hi_bound"]

            # Coverage: true gamma in [lo_bound, hi_bound]?
            if lo_b is None or hi_b is None:
                covered.append(False)
                widths.append(float("nan"))
            else:
                covered.append(lo_b <= g <= hi_b)
                widths.append(hi_b - lo_b)

        gamma_cals_arr = np.array([x for x in gamma_cals if not math.isnan(x)])
        bias = float(np.mean(gamma_cals_arr) - g) if len(gamma_cals_arr) > 0 else float("nan")
        coverage = float(np.mean(covered))
        valid_widths = [w for w in widths if not math.isnan(w)]
        median_width = float(np.median(valid_widths)) if valid_widths else float("nan")

        g2_pass = (
            not math.isnan(bias) and abs(bias) < G2_BIAS_THR
            and not math.isnan(coverage) and G2_COV_LO <= coverage <= G2_COV_HI
        )

        stats[g] = {
            "n":            n,
            "bias":         bias,
            "coverage":     coverage,
            "median_width": median_width,
            "g2_pass":      g2_pass,
        }

    all_pass = all(v["g2_pass"] for v in stats.values())
    return {"per_gamma": stats, "all_pass": all_pass}


# ============================================================================
# Oracle check
# ============================================================================

def compute_oracle_check(
    orc_results: list,
    gamma_grid: List[float],
) -> dict:
    """
    Oracle check: for the oracle pipeline (true categories + latent H),
    the estimator should approximately recover true gamma.
    """
    stats = {}
    for g in sorted(gamma_grid):
        hats = _extract_gamma_hats(orc_results, g, "oracle")
        if len(hats) == 0:
            stats[g] = {"n": 0, "mean": float("nan"), "std": float("nan"),
                        "bias": float("nan")}
            continue
        stats[g] = {
            "n":    len(hats),
            "mean": float(np.mean(hats)),
            "std":  float(np.std(hats, ddof=1)) if len(hats) > 1 else float("nan"),
            "bias": float(np.mean(hats) - g),
        }
    return stats


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 68, flush=True)
    print("E32_calibration: Simulation-calibrated estimator + Neyman belt")
    print("=" * 68, flush=True)

    # -- Data loading ----------------------------------------------------------
    print("\n[1] Loading real EL data ...", flush=True)
    raw_dlgs    = load_el_raw()
    q_pool      = empirical_q_pool(raw_dlgs)
    confusion   = compute_empirical_confusion(raw_dlgs)
    H_dist      = empirical_H_pool(raw_dlgs)
    dlg_lengths = np.array([len(d) for d in raw_dlgs], dtype=np.int64)
    n_dlg       = len(raw_dlgs)
    n_ev        = int(dlg_lengths.sum())
    print(f"    {n_dlg} dialogues, {n_ev} events", flush=True)

    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]
    H_gm_val = float(H_dist.mean())

    # -- DT parameters from real within-cell fit -------------------------------
    print("\n[2] Real within-cell DT parameters (for DGP) ...", flush=True)
    raw_dt = [
        {"cats":   np.array([e["cat"] for e in d], dtype=np.int64),
         "Hs_raw": np.array([e["H"]   for e in d], dtype=float)}
        for d in raw_dlgs
    ]
    dlgs_r    = residualize_within_cell(raw_dt)
    H_bar_r   = global_H_mean(dlgs_r)
    dlgs_dt_r = prepare_dt_dlgs(dlgs_r, H_bar_r)
    real_fit  = fit_dt_bounded(dlgs_dt_r, H_bar_r, n_restarts=3, seed=0)
    print(f"    gamma = {real_fit['gamma_hat']:+.4f}  "
          f"(ref: {REAL_GAMMA_WITHINCELL_FULL:+.4f})", flush=True)

    if real_fit["v_hat"]:
        v = np.array(real_fit["v_hat"])
        mu_r, alpha_r, beta_r, _ = unpack_v_dt(v, K)
    else:
        mu_r    = np.ones(K) / K * 0.2
        alpha_r = np.ones((K, K)) * 0.05
        beta_r  = 1.0

    alpha_flat = alpha_r.ravel()

    # -- Numba JIT warmup ------------------------------------------------------
    print("\n[3] Numba JIT warmup ...", flush=True)
    t_jit0 = time.time()
    _warmup_numba(K)
    print(f"    JIT warmup: {time.time() - t_jit0:.2f}s", flush=True)

    # -- Phase 1: Calibration (200 reps per gamma) -----------------------------
    print(f"\n[4] Phase 1 - Calibration: "
          f"{N_CAL_REPS} reps x {len(GAMMA_GRID)} gamma values = "
          f"{N_CAL_REPS * len(GAMMA_GRID)} tasks ...", flush=True)
    print(f"    Workers: {N_WORKERS_E32}, checkpoint every {CKPT_INTERVAL} tasks",
          flush=True)

    cal_args = [
        (rep, mu_r, alpha_flat, float(beta_r), float(g),
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "cal")
        for g in GAMMA_GRID
        for rep in range(N_CAL_REPS)
    ]

    ckpt_cal = OUT_DIR / "E32_cal_ckpt.json"
    t4 = time.time()
    cal_results = _run_with_progress_e32(cal_args, ckpt_cal, "cal")
    print(f"    Phase 1 done in {time.time()-t4:.1f}s", flush=True)

    # Build calibration map
    cal_map = build_calibration_map(cal_results, GAMMA_GRID)
    print("\n  Calibration map (mean E[gamma_hat_obs]):", flush=True)
    for g in sorted(cal_map["grid_stats"].keys()):
        s = cal_map["grid_stats"][g]
        print(f"    gamma={g:+.1f}: n={s['n']:3d}  "
              f"mean={s['mean']:+.4f}  "
              f"Q2.5={s['q2_5']:+.4f}  "
              f"Q97.5={s['q97_5']:+.4f}", flush=True)
    print(f"  Monotonicity - mean: {cal_map['monotone_mean']}, "
          f"Q2.5: {cal_map['monotone_lo']}, "
          f"Q97.5: {cal_map['monotone_hi']}", flush=True)
    if not cal_map["monotone_mean"]:
        print("  WARNING: Mean calibration curve is NOT monotone.", flush=True)
    if not cal_map["monotone_lo"]:
        print("  WARNING: Q2.5 calibration curve is NOT monotone.", flush=True)
    if not cal_map["monotone_hi"]:
        print("  WARNING: Q97.5 calibration curve is NOT monotone.", flush=True)

    # -- Phase 2: G2 held-out (100 reps per gamma) -----------------------------
    print(f"\n[5] Phase 2 - G2 held-out: "
          f"{N_G2_REPS} reps x {len(GAMMA_GRID)} gamma values = "
          f"{N_G2_REPS * len(GAMMA_GRID)} tasks ...", flush=True)

    g2_args = [
        (rep, mu_r, alpha_flat, float(beta_r), float(g),
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "g2")
        for g in GAMMA_GRID
        for rep in range(N_CAL_REPS, N_CAL_REPS + N_G2_REPS)   # reps 200..299
    ]

    ckpt_g2 = OUT_DIR / "E32_g2_ckpt.json"
    t5 = time.time()
    g2_results = _run_with_progress_e32(g2_args, ckpt_g2, "g2")
    print(f"    Phase 2 done in {time.time()-t5:.1f}s", flush=True)

    # Compute G2 gate
    g2_stats = compute_g2_stats(g2_results, cal_map, GAMMA_GRID)
    print(f"\n  G2 gate results:", flush=True)
    for g in sorted(g2_stats["per_gamma"].keys()):
        s = g2_stats["per_gamma"][g]
        print(f"    gamma={g:+.1f}: bias={s['bias']:+.4f}  "
              f"coverage={s['coverage']:.3f}  "
              f"med_width={s['median_width']:.3f}  "
              f"{'PASS' if s['g2_pass'] else 'FAIL'}", flush=True)
    verdict = "PASS" if g2_stats["all_pass"] else "FAIL"
    print(f"\n  G2 Overall Verdict: {verdict}", flush=True)

    # -- Phase 3: Oracle check (50 reps per gamma) -----------------------------
    print(f"\n[6] Phase 3 - Oracle check: "
          f"{N_ORC_REPS} reps x {len(GAMMA_GRID)} gamma values = "
          f"{N_ORC_REPS * len(GAMMA_GRID)} tasks ...", flush=True)

    orc_args = [
        (rep, mu_r, alpha_flat, float(beta_r), float(g),
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "oracle")
        for g in GAMMA_GRID
        for rep in range(N_CAL_REPS + N_G2_REPS,
                         N_CAL_REPS + N_G2_REPS + N_ORC_REPS)   # reps 300..349
    ]

    ckpt_orc = OUT_DIR / "E32_orc_ckpt.json"
    t6 = time.time()
    orc_results = _run_with_progress_e32(orc_args, ckpt_orc, "oracle")
    print(f"    Phase 3 done in {time.time()-t6:.1f}s", flush=True)

    oracle_stats = compute_oracle_check(orc_results, GAMMA_GRID)
    print("\n  Oracle check (should approximately recover true gamma):", flush=True)
    for g in sorted(oracle_stats.keys()):
        s = oracle_stats[g]
        print(f"    gamma={g:+.1f}: mean={s['mean']:+.4f}  "
              f"std={s['std']:.4f}  bias={s['bias']:+.4f}", flush=True)

    # -- Apply to real data ----------------------------------------------------
    print(f"\n[7] Applying to real data (gamma_hat_obs = {REAL_GAMMA_OBS:.4f}) ...",
          flush=True)
    gamma_cal = calibrate_point_estimate(cal_map, REAL_GAMMA_OBS)
    nint = neyman_interval(cal_map, REAL_GAMMA_OBS)

    if gamma_cal is None:
        print(f"    gamma_cal: not computable (obs outside mean-curve range)",
              flush=True)
    else:
        print(f"    gamma_cal = {gamma_cal:.4f}", flush=True)
    print(f"    Neyman 95% interval: [{nint['lo_bound']}, {nint['hi_bound']}]",
          flush=True)
    print(f"    Open ends: lo={nint['open_lo']}, hi={nint['open_hi']}", flush=True)
    print(f"    Contains zero: {nint['contains_zero']}", flush=True)
    print(f"    Note: {nint['note']}", flush=True)

    # -- Serialize calibration map (remove numpy arrays, keep lists) -----------
    cal_map_json = {
        "grid":          [float(g) for g in cal_map["grid"]],
        "g_arr":         [float(x) for x in cal_map["g_arr"]],
        "m_arr":         [float(x) for x in cal_map["m_arr"]],
        "lo_arr":        [float(x) for x in cal_map["lo_arr"]],
        "hi_arr":        [float(x) for x in cal_map["hi_arr"]],
        "monotone_mean": cal_map["monotone_mean"],
        "monotone_lo":   cal_map["monotone_lo"],
        "monotone_hi":   cal_map["monotone_hi"],
        "grid_stats":    {str(k): v for k, v in cal_map["grid_stats"].items()},
    }

    elapsed = time.time() - t0
    out = {
        "experiment":    "E32_calibration",
        "config": {
            "gamma_grid":     GAMMA_GRID,
            "n_cal_reps":     N_CAL_REPS,
            "n_g2_reps":      N_G2_REPS,
            "n_orc_reps":     N_ORC_REPS,
            "n_workers":      N_WORKERS_E32,
            "ckpt_interval":  CKPT_INTERVAL,
            "g2_gamma_set":   sorted(G2_GAMMA_SET),
            "g2_bias_thr":    G2_BIAS_THR,
            "g2_cov_lo":      G2_COV_LO,
            "g2_cov_hi":      G2_COV_HI,
            "real_gamma_obs": REAL_GAMMA_OBS,
            "n_refine":       N_REFINE,
            "q_noise_std":    Q_NOISE_STD,
        },
        "calibration_map": cal_map_json,
        "g2": {
            "per_gamma": {str(k): v for k, v in g2_stats["per_gamma"].items()},
            "all_pass":  g2_stats["all_pass"],
        },
        "oracle_check": {str(k): v for k, v in oracle_stats.items()},
        "real_data_application": {
            "gamma_hat_obs": float(REAL_GAMMA_OBS),
            "gamma_cal":     float(gamma_cal) if gamma_cal is not None else None,
        },
        "real_data_neyman_interval": {
            "lo_bound":       nint["lo_bound"],
            "hi_bound":       nint["hi_bound"],
            "open_lo":        nint["open_lo"],
            "open_hi":        nint["open_hi"],
            "contains_zero":  nint["contains_zero"],
            "gamma_included": nint["gamma_included"],
            "note":           nint["note"],
        },
        "elapsed_sec": float(elapsed),
    }

    results_path = OUT_DIR / "E32_results.json"
    results_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print(f"\nResults -> {results_path}", flush=True)

    print(f"\n=== E32 complete ({elapsed:.1f}s = {elapsed/3600:.2f}h) ===",
          flush=True)


if __name__ == "__main__":
    main()
