"""
E32 follow-up (not pre-registered): calibration grid extended to true gamma = +0.8 and +1.2.

At gamma = +0.4 many held-out estimates in E32 exceed the top of the 5-point mean curve and cannot be
inverted. This run adds two calibration points and re-tests G2 at gamma = +0.4 with fresh replications;
the pre-registered E32 verdict is unchanged.
  Phase A: 200 calibration reps at gamma = +0.8 (rep ids 0..199)
  Phase B: 200 calibration reps at gamma = +1.2 (rep ids 0..199)
  Phase C: 100 new G2 reps at gamma = +0.4 (rep ids 400..499, disjoint from all E32 reps)
The seed formula is the one of E32, so the new grid points do not collide with E32 seeds.
Requires E32_cal_ckpt.json from run_E32.py.

Outputs (RESULTS_ROOT/E32): E32_fup_calA_ckpt.json, E32_fup_calB_ckpt.json, E32_fup_g2c_ckpt.json,
E32_followup_extended_grid.json
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
OUT_DIR           = config.results_dir("E32")

# E32 base grid (5 points)
GAMMA_GRID_BASE   = [0.4, 0.0, -0.4, -0.8, -1.2]

# New extension gammas
GAMMA_GRID_EXT    = [0.8, 1.2]   # two new cal points

# Full 7-point extended grid
GAMMA_GRID_FULL   = sorted(set(GAMMA_GRID_BASE + GAMMA_GRID_EXT))  # ascending

N_CAL_REPS        = 200   # Phase A/B: new cal reps
N_G2C_REPS        = 100   # Phase C: new G2 reps for gamma=+0.4
G2C_REP_OFFSET    = 400   # rep ids 400..499

N_WORKERS         = config.n_workers(4)
CKPT_INTERVAL     = 25

# gate G2 criteria (as in E32)
G2_BIAS_THR       = 0.1
G2_COV_LO         = 0.90
G2_COV_HI         = 0.99

REAL_GAMMA_OBS    = REAL_GAMMA_WITHINCELL_FULL


# ============================================================================
# Worker (same as run_E32._e32_worker)
# ============================================================================

def _fup_worker(args: tuple) -> dict:
    """Scheme C' worker: fits obs/pl=1 and (for cal/oracle) oracle/pl=1."""
    (rep, mu_arr, alpha_flat, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr,
     dlg_lengths, H_sorted_arr, q_sorted_arr,
     Q_NOISE_STD_val, phase) = args

    K_val  = 7
    N_ANN  = 5
    alpha_mat = alpha_flat.reshape(K_val, K_val)

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

    obs_fit = {"gamma_hat": float("nan"), "converged": False,
               "at_bound": False, "n_dlg": 0}
    obs_pl1 = [d for d in obs_dlgs if len(d) >= 2]
    if len(obs_pl1) >= 5:
        wc = fit_within_cell(obs_pl1, seed=rep)
        obs_fit = {
            "gamma_hat": float(wc["gamma_hat"]),
            "converged":  bool(wc["converged"]),
            "at_bound":   bool(wc["at_bound"]),
            "n_dlg":      len(obs_pl1),
        }

    orc_fit = {"gamma_hat": float("nan"), "converged": False,
               "at_bound": False, "n_dlg": 0}
    if phase in ("cal", "oracle"):
        orc_pl1 = [d for d in oracle_dlgs if len(d) >= 2]
        if len(orc_pl1) >= 5:
            wc_o = fit_within_cell(orc_pl1, seed=rep + 50000)
            orc_fit = {
                "gamma_hat": float(wc_o["gamma_hat"]),
                "converged":  bool(wc_o["converged"]),
                "at_bound":   bool(wc_o["at_bound"]),
                "n_dlg":      len(orc_pl1),
            }

    return {
        "rep":       rep,
        "gamma_inj": float(gamma_inj),
        "phase":     phase,
        "obs":       obs_fit,
        "oracle":    orc_fit,
    }


# ============================================================================
# Pool runner with resumable checkpoints
# ============================================================================

def _run_phase(
    args_list: list,
    checkpoint_path: Path,
    key_name: str,
    n_workers: int = N_WORKERS,
) -> list:
    done_keys: set = set()
    results:   list = []

    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            results   = data.get(key_name, [])
            done_keys = {(r["rep"], r["gamma_inj"]) for r in results}
            print(f"  Checkpoint: {len(done_keys)} reps already done "
                  f"({checkpoint_path.name})", flush=True)
        except Exception as ex:
            print(f"  Warning: checkpoint load failed ({ex}); starting fresh",
                  flush=True)

    pending = [a for a in args_list
               if (a[0], float(a[4])) not in done_keys]
    n_total = len(args_list)

    if not pending:
        print(f"  All {n_total} tasks found in checkpoint.", flush=True)
        return results

    print(f"  {len(pending)}/{n_total} tasks pending on {n_workers} workers ...",
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
        for result in pool.imap_unordered(_fup_worker, pending):
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
# Calibration analysis (as in run_E32)
# ============================================================================

def _extract_hats(results: list, gamma_val: float, path: str) -> np.ndarray:
    vals = []
    for r in results:
        if abs(r["gamma_inj"] - gamma_val) < 1e-9:
            g = r[path]["gamma_hat"]
            if not math.isnan(g):
                vals.append(g)
    return np.array(vals, dtype=float)


def build_cal_map(cal_results: list, gamma_grid: List[float]) -> dict:
    grid = sorted(gamma_grid)
    means, lo, hi, ns = [], [], [], []
    for g in grid:
        hats = _extract_hats(cal_results, g, "obs")
        ns.append(len(hats))
        means.append(float(np.mean(hats)) if len(hats) > 0 else float("nan"))
        lo.append(float(np.percentile(hats, 2.5)) if len(hats) > 0 else float("nan"))
        hi.append(float(np.percentile(hats, 97.5)) if len(hats) > 0 else float("nan"))

    g_arr  = np.array(grid, dtype=float)
    m_arr  = np.array(means, dtype=float)
    lo_arr = np.array(lo, dtype=float)
    hi_arr = np.array(hi, dtype=float)

    def _mono(arr):
        return bool(np.all(np.diff(arr) >= 0))

    grid_stats = {
        g: {"n": ns[i], "mean": means[i], "q2_5": lo[i], "q97_5": hi[i]}
        for i, g in enumerate(grid)
    }
    return {
        "grid": grid, "g_arr": g_arr, "m_arr": m_arr,
        "lo_arr": lo_arr, "hi_arr": hi_arr,
        "grid_stats": grid_stats,
        "monotone_mean": _mono(m_arr),
        "monotone_lo":   _mono(lo_arr),
        "monotone_hi":   _mono(hi_arr),
    }


def calibrate_pt(cal_map: dict, obs: float) -> Optional[float]:
    m, g = cal_map["m_arr"], cal_map["g_arr"]
    if obs < m.min() or obs > m.max():
        return None
    return float(np.interp(obs, m, g))


def neyman_ci(cal_map: dict, obs: float) -> dict:
    g_arr  = cal_map["g_arr"]
    lo_arr = cal_map["lo_arr"]
    hi_arr = cal_map["hi_arr"]
    n = len(g_arr)

    included = [bool(lo_arr[i] <= obs <= hi_arr[i]) for i in range(n)]
    included_g = [float(g_arr[i]) for i in range(n) if included[i]]

    if not included_g:
        return {"lo_bound": None, "hi_bound": None,
                "open_lo": True, "open_hi": True,
                "contains_zero": False,
                "note": "obs outside belt at all grid points"}

    # Lower bound
    if hi_arr[0] >= obs:
        lo_bound, open_lo = float(g_arr[0]), True
    else:
        lo_bound, open_lo = float(g_arr[-1]), False
        for i in range(1, n):
            if hi_arr[i] >= obs:
                dh = hi_arr[i] - hi_arr[i-1]
                lo_bound = (float(g_arr[i-1]) if abs(dh) < 1e-12 else
                            float(g_arr[i-1] + (obs - hi_arr[i-1]) / dh
                                  * (g_arr[i] - g_arr[i-1])))
                break

    # Upper bound
    if lo_arr[-1] <= obs:
        hi_bound, open_hi = float(g_arr[-1]), True
    else:
        hi_bound, open_hi = float(g_arr[0]), False
        for i in range(n-2, -1, -1):
            if lo_arr[i] <= obs:
                dh = lo_arr[i+1] - lo_arr[i]
                hi_bound = (float(g_arr[i+1]) if abs(dh) < 1e-12 else
                            float(g_arr[i] + (obs - lo_arr[i]) / dh
                                  * (g_arr[i+1] - g_arr[i])))
                break

    parts = []
    if open_lo:
        parts.append(f"open-ended at lower grid edge ({float(g_arr[0]):.1f})")
    if open_hi:
        parts.append(f"open-ended at upper grid edge ({float(g_arr[-1]):.1f})")

    return {
        "lo_bound": lo_bound, "hi_bound": hi_bound,
        "open_lo": open_lo, "open_hi": open_hi,
        "contains_zero": bool(lo_bound <= 0.0 <= hi_bound),
        "gamma_included": included_g,
        "note": "; ".join(parts) if parts else "interval within grid",
    }


def compute_g2_at_gamma(
    g2_results: list,
    cal_map: dict,
    gamma_val: float,
) -> dict:
    """G2 stats for a single gamma value, using the provided calibration map."""
    hats = _extract_hats(g2_results, gamma_val, "obs")
    n = len(hats)
    if n == 0:
        return {"n": 0, "bias": float("nan"), "coverage": float("nan"),
                "median_width": float("nan"), "g2_pass": False,
                "n_valid_cal": 0}

    gamma_cals, covered, widths = [], [], []
    for gh in hats:
        gc = calibrate_pt(cal_map, gh)
        gamma_cals.append(float("nan") if gc is None else gc)

        ci = neyman_ci(cal_map, gh)
        lb, hb = ci["lo_bound"], ci["hi_bound"]
        if lb is None or hb is None:
            covered.append(False)
            widths.append(float("nan"))
        else:
            covered.append(bool(lb <= gamma_val <= hb))
            widths.append(hb - lb)

    valid_cals = np.array([x for x in gamma_cals if not math.isnan(x)])
    bias     = float(np.mean(valid_cals) - gamma_val) if len(valid_cals) > 0 else float("nan")
    coverage = float(np.mean(covered))
    vw       = [w for w in widths if not math.isnan(w)]
    med_w    = float(np.median(vw)) if vw else float("nan")

    g2_pass = (
        not math.isnan(bias) and abs(bias) < G2_BIAS_THR
        and not math.isnan(coverage) and G2_COV_LO <= coverage <= G2_COV_HI
    )
    return {
        "n":            n,
        "n_valid_cal":  int(len(valid_cals)),
        "bias":         bias,
        "coverage":     coverage,
        "median_width": med_w,
        "g2_pass":      g2_pass,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 68, flush=True)
    print("E32_followup: Extended calibration grid (+0.8, +1.2)", flush=True)
    print("=" * 68, flush=True)

    # -- Load real data ---------------------------------------------------------
    print("\n[1] Loading real EL data ...", flush=True)
    raw_dlgs    = load_el_raw()
    q_pool      = empirical_q_pool(raw_dlgs)
    confusion   = compute_empirical_confusion(raw_dlgs)
    H_dist      = empirical_H_pool(raw_dlgs)
    dlg_lengths = np.array([len(d) for d in raw_dlgs], dtype=np.int64)
    print(f"    {len(raw_dlgs)} dialogues, {int(dlg_lengths.sum())} events",
          flush=True)

    H_sorted = np.sort(H_dist)
    q_sorted = np.sort(q_pool)[::-1]
    H_gm_val = float(H_dist.mean())

    # -- DT params from real within-cell fit -----------------------------------
    print("\n[2] Real within-cell DT parameters ...", flush=True)
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
    t_jit = time.time()
    _warmup_numba(K)
    print(f"    JIT warmup: {time.time()-t_jit:.2f}s", flush=True)

    # -- Load existing E32 Phase 1 cal data (5 gammas) -------------------------
    print("\n[4] Loading E32 Phase 1 calibration data ...", flush=True)
    ckpt_e32 = OUT_DIR / "E32_cal_ckpt.json"
    base_cal = json.loads(ckpt_e32.read_text(encoding="utf-8"))["cal"]
    print(f"    Loaded {len(base_cal)} existing cal reps from {ckpt_e32.name}",
          flush=True)

    # Per-gamma summary of base data
    for g in sorted(GAMMA_GRID_BASE, reverse=True):
        hats = _extract_hats(base_cal, g, "obs")
        print(f"    gamma={g:+.1f}: n={len(hats)} mean={np.mean(hats):+.4f}",
              flush=True)

    # -- Phase A: 200 cal reps for gamma=+0.8 ---------------------------------
    print(f"\n[5] Phase A - Calibration gamma=+0.8: "
          f"{N_CAL_REPS} reps ...", flush=True)
    print(f"    Workers: {N_WORKERS}, checkpoint every {CKPT_INTERVAL}", flush=True)

    calA_args = [
        (rep, mu_r, alpha_flat, float(beta_r), 0.8,
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "cal")
        for rep in range(N_CAL_REPS)
    ]
    ckpt_calA = OUT_DIR / "E32_fup_calA_ckpt.json"
    tA = time.time()
    calA_results = _run_phase(calA_args, ckpt_calA, "cal")
    print(f"    Phase A done in {time.time()-tA:.1f}s", flush=True)
    hA = _extract_hats(calA_results, 0.8, "obs")
    print(f"    gamma=+0.8: n={len(hA)} mean={np.mean(hA):+.4f} "
          f"Q2.5={np.percentile(hA,2.5):+.4f} Q97.5={np.percentile(hA,97.5):+.4f}",
          flush=True)

    # -- Phase B: 200 cal reps for gamma=+1.2 ---------------------------------
    print(f"\n[6] Phase B - Calibration gamma=+1.2: "
          f"{N_CAL_REPS} reps ...", flush=True)

    calB_args = [
        (rep, mu_r, alpha_flat, float(beta_r), 1.2,
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "cal")
        for rep in range(N_CAL_REPS)
    ]
    ckpt_calB = OUT_DIR / "E32_fup_calB_ckpt.json"
    tB = time.time()
    calB_results = _run_phase(calB_args, ckpt_calB, "cal")
    print(f"    Phase B done in {time.time()-tB:.1f}s", flush=True)
    hB = _extract_hats(calB_results, 1.2, "obs")
    print(f"    gamma=+1.2: n={len(hB)} mean={np.mean(hB):+.4f} "
          f"Q2.5={np.percentile(hB,2.5):+.4f} Q97.5={np.percentile(hB,97.5):+.4f}",
          flush=True)

    # -- Build extended 7-gamma calibration map -----------------------------
    print("\n[7] Building extended 7-gamma calibration map ...", flush=True)
    all_cal = base_cal + calA_results + calB_results
    ext_cal_map = build_cal_map(all_cal, GAMMA_GRID_FULL)

    print("  Extended calibration map:", flush=True)
    for g in sorted(ext_cal_map["grid_stats"].keys()):
        s = ext_cal_map["grid_stats"][g]
        print(f"    gamma={g:+.1f}: n={s['n']:3d}  "
              f"mean={s['mean']:+.4f}  "
              f"Q2.5={s['q2_5']:+.4f}  "
              f"Q97.5={s['q97_5']:+.4f}", flush=True)
    print(f"  Monotonicity - mean: {ext_cal_map['monotone_mean']}, "
          f"Q2.5: {ext_cal_map['monotone_lo']}, "
          f"Q97.5: {ext_cal_map['monotone_hi']}", flush=True)

    # -- Phase C: 100 new G2 reps for gamma=+0.4 (reps 400..499) --------------
    print(f"\n[8] Phase C - New G2 reps gamma=+0.4: "
          f"{N_G2C_REPS} reps (reps {G2C_REP_OFFSET}..{G2C_REP_OFFSET+N_G2C_REPS-1}) ...",
          flush=True)

    g2c_args = [
        (rep, mu_r, alpha_flat, float(beta_r), 0.4,
         H_dist, H_gm_val, q_pool, confusion, dlg_lengths,
         H_sorted, q_sorted, Q_NOISE_STD, "g2")
        for rep in range(G2C_REP_OFFSET, G2C_REP_OFFSET + N_G2C_REPS)
    ]
    ckpt_g2c = OUT_DIR / "E32_fup_g2c_ckpt.json"
    tC = time.time()
    g2c_results = _run_phase(g2c_args, ckpt_g2c, "g2")
    print(f"    Phase C done in {time.time()-tC:.1f}s", flush=True)

    hC = _extract_hats(g2c_results, 0.4, "obs")
    print(f"    gamma=+0.4 (new G2): n={len(hC)} mean={np.mean(hC):+.4f} "
          f"std={np.std(hC,ddof=1):.4f}", flush=True)

    # -- G2 check for gamma=+0.4 with extended map -----------------------------
    print("\n[9] G2 check for gamma=+0.4 with extended calibration map ...",
          flush=True)
    g2c_stats = compute_g2_at_gamma(g2c_results, ext_cal_map, 0.4)
    print(f"    gamma=+0.4: bias={g2c_stats['bias']:+.4f}  "
          f"coverage={g2c_stats['coverage']:.3f}  "
          f"med_width={g2c_stats['median_width']:.3f}  "
          f"n_valid_cal={g2c_stats['n_valid_cal']}/{g2c_stats['n']}  "
          f"{'PASS' if g2c_stats['g2_pass'] else 'FAIL'}", flush=True)

    # -- Apply extended map to real data ---------------------------------------
    print(f"\n[10] Real data application (obs = {REAL_GAMMA_OBS:.4f}) ...",
          flush=True)
    gamma_cal_ext = calibrate_pt(ext_cal_map, REAL_GAMMA_OBS)
    nint_ext = neyman_ci(ext_cal_map, REAL_GAMMA_OBS)

    if gamma_cal_ext is None:
        print("    gamma_cal: not computable (obs outside mean-curve range)",
              flush=True)
    else:
        print(f"    gamma_cal = {gamma_cal_ext:.4f}", flush=True)
    print(f"    Neyman 95% CI: [{nint_ext['lo_bound']}, {nint_ext['hi_bound']}]",
          flush=True)
    print(f"    Open ends: lo={nint_ext['open_lo']}, hi={nint_ext['open_hi']}",
          flush=True)
    print(f"    Contains zero: {nint_ext['contains_zero']}", flush=True)
    print(f"    Note: {nint_ext['note']}", flush=True)

    # -- Also apply original E32 (5-gamma) map for comparison -----------------
    print("\n[10b] Comparison: original 5-gamma map ...", flush=True)
    base_cal_map = build_cal_map(base_cal, GAMMA_GRID_BASE)
    gamma_cal_base = calibrate_pt(base_cal_map, REAL_GAMMA_OBS)
    nint_base = neyman_ci(base_cal_map, REAL_GAMMA_OBS)
    print(f"    gamma_cal (base)  = {gamma_cal_base}", flush=True)
    print(f"    Neyman 95% CI (base): [{nint_base['lo_bound']:.4f}, "
          f"{nint_base['hi_bound']} (open_hi={nint_base['open_hi']})]",
          flush=True)

    # -- Serialize -------------------------------------------------------------
    def _map_json(m: dict) -> dict:
        return {
            "grid":   [float(x) for x in m["grid"]],
            "g_arr":  [float(x) for x in m["g_arr"]],
            "m_arr":  [float(x) for x in m["m_arr"]],
            "lo_arr": [float(x) for x in m["lo_arr"]],
            "hi_arr": [float(x) for x in m["hi_arr"]],
            "monotone_mean": m["monotone_mean"],
            "monotone_lo":   m["monotone_lo"],
            "monotone_hi":   m["monotone_hi"],
            "grid_stats":    {str(k): v for k, v in m["grid_stats"].items()},
        }

    elapsed = time.time() - t0
    out = {
        "experiment": "E32_followup_extended_grid",
        "note":       "Not pre-registered; the pre-registered E32 G2 verdict is unchanged.",
        "config": {
            "gamma_grid_base":  GAMMA_GRID_BASE,
            "gamma_grid_ext":   GAMMA_GRID_EXT,
            "gamma_grid_full":  GAMMA_GRID_FULL,
            "n_cal_reps":       N_CAL_REPS,
            "n_g2c_reps":       N_G2C_REPS,
            "g2c_rep_range":    [G2C_REP_OFFSET, G2C_REP_OFFSET + N_G2C_REPS - 1],
            "n_workers":        N_WORKERS,
            "ckpt_interval":    CKPT_INTERVAL,
            "g2_bias_thr":      G2_BIAS_THR,
            "g2_cov_lo":        G2_COV_LO,
            "g2_cov_hi":        G2_COV_HI,
            "real_gamma_obs":   REAL_GAMMA_OBS,
        },
        "ext_calibration_map": _map_json(ext_cal_map),
        "new_gamma_stats": {
            "+0.8": {
                "n": int(len(hA)),
                "mean": float(np.mean(hA)),
                "std":  float(np.std(hA, ddof=1)),
                "q2_5":  float(np.percentile(hA, 2.5)),
                "q97_5": float(np.percentile(hA, 97.5)),
            },
            "+1.2": {
                "n": int(len(hB)),
                "mean": float(np.mean(hB)),
                "std":  float(np.std(hB, ddof=1)),
                "q2_5":  float(np.percentile(hB, 2.5)),
                "q97_5": float(np.percentile(hB, 97.5)),
            },
        },
        "g2_extended_gamma_0p4": {
            "reps_used": [G2C_REP_OFFSET, G2C_REP_OFFSET + N_G2C_REPS - 1],
            **g2c_stats,
        },
        "real_data_ext": {
            "gamma_hat_obs": float(REAL_GAMMA_OBS),
            "gamma_cal":     float(gamma_cal_ext) if gamma_cal_ext is not None else None,
            "neyman_ci": {
                "lo_bound":      nint_ext["lo_bound"],
                "hi_bound":      nint_ext["hi_bound"],
                "open_lo":       nint_ext["open_lo"],
                "open_hi":       nint_ext["open_hi"],
                "contains_zero": nint_ext["contains_zero"],
                "note":          nint_ext["note"],
            },
        },
        "real_data_base_for_comparison": {
            "gamma_cal":     float(gamma_cal_base) if gamma_cal_base is not None else None,
            "neyman_ci_open_hi": nint_base["open_hi"],
            "neyman_ci_lo_bound": nint_base["lo_bound"],
            "neyman_ci_hi_bound": nint_base["hi_bound"],
        },
        "elapsed_sec": float(elapsed),
    }

    results_path = OUT_DIR / "E32_followup_extended_grid.json"
    results_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print(f"\nResults -> {results_path}", flush=True)

    print(f"\n=== E32_followup complete ({elapsed:.1f}s = {elapsed/3600:.2f}h) ===",
          flush=True)


if __name__ == "__main__":
    main()
