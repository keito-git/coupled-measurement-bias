"""
E36: over-shoot of the oracle estimator (true categories + latent H) by modifier definition and sample size.

Scheme C' generator (as in E27e) with true gamma in {0, -0.4, -0.8} and n in {500, 2000, 8000} dialogues,
N_REPS replications per cell. The oracle data are fitted with three modifiers:
  (i)   wc: within-cell residualised latent H (fit_within_cell)
  (ii)  rh: latent H centred by the simulated global mean (fit_raw_H)
  (iii) fg: latent H centred by the fixed empirical mean H_gm
At n = 2000 the dialogue lengths and seeds are those of E27e; for n = 500 and 8000 the lengths are resampled.
Before the run, the E27e oracle estimates (n = 2000, gamma = -0.4, reps 0-4) are reproduced as a check.

Outputs (RESULTS_ROOT/E36): E36_ckpt.json, E36_results.json
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402

from dtsim_core import (  # noqa: E402
    K, REAL_GAMMA_WITHINCELL_FULL, N_RESTARTS, load_el_raw, empirical_q_pool, compute_empirical_confusion,
    empirical_H_pool, residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded, unpack_v_dt,
)
from dtsim_fits import N_REFINE, Q_NOISE_STD, fit_within_cell, fit_raw_H  # noqa: E402
from dtsim_kernels import _seed_nb, _process_one_dlg_scheme_c_nb, _warmup_numba  # noqa: E402

OUT_DIR = config.results_dir("E36")
GAMMA_CONDITIONS = [0.0, -0.4, -0.8]
N_REPS = 100                    # replications per cell
N_LIST = [500, 2000, 8000]      # number of dialogues
N_WORKERS = config.n_workers(3)
CKPT_INTERVAL = 20

# generator seed offset per n (offset 0 at n = 2000 reproduces the E27e seeds);
# dialogue lengths for n != 2000 are resampled with seed rep * 13337 + N_OFFSET[n]
N_OFFSET: Dict[int, int] = {500: 1_000_000, 2000: 0, 8000: 2_000_000}


# ============================================================================
# modifier (iii): fixed global-mean centring
# ============================================================================

def fit_fixed_global_H(
    dlgs_evs: List[List[dict]],
    H_gm_fixed: float,
    seed: int,
) -> dict:
    """
    Modifier (iii): centre H by the fixed empirical mean H_gm instead of the simulated H_bar.
    The centring constant is absorbed into alpha, so this should agree with fit_raw_H up to
    numerical error.
    """
    dlgs_dt = [
        {"cats":  np.array([e["cat"] for e in evs], dtype=np.int64),
         "Hs_c": np.array([e["H"]   for e in evs], dtype=np.float64) - H_gm_fixed}
        for evs in dlgs_evs
    ]
    return fit_dt_bounded(dlgs_dt, H_gm_fixed, n_restarts=N_RESTARTS, seed=seed)


# ============================================================================
# Worker
# ============================================================================

def _e36_worker(args: tuple) -> dict:
    """One replication: Scheme C' oracle data fitted with the three modifiers.
    Generator seed = rep * 91723 + g_seed * 1000 + 77 + n_offset."""
    (rep, n_dlg, dlg_lengths,
     mu_arr, alpha_flat, beta_val, gamma_inj,
     H_dist_arr, H_gm_val, q_pool_arr, confusion_arr,
     H_sorted_arr, q_sorted_arr, Q_NOISE_STD_val,
     n_offset) = args

    K_val    = 7
    N_ANN    = 5
    alpha_mat = alpha_flat.reshape(K_val, K_val)

    g_seed = int(round((gamma_inj + 2.0) * 10000))
    _seed_nb(rep * 91723 + g_seed * 1000 + 77 + n_offset)

    oracle_dlgs: List[List[dict]] = []

    for dlg_len in dlg_lengths:
        if dlg_len < 2:
            continue

        (oc, oh, op, sc, sh, sp, H_raw) = _process_one_dlg_scheme_c_nb(
            dlg_len, mu_arr, alpha_mat, float(beta_val), gamma_inj,
            H_dist_arr, H_gm_val, H_sorted_arr, q_sorted_arr,
            confusion_arr, Q_NOISE_STD_val, N_REFINE, K_val, N_ANN,
        )

        # oracle data: true categories and latent H
        oracle_dlgs.append([
            {"cat": int(oc[m]), "H": float(oh[m]), "plurality": int(op[m])}
            for m in range(dlg_len)
        ])

    orc_dlgs_pl1 = [d for d in oracle_dlgs if len(d) >= 2]

    def _fit_safe(fn, *args, **kwargs):
        if len(orc_dlgs_pl1) < 5:
            return {"gamma_hat": float("nan"), "converged": False, "at_bound": False}
        try:
            return fn(*args, **kwargs)
        except Exception:
            return {"gamma_hat": float("nan"), "converged": False, "at_bound": False}

    fit_i = _fit_safe(fit_within_cell, orc_dlgs_pl1, seed=rep)

    fit_ii = _fit_safe(fit_raw_H, orc_dlgs_pl1, seed=rep + 100_000)

    fit_iii = _fit_safe(fit_fixed_global_H, orc_dlgs_pl1, H_gm_val, seed=rep + 200_000)

    return {
        "rep":       rep,
        "n_dlg":     n_dlg,
        "gamma_inj": float(gamma_inj),
        "n_offset":  n_offset,
        "wc":  {"gamma_hat": float(fit_i["gamma_hat"]),
                "converged": bool(fit_i["converged"])},
        "rh":  {"gamma_hat": float(fit_ii["gamma_hat"]),
                "converged": bool(fit_ii["converged"])},
        "fg":  {"gamma_hat": float(fit_iii["gamma_hat"]),
                "converged": bool(fit_iii["converged"])},
    }


# ============================================================================
# Pool runner with resumable checkpoints
# ============================================================================

def _run_with_progress(
    args_list: list,
    checkpoint_path: Path,
    key_name: str = "results",
    n_workers: int = N_WORKERS,
) -> list:
    done_keys: set = set()
    results: list = []
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            results = data.get(key_name, [])
            done_keys = {(r["rep"], r["gamma_inj"], r["n_dlg"]) for r in results}
            print(f"  Checkpoint: {len(done_keys)} tasks done", flush=True)
        except Exception as ex:
            print(f"  Warning: checkpoint load failed ({ex}); starting fresh", flush=True)

    pending = [a for a in args_list
               if (a[0], float(a[6]), a[1]) not in done_keys]  # (rep, gamma_inj, n_dlg)
    n_total = len(args_list)

    if not pending:
        print(f"  All {n_total} tasks done (checkpoint).", flush=True)
        return results

    print(f"  {len(pending)} tasks pending on {n_workers} workers ...", flush=True)

    ctx = mp.get_context("spawn")

    unsaved = 0

    def _save():
        checkpoint_path.write_text(
            json.dumps({key_name: results}, ensure_ascii=False,
                       default=lambda o: o.item() if hasattr(o, "item") else str(o)),
            encoding="utf-8",
        )

    with ctx.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_e36_worker, pending):
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
# Aggregation
# ============================================================================

def _cell_stats(
    results: list,
    gamma_val: float,
    n_dlg: int,
    modifier: str,  # "wc", "rh", "fg"
) -> dict:
    """Mean, SD and Monte-Carlo SE of gamma_hat - gamma for one cell."""
    vals = []
    for r in results:
        if abs(r["gamma_inj"] - gamma_val) < 1e-9 and r["n_dlg"] == n_dlg:
            gh = r[modifier]["gamma_hat"]
            if not math.isnan(gh):
                vals.append(gh)
    if not vals:
        return {"n": 0, "mean_gh": float("nan"), "mean_bias": float("nan"),
                "sd": float("nan"), "mc_se": float("nan"), "n_nan": 0}
    arr = np.array(vals, dtype=float)
    n = len(arr)
    n_total = sum(1 for r in results
                  if abs(r["gamma_inj"] - gamma_val) < 1e-9 and r["n_dlg"] == n_dlg)
    return {
        "n":         n,
        "n_nan":     n_total - n,
        "mean_gh":   float(np.mean(arr)),
        "mean_bias": float(np.mean(arr) - gamma_val),
        "sd":        float(np.std(arr, ddof=1)) if n > 1 else float("nan"),
        "mc_se":     float(np.std(arr, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
    }


def aggregate(results: list) -> dict:
    out: dict = {}
    for g in GAMMA_CONDITIONS:
        gk = f"g{str(g).replace('-', 'm').replace('.', 'p')}"
        for n in N_LIST:
            for mod in ["wc", "rh", "fg"]:
                key = f"{mod}_n{n}_{gk}"
                out[key] = _cell_stats(results, g, n, mod)
    return out


# ============================================================================
# Reproduction check
# ============================================================================

def run_audit_check(
    mu_r: np.ndarray,
    alpha_r: np.ndarray,
    beta_r: float,
    H_dist: np.ndarray,
    H_gm_val: float,
    q_pool: np.ndarray,
    confusion: np.ndarray,
    real_dlg_lengths: np.ndarray,
    H_sorted: np.ndarray,
    q_sorted: np.ndarray,
) -> dict:
    """Reproduce the E27e oracle estimates at n = 2000, gamma = -0.4, reps 0-4 (pass if |delta| < 0.001)."""
    print("\nReproduction check: E27e oracle, n = 2000, gamma = -0.4, reps 0-4", flush=True)

    # oracle gamma_hat in E27e_partial_schemeC.json
    e27e_ref = {
        0: -0.4604,
        1: -0.6495,
        2: -0.5502,
        3: -0.2734,
        4: -0.5762,
    }

    comparisons = []
    gamma_inj   = -0.4
    g_seed      = int(round((gamma_inj + 2.0) * 10000))
    K_val  = 7
    N_ANN  = 5
    alpha_mat = alpha_r.reshape(K_val, K_val)

    for rep in range(5):
        _seed_nb(rep * 91723 + g_seed * 1000 + 77)

        oracle_dlgs: List[List[dict]] = []
        for dlg_len in real_dlg_lengths:
            if dlg_len < 2:
                continue
            (oc, oh, op, sc, sh, sp, H_raw) = _process_one_dlg_scheme_c_nb(
                int(dlg_len), mu_r, alpha_mat, float(beta_r), gamma_inj,
                H_dist, H_gm_val, H_sorted, q_sorted,
                confusion, Q_NOISE_STD, N_REFINE, K_val, N_ANN,
            )
            oracle_dlgs.append([
                {"cat": int(oc[m]), "H": float(oh[m]), "plurality": int(op[m])}
                for m in range(int(dlg_len))
            ])

        orc_pl1 = [d for d in oracle_dlgs if len(d) >= 2]
        wc = fit_within_cell(orc_pl1, seed=rep)
        e36_val  = float(wc["gamma_hat"])
        e27e_val = e27e_ref[rep]
        delta    = e36_val - e27e_val

        comparisons.append({
            "rep":      rep,
            "e27e_val": e27e_val,
            "e36_val":  e36_val,
            "delta":    delta,
            "pass":     abs(delta) < 0.001,
        })
        print(f"  rep={rep}: E27e={e27e_val:+.4f}  E36={e36_val:+.4f}  "
              f"delta={delta:+.6f}  {'PASS' if abs(delta)<0.001 else 'FAIL'}",
              flush=True)

    all_pass = all(c["pass"] for c in comparisons)
    status = "PASS" if all_pass else "FAIL"
    print(f"Reproduction check: {status}", flush=True)

    return {"pass": all_pass, "comparisons": comparisons}


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    t0 = time.time()
    print("=" * 68, flush=True)
    print("E36_oracle: Oracle over-shoot diagnosis (3 modifiers x 3 n x 3 gamma)")
    print("=" * 68, flush=True)

    print("\n[1] Loading real EL data ...", flush=True)
    raw_dlgs    = load_el_raw()
    q_pool      = empirical_q_pool(raw_dlgs)
    confusion   = compute_empirical_confusion(raw_dlgs)
    H_dist      = empirical_H_pool(raw_dlgs)
    real_dlg_lengths = np.array([len(d) for d in raw_dlgs], dtype=np.int64)
    n_real      = len(raw_dlgs)
    H_gm_val    = float(H_dist.mean())
    H_sorted    = np.sort(H_dist)
    q_sorted    = np.sort(q_pool)[::-1]
    print(f"    {n_real} dialogues, H_gm={H_gm_val:.4f}", flush=True)

    print("\n[2] Fitting DT params (for DGP) ...", flush=True)
    raw_dt = [
        {"cats":   np.array([e["cat"] for e in d], dtype=np.int64),
         "Hs_raw": np.array([e["H"]   for e in d], dtype=float)}
        for d in raw_dlgs
    ]
    dlgs_r   = residualize_within_cell(raw_dt)
    H_bar_r  = global_H_mean(dlgs_r)
    dlgs_dtr = prepare_dt_dlgs(dlgs_r, H_bar_r)
    real_fit = fit_dt_bounded(dlgs_dtr, H_bar_r, n_restarts=3, seed=0)
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

    print("\n[3] numba JIT warmup ...", flush=True)
    t_jit0 = time.time()
    _warmup_numba(K)
    print(f"    JIT warmup: {time.time() - t_jit0:.2f}s", flush=True)

    audit = run_audit_check(
        mu_r, alpha_flat, float(beta_r),
        H_dist, H_gm_val,
        q_pool, confusion,
        real_dlg_lengths, H_sorted, q_sorted,
    )
    if not audit["pass"]:
        print("Reproduction check failed; stopping.", flush=True)
        out = {
            "experiment": "E36_oracle",
            "audit": audit,
            "status": "ABORTED: audit FAIL",
        }
        (OUT_DIR / "E36_results.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return


    # n = 2000: real dialogue lengths; n = 500, 8000: lengths resampled per rep
    print("\n[4] Building worker args ...", flush=True)
    all_args = []
    for n in N_LIST:
        n_offset = N_OFFSET[n]
        for g in GAMMA_CONDITIONS:
            for rep in range(N_REPS):
                if n == 2000:
                    dlg_lengths = real_dlg_lengths
                else:
                    rng_len = np.random.default_rng(rep * 13337 + n_offset)
                    dlg_lengths = rng_len.choice(real_dlg_lengths, size=n, replace=True)

                all_args.append((
                    rep, n, dlg_lengths,
                    mu_r, alpha_flat, float(beta_r), float(g),
                    H_dist, H_gm_val, q_pool, confusion,
                    H_sorted, q_sorted, Q_NOISE_STD, n_offset,
                ))

    print(f"    Total tasks: {len(all_args)}", flush=True)

    ckpt_path = OUT_DIR / "E36_ckpt.json"
    print(f"\n[5] Running {len(all_args)} tasks on {N_WORKERS} workers ...", flush=True)
    t5 = time.time()
    results = _run_with_progress(all_args, ckpt_path)
    print(f"    Done in {time.time() - t5:.1f}s", flush=True)

    print("\n[6] Aggregating ...", flush=True)
    agg = aggregate(results)

    for g in GAMMA_CONDITIONS:
        gk = f"g{str(g).replace('-', 'm').replace('.', 'p')}"
        print(f"\n  gamma = {g:+.1f}")
        print(f"  {'n':>6}  {'mod':>4}  {'mean_gh':>9}  {'bias':>8}  {'SD':>7}  {'MC_SE':>7}")
        for n in N_LIST:
            for mod in ["wc", "rh", "fg"]:
                key = f"{mod}_n{n}_{gk}"
                s = agg.get(key, {})
                def _f(x):
                    return f"{x:+.4f}" if not math.isnan(x) else "  NaN  "
                print(f"  {n:6d}  {mod:>4}  {_f(s.get('mean_gh', float('nan'))):>9}  "
                      f"{_f(s.get('mean_bias', float('nan'))):>8}  "
                      f"{_f(s.get('sd', float('nan'))):>7}  "
                      f"{_f(s.get('mc_se', float('nan'))):>7}", flush=True)

    elapsed = time.time() - t0

    out = {
        "experiment":  "E36_oracle",
        "config": {
            "gamma_conditions": GAMMA_CONDITIONS,
            "n_list":           N_LIST,
            "n_reps":           N_REPS,
            "n_workers":        N_WORKERS,
            "n_refine":         N_REFINE,
            "q_noise_std":      Q_NOISE_STD,
            "real_gamma_wc":    REAL_GAMMA_WITHINCELL_FULL,
        },
        "audit":     audit,
        "aggregate": agg,
        "raw_count": len(results),
        "elapsed_sec": float(elapsed),
    }

    results_path = OUT_DIR / "E36_results.json"
    results_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print(f"\nResults -> {results_path}", flush=True)

    print(f"\n=== E36 complete ({elapsed:.1f}s = {elapsed/3600:.2f}h) ===", flush=True)


if __name__ == "__main__":
    main()
