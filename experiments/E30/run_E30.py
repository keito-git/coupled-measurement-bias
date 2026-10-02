"""
E30: AR(1) latent-difficulty annotator model.

Extends the E29 model (q = sigmoid(a - b*u), Dawid-Skene confusion) with an AR(1) difficulty within
each dialogue:
    u_1 ~ N(0, 1),   u_m = rho * u_{m-1} + sqrt(1 - rho^2) * eps_m,   eps_m ~ N(0, 1)
u is discretised to N_U Gauss-Hermite nodes and the marginal likelihood is computed with the HMM forward
algorithm per dialogue. For M3ED (3 annotators, only items with a majority are released) the likelihood is
truncated to vote patterns with a majority.

1. E30_fit.json      EM fit with dialogue-bootstrap SEs (100 reps) for EL Friends, EL EmotionPush, M3ED
2. E30_g3.json       gate G3': recovery of (b, rho) from synthetic data, 30 reps x {0.5, 1, 2} multipliers
3. E30_ppc.json, E30_ppc_gate.json   posterior predictive check (20 copies), criteria
                     |pred - real| < 0.05 (lag-1 entropy correlation), < 0.03 (entropy mean, SD,
                     each plurality bin)
"""

import json
import logging
import math
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import config  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.special import roots_hermite, logsumexp  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

OUT_DIR = config.results_dir("E30")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(OUT_DIR / "E30_run.log"), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

K = 7               # emotion categories
N_U = 19            # Gauss-Hermite nodes for u
N_SIM = 20          # PPC copies
N_BOOT = 100        # bootstrap replications
G3_N_REPS = 30      # G3' replications per setting
G3_MULTS = [0.5, 1.0, 2.0]
RHO_MAX = 0.95
EM_MAX_ITER = 200
EM_TOL = 1e-5
MAX_WORKERS = config.n_workers(4)
LAPLACE = 1e-5

_x_gh, _w_gh = roots_hermite(N_U)
U_NODES = np.sqrt(2) * _x_gh       # nodes for N(0, 1)
W_NODES = _w_gh / np.sqrt(np.pi)   # weights (sum to 1)
PI0 = W_NODES / W_NODES.sum()      # stationary initial distribution
LOG_PI0 = np.log(PI0 + 1e-300)


def _name_offset(name: str) -> int:
    """Deterministic per-dataset seed offset."""
    return zlib.crc32(name.encode()) % 999


# ===============================================================================
# 1. Data Loading (reused from E29)
# ===============================================================================

def load_el_raw_split() -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    files = {
        "friends":     config.FRIENDS_JSON,
        "emotionpush": config.EMOTIONPUSH_JSON,
    }
    splits: Dict[str, List[List[np.ndarray]]] = {}
    for sname, fpath in files.items():
        data = json.loads(fpath.read_text())
        dlgs: List[List[np.ndarray]] = []
        for dialog in data:
            evs: List[np.ndarray] = []
            for utt in dialog:
                ann = utt.get("annotation", "")
                if len(ann) != K or not ann.isdigit():
                    continue
                votes = np.array([int(c) for c in ann], dtype=np.int64)
                if votes.sum() == 0:
                    continue
                evs.append(votes)
            if len(evs) >= 2:
                dlgs.append(evs)
        splits[sname] = dlgs
        log.info(f"EL {sname}: {len(dlgs)} dialogues, "
                 f"{sum(len(d) for d in dlgs)} items")
    return splits["friends"], splits["emotionpush"]


def load_m3ed_votes() -> List[List[np.ndarray]]:
    import pandas as pd
    df = pd.read_parquet(config.M3ED_PARQUET)
    df = df[df["dataset_source"] == "m3ed"]
    df = df[df["n_raters"] == 3].copy()
    df = df.sort_values(["dialog_id", "turn_id"])
    dlgs: List[List[np.ndarray]] = []
    for did, grp in df.groupby("dialog_id"):
        evs: List[np.ndarray] = []
        for _, row in grp.iterrows():
            votes = np.round(row["p_dist"] * 3).astype(np.int64)
            if votes.sum() != 3:
                diff = 3 - int(votes.sum())
                fracs = row["p_dist"] * 3 - votes
                idx = np.argsort(fracs)[::-1]
                for i in range(abs(diff)):
                    votes[idx[i]] += 1 if diff > 0 else -1
            evs.append(votes)
        if len(evs) >= 2:
            dlgs.append(evs)
    log.info(f"M3ED: {len(dlgs)} dialogues, "
             f"{sum(len(d) for d in dlgs)} items (n_raters=3 only)")
    return dlgs


# ===============================================================================
# 2. Statistics (identical to E29)
# ===============================================================================

def compute_vote_stats(dlgs: List[List[np.ndarray]], n_ann: int,
                       label: str = "real") -> Dict:
    all_votes: List[np.ndarray] = []
    H_obs: List[float] = []
    consec_pairs_h: List[Tuple[float, float]] = []
    for dlg in dlgs:
        h_dlg: List[float] = []
        for v in dlg:
            tot = v.sum()
            p = v / tot
            h = float(-(p * np.log(p + 1e-15)).sum())
            H_obs.append(h); h_dlg.append(h); all_votes.append(v)
        for i in range(len(h_dlg) - 1):
            consec_pairs_h.append((h_dlg[i], h_dlg[i + 1]))
    H_arr = np.array(H_obs)
    votes_mat = np.stack(all_votes)
    plur = votes_mat.max(axis=1)
    plur_dist: Dict[str, float] = {}
    n_items = len(all_votes)
    for pl in range(n_ann, 0, -1):
        plur_dist[f"{pl}/{n_ann}"] = round(float((plur == pl).sum()) / n_items, 6)
    q10, q25, q50, q75, q90 = np.percentile(H_arr, [10, 25, 50, 75, 90])
    distinct = (votes_mat > 0).sum(axis=1)
    if len(consec_pairs_h) >= 3:
        h1 = np.array([p[0] for p in consec_pairs_h])
        h2 = np.array([p[1] for p in consec_pairs_h])
        consec_corr = float(pearsonr(h1, h2)[0]) if h1.std() > 1e-9 and h2.std() > 1e-9 else float("nan")
    else:
        consec_corr = float("nan")
    return {"n_dialogues": len(dlgs), "n_items": n_items, "n_ann": n_ann,
            "plurality_dist": plur_dist,
            "H_obs_mean": float(H_arr.mean()), "H_obs_sd": float(H_arr.std()),
            "H_obs_q10": float(q10), "H_obs_q25": float(q25),
            "H_obs_q50": float(q50), "H_obs_q75": float(q75),
            "H_obs_q90": float(q90),
            "mean_distinct_cats": float(distinct.mean()),
            "consec_H_corr": consec_corr, "label": label}


def aggregate_sim_stats(sim_stats_list: List[Dict]) -> Dict:
    keys = ["H_obs_mean", "H_obs_sd", "H_obs_q10", "H_obs_q25",
            "H_obs_q50", "H_obs_q75", "H_obs_q90",
            "mean_distinct_cats", "consec_H_corr"]
    result: Dict = {"n_copies": len(sim_stats_list)}
    pl_keys = list(sim_stats_list[0]["plurality_dist"].keys())
    for pk in pl_keys:
        vals = [s["plurality_dist"][pk] for s in sim_stats_list]
        result[f"plurality_{pk}_mean"] = float(np.mean(vals))
        result[f"plurality_{pk}_sd"] = float(np.std(vals, ddof=1))
    for k in keys:
        vals = [s[k] for s in sim_stats_list if not math.isnan(s[k])]
        if vals:
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_sd"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        else:
            result[f"{k}_mean"] = float("nan")
            result[f"{k}_sd"] = float("nan")
    return result


# ===============================================================================
# 3. AR(1) Model Core
# ===============================================================================

def make_transition(rho: float) -> np.ndarray:
    """Row-stochastic AR(1) transition matrix (N_U, N_U).
    T[j,k] proportional to N(u_k; rho*u_j, sqrt(1-rho^2)) * W_NODES[k].
    """
    if abs(rho) < 1e-8:
        return np.tile(PI0, (N_U, 1))
    std = max(math.sqrt(1.0 - rho * rho), 1e-8)
    # diff[j,k] = u_k - rho*u_j
    diff = U_NODES[np.newaxis, :] - rho * U_NODES[:, np.newaxis]  # (N_U, N_U)
    log_phi = -0.5 * (diff / std) ** 2                             # unnormalized Gaussian
    log_T = log_phi + np.log(W_NODES)[np.newaxis, :]               # add quadrature weight
    log_T -= logsumexp(log_T, axis=1, keepdims=True)               # row-normalize
    return np.exp(log_T)


def compute_log_pvcu(
    votes: np.ndarray,    # (M, K) integer vote counts
    n_ann: int,
    a: float, b: float,
    log_conf_nd: np.ndarray,  # (K, K) log confusion, diagonal=0
) -> np.ndarray:
    """Compute log P(v_m | c, u_j) for all (m, c, j).
    Returns shape (M, K, N_U).
    """
    q = 1.0 / (1.0 + np.exp(-(a - b * U_NODES)))  # (N_U,)
    q = np.clip(q, 1e-9, 1.0 - 1e-9)
    log_q   = np.log(q)
    log_1mq = np.log(1.0 - q)

    # Binomial-like term: votes[m,c]*log(q_j) + (n_ann-votes[m,c])*log(1-q_j)
    # shape: (M, K, N_U)
    vc  = votes[:, :, np.newaxis].astype(float)                      # (M, K, 1)
    vw  = (n_ann - votes)[:, :, np.newaxis].astype(float)            # (M, K, 1)
    binom = vc * log_q[np.newaxis, np.newaxis, :] + vw * log_1mq[np.newaxis, np.newaxis, :]

    # Confusion term: sum_{k!=c} votes[m,k]*log_conf[c,k]   shape: (M, K)
    conf_part = votes.astype(float) @ log_conf_nd.T

    return binom + conf_part[:, :, np.newaxis]  # (M, K, N_U)


def compute_log_emission(
    votes: np.ndarray,    # (M, K)
    n_ann: int,
    a: float, b: float,
    log_conf_nd: np.ndarray,
    log_prior: np.ndarray,    # (K,)
    truncate: bool = False,   # M3ED majority-only truncation
    log_p_valid: Optional[np.ndarray] = None,  # (K, N_U) if truncate
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      log_emission: (M, N_U)   log P(v_m | u_j)
      log_pvcu:     (M, K, N_U) log P(v_m | c, u_j)
    """
    log_pvcu = compute_log_pvcu(votes, n_ann, a, b, log_conf_nd)  # (M, K, N_U)

    if truncate and log_p_valid is not None:
        # Subtract log P(valid | c, u_j) from log P(v | c, u_j)
        log_pvcu_trunc = log_pvcu - log_p_valid[np.newaxis, :, :]  # (M, K, N_U)
        log_emission = logsumexp(
            log_prior[np.newaxis, :, np.newaxis] + log_pvcu_trunc, axis=1
        )  # (M, N_U)
    else:
        log_emission = logsumexp(
            log_prior[np.newaxis, :, np.newaxis] + log_pvcu, axis=1
        )  # (M, N_U)

    return log_emission, log_pvcu


def compute_log_p_valid(
    n_ann: int,
    a: float, b: float,
    conf: np.ndarray,   # (K, K) confusion matrix (not log)
) -> np.ndarray:
    """
    For M3ED (n_ann=3): compute log P(majority-vote exists | c, u_j) for each (c, j).
    Returns (K, N_U).

    P(no_majority | c, u_j):
      = Binom(3,q)[1] * P(2 wrong in 2 diff cats)
      + Binom(3,q)[0] * P(3 wrong in 3 diff cats)

    P(2 wrong in 2 diff cats) = sum_{k1<k2, k1!=c, k2!=c} 2*conf[c,k1]*conf[c,k2]
                               = 1 - sum_{k!=c} conf[c,k]^2   (since sum_{k!=c}conf[c,k]=1)
    P(3 wrong in 3 diff cats) = 1 - 3*A2 + 2*A3
      where A2 = sum_{k!=c} conf[c,k]^2,  A3 = sum_{k!=c} conf[c,k]^3
    """
    q = 1.0 / (1.0 + np.exp(-(a - b * U_NODES)))  # (N_U,)
    q = np.clip(q, 1e-9, 1.0 - 1e-9)

    # Precompute per-category confusion statistics
    A2 = np.array([np.sum(conf[c] ** 2) for c in range(K)])  # (K,)  (diagonal=0 so ok)
    A3 = np.array([np.sum(conf[c] ** 3) for c in range(K)])  # (K,)

    p2_wrong_diff = 1.0 - A2  # P(2 wrong in 2 different cats) = 1 - sum conf^2
    p3_wrong_diff = 1.0 - 3.0 * A2 + 2.0 * A3  # P(3 wrong in 3 different cats)

    # For each (c, j): P(no_maj | c, u_j)
    # = 3*q*(1-q)^2 * p2_wrong_diff[c]  +  (1-q)^3 * p3_wrong_diff[c]
    # shapes: q (N_U,), others (K,) -> broadcast to (K, N_U)
    q_j    = q[np.newaxis, :]                    # (1, N_U)
    p2_c   = p2_wrong_diff[:, np.newaxis]        # (K, 1)
    p3_c   = p3_wrong_diff[:, np.newaxis]        # (K, 1)

    p_nomaj = (3.0 * q_j * (1.0 - q_j) ** 2 * p2_c
               + (1.0 - q_j) ** 3 * p3_c)       # (K, N_U)
    p_valid = np.clip(1.0 - p_nomaj, 1e-9, 1.0)
    return np.log(p_valid)                        # (K, N_U)


def forward_backward_dlg(
    log_emission: np.ndarray,   # (M, N_U)
    log_T: np.ndarray,          # (N_U, N_U)
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    HMM forward-backward for a single dialogue.
    Returns: gamma (M, N_U), xi (M-1, N_U, N_U), log_Z scalar.
    """
    M, N = log_emission.shape

    # Forward pass (log scale, scaled for stability)
    log_alpha = np.empty((M, N))
    log_alpha[0] = LOG_PI0 + log_emission[0]

    for m in range(1, M):
        # log sum_j alpha[m-1,j] * T[j,k]
        log_alpha[m] = (
            logsumexp(log_alpha[m - 1][:, np.newaxis] + log_T, axis=0)
            + log_emission[m]
        )

    log_Z = logsumexp(log_alpha[M - 1])

    # Backward pass
    log_beta = np.zeros((M, N))   # log_beta[M-1, :] = 0
    for m in range(M - 2, -1, -1):
        log_beta[m] = logsumexp(
            log_T + log_emission[m + 1][np.newaxis, :] + log_beta[m + 1][np.newaxis, :],
            axis=1
        )

    # Smoothed marginals gamma[m, j]
    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)   # (M, N_U)

    # Two-slice marginals xi[m, j, k] for m = 0..M-2
    xi = np.empty((max(M - 1, 0), N, N))
    for m in range(M - 1):
        log_xi = (log_alpha[m][:, np.newaxis]
                  + log_T
                  + log_emission[m + 1][np.newaxis, :]
                  + log_beta[m + 1][np.newaxis, :])
        log_xi -= logsumexp(log_xi.ravel())
        xi[m] = np.exp(log_xi)

    return gamma, xi, log_Z


# ===============================================================================
# 4. EM for AR(1) Annotator Model
# ===============================================================================

def _mstep_ab(V_j: np.ndarray, W_j: np.ndarray, n_ann: int,
              a_init: float, b_init: float) -> Tuple[float, float]:
    """M-step for (a, b) via L-BFGS-B on Q(a,b)."""
    def neg_Q(params: np.ndarray) -> float:
        a, b = params
        q = 1.0 / (1.0 + np.exp(-(a - b * U_NODES)))
        q = np.clip(q, 1e-9, 1.0 - 1e-9)
        log_q   = np.log(q)
        log_1mq = np.log(1.0 - q)
        return -float(np.dot(V_j, log_q) + np.dot(n_ann * W_j - V_j, log_1mq))

    def grad_Q(params: np.ndarray) -> np.ndarray:
        a, b = params
        q = 1.0 / (1.0 + np.exp(-(a - b * U_NODES)))
        q = np.clip(q, 1e-9, 1.0 - 1e-9)
        dq = q * (1.0 - q)
        resid = V_j - n_ann * q * W_j
        gA = -float(np.dot(resid, dq / (q * (1.0 - q) + 1e-30)))
        gB = float(np.dot(resid * U_NODES, dq / (q * (1.0 - q) + 1e-30)))
        return np.array([gA, gB])

    res = minimize(neg_Q, x0=np.array([a_init, b_init]),
                   jac=grad_Q,
                   bounds=[(-5.0, 5.0), (0.0, 5.0)],
                   method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    return float(res.x[0]), float(res.x[1])


def _mstep_rho(xi_sum: np.ndarray, current_rho: float) -> float:
    """M-step for rho via Brent's method on Q(rho).
    xi_sum: (N_U, N_U) accumulated two-slice marginals.
    """
    def neg_Q_rho(rho: float) -> float:
        T = make_transition(rho)
        log_T = np.log(T + 1e-300)
        return -float(np.sum(xi_sum * log_T))

    res = minimize_scalar(neg_Q_rho, bounds=(0.0, RHO_MAX), method="bounded",
                          options={"xatol": 1e-8, "maxiter": 200})
    return float(np.clip(res.x, 0.0, RHO_MAX))


def fit_ar1_model(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    truncate: bool = False,
    a_init: float = 1.0,
    b_init: float = 0.5,
    rho_init: float = 0.3,
    verbose: bool = False,
) -> Dict:
    """Fit AR(1) annotator model via EM.

    Returns dict with: a, b, rho, logL, n_iter, confusion (K,K), prior (K,).
    """
    t0 = time.time()

    # Initialise parameters
    a, b = a_init, b_init
    rho = rho_init
    prior = np.ones(K) / K
    conf  = np.eye(K) * 0.0   # placeholder; empirical init below

    # Empirical confusion init (plurality as true cat)
    C_init = np.zeros((K, K))
    for dlg in dlgs:
        for v in dlg:
            tc = int(v.argmax())
            for k in range(K):
                if k != tc:
                    C_init[tc, k] += v[k]
    for i in range(K):
        rs = C_init[i].sum()
        if rs > 0:
            C_init[i] /= rs
        else:
            C_init[i] = np.ones(K) / (K - 1)
            C_init[i, i] = 0.0
    conf = C_init.copy()

    # Empirical prior init
    total_votes = np.zeros(K)
    for dlg in dlgs:
        for v in dlg:
            total_votes[int(v.argmax())] += 1
    prior = (total_votes + LAPLACE) / (total_votes + LAPLACE).sum()

    # Prep vote arrays per dialogue
    dlg_votes: List[np.ndarray] = [np.stack(dlg) for dlg in dlgs]  # list of (M, K)

    prev_logL = -np.inf
    for it in range(EM_MAX_ITER):
        # Precompute
        log_conf_nd = np.log(np.maximum(conf, 1e-300)); np.fill_diagonal(log_conf_nd, 0.0)
        log_prior   = np.log(prior + 1e-300)
        T    = make_transition(rho)
        log_T = np.log(T + 1e-300)

        if truncate:
            log_p_valid = compute_log_p_valid(n_ann, a, b, conf)  # (K, N_U)
        else:
            log_p_valid = None

        # Sufficient statistics
        V_j      = np.zeros(N_U)
        W_j      = np.zeros(N_U)
        xi_sum   = np.zeros((N_U, N_U))
        C_new    = np.zeros((K, K))
        prior_acc = np.zeros(K)
        total_logL = 0.0

        for d_idx, (dlg, votes) in enumerate(zip(dlgs, dlg_votes)):
            M = len(dlg)
            log_emission, log_pvcu = compute_log_emission(
                votes, n_ann, a, b, log_conf_nd, log_prior,
                truncate=truncate, log_p_valid=log_p_valid,
            )

            gamma, xi, log_Z = forward_backward_dlg(log_emission, log_T)
            total_logL += log_Z

            # Two-slice xi accumulation (skip length-1 dialogues)
            if M > 1:
                xi_sum += xi.sum(axis=0)  # (N_U, N_U)

            # Conditional c|u posterior  omega[m, c, j] (normalised over c)
            # omega = softmax over c of (log_prior[c] + log_pvcu[m,c,j])
            log_omega_raw = log_prior[np.newaxis, :, np.newaxis] + log_pvcu  # (M,K,N_U)
            if truncate and log_p_valid is not None:
                log_omega_raw = log_omega_raw - log_p_valid[np.newaxis, :, :]
            log_omega = log_omega_raw - logsumexp(log_omega_raw, axis=1, keepdims=True)
            omega = np.exp(log_omega)  # (M, K, N_U)

            # Weighted by gamma[m, j]
            # gamma_omega[m, c, j] = gamma[m,j] * omega[m,c,j]
            gamma_omega = gamma[:, np.newaxis, :] * omega  # (M, K, N_U)

            # Sufficient stats for (a, b)
            # V_j[j] += sum_{m,c} gamma_omega[m,c,j] * votes[m,c]
            V_j += (gamma_omega * votes[:, :, np.newaxis]).sum(axis=(0, 1))
            W_j += gamma.sum(axis=0)   # gamma sums to 1 over u, so W_j[j] = sum_m gamma[m,j]

            # Confusion M-step: C[c,k] += sum_{m,j} gamma_omega[m,c,j] * votes[m,k]
            # = (gamma_omega.sum(axis=2)).T @ votes   shape: (K, M) @ (M, K) = (K, K)
            marginal_c = gamma_omega.sum(axis=2)  # (M, K)
            C_new += marginal_c.T @ votes.astype(float)   # (K, K)

            # Prior accumulation: prior_acc[c] += sum_{m,j} gamma_omega[m,c,j]
            prior_acc += marginal_c.sum(axis=0)   # (K,)

        # M-step
        a, b = _mstep_ab(V_j, W_j, n_ann, a, b)
        rho  = _mstep_rho(xi_sum, rho)

        # Confusion M-step: zero diagonal, Laplace smooth, row-normalise
        np.fill_diagonal(C_new, 0.0)
        C_new = np.maximum(C_new, LAPLACE)
        conf  = C_new / C_new.sum(axis=1, keepdims=True)

        # Prior M-step
        prior = (prior_acc + LAPLACE) / (prior_acc + LAPLACE).sum()

        if verbose and (it % 10 == 0 or it < 5):
            log.info(f"    EM iter {it:4d}: logL={total_logL:.4f}, "
                     f"a={a:.4f}, b={b:.4f}, rho={rho:.4f}")

        if total_logL - prev_logL < EM_TOL and it > 5:
            if verbose:
                log.info(f"    Converged at iter {it} (|delta logL| = "
                         f"{total_logL - prev_logL:.2e})")
            break
        prev_logL = total_logL

    return {
        "a": float(a), "b": float(b), "rho": float(rho),
        "logL": float(total_logL),
        "n_iter": it + 1,
        "fit_time_s": round(time.time() - t0, 2),
        "confusion": conf.tolist(),
        "prior": prior.tolist(),
    }


# ===============================================================================
# 5. Bootstrap
# ===============================================================================

def _boot_worker(args: Tuple) -> Dict:
    """Worker for bootstrap resampling."""
    seed, dlgs, n_ann, a_init, b_init, rho_init, truncate = args
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dlgs), size=len(dlgs), replace=True)
    resampled = [dlgs[i] for i in idx]
    try:
        fit = fit_ar1_model(resampled, n_ann,
                            truncate=truncate,
                            a_init=a_init, b_init=b_init, rho_init=rho_init,
                            verbose=False)
        return {"seed": seed, "a": fit["a"], "b": fit["b"],
                "rho": fit["rho"], "ok": True}
    except Exception as e:
        return {"seed": seed, "a": float("nan"), "b": float("nan"),
                "rho": float("nan"), "ok": False, "err": str(e)}


def compute_bootstrap_ses(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_fit: float, b_fit: float, rho_fit: float,
    truncate: bool = False,
    n_boot: int = N_BOOT,
    n_workers: int = MAX_WORKERS,
    ckpt_path: Optional[Path] = None,
) -> Dict:
    results: List[Dict] = []
    done_seeds: set = set()
    if ckpt_path and ckpt_path.exists():
        try:
            ck = json.loads(ckpt_path.read_text())
            results = ck.get("bootstrap_reps", [])
            done_seeds = {r["seed"] for r in results if r.get("ok")}
            log.info(f"  Bootstrap checkpoint: {len(done_seeds)} reps done")
        except Exception:
            pass

    args_list = [
        (s, dlgs, n_ann, a_fit, b_fit, rho_fit, truncate)
        for s in range(1, n_boot + 1)
        if s not in done_seeds
    ]
    if args_list:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for res in pool.imap_unordered(_boot_worker, args_list):
                results.append(res)
                if len(results) % 10 == 0:
                    log.info(f"Bootstrap: {len(results)}/{n_boot} done")
                    if ckpt_path:
                        ckpt_path.write_text(json.dumps({"bootstrap_reps": results}))

    ok = [r for r in results if r.get("ok")]
    a_vals   = np.array([r["a"]   for r in ok])
    b_vals   = np.array([r["b"]   for r in ok])
    rho_vals = np.array([r["rho"] for r in ok])
    return {
        "n_ok": len(ok),
        "a_se":   float(np.std(a_vals,   ddof=1)) if len(a_vals)   > 1 else float("nan"),
        "b_se":   float(np.std(b_vals,   ddof=1)) if len(b_vals)   > 1 else float("nan"),
        "rho_se": float(np.std(rho_vals, ddof=1)) if len(rho_vals) > 1 else float("nan"),
    }


# ===============================================================================
# 6. Simulate from Fitted Model
# ===============================================================================

def simulate_ar1_from_fitted(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a: float, b: float, rho: float,
    confusion: np.ndarray,
    prior: np.ndarray,
    seed: int,
) -> List[List[np.ndarray]]:
    """Sample votes from the fitted AR(1) annotator model."""
    rng = np.random.default_rng(seed)
    T = make_transition(rho)
    sim_dlgs: List[List[np.ndarray]] = []
    p_norm = prior / prior.sum()
    for dlg in dlgs:
        M = len(dlg)
        # Sample u trajectory via Markov chain
        u_idx = np.empty(M, dtype=int)
        u_idx[0] = int(rng.choice(N_U, p=PI0))
        for m in range(1, M):
            u_idx[m] = int(rng.choice(N_U, p=T[u_idx[m - 1]]))
        sim_dlg: List[np.ndarray] = []
        for m in range(M):
            q = float(1.0 / (1.0 + math.exp(-(a - b * U_NODES[u_idx[m]]))))
            q = max(min(q, 0.9999), 0.0001)
            c = int(rng.choice(K, p=p_norm))
            conf_row = confusion[c].copy()
            conf_row[c] = 0.0
            rs = conf_row.sum()
            if rs > 1e-12:
                conf_row /= rs
            else:
                conf_row = np.ones(K) / (K - 1)
                conf_row[c] = 0.0
                conf_row /= conf_row.sum()
            votes_sim = np.zeros(K, dtype=np.int64)
            for _ in range(n_ann):
                if rng.random() < q:
                    votes_sim[c] += 1
                else:
                    votes_sim[int(rng.choice(K, p=conf_row))] += 1
            sim_dlg.append(votes_sim)
        sim_dlgs.append(sim_dlg)
    return sim_dlgs


# ===============================================================================
# 7. Gate G3'
# ===============================================================================

def _g3_worker(args: Tuple) -> Dict:
    """G3' worker: simulate with (b_true, rho_true), refit, record recovery."""
    rep, dlgs, n_ann, a_fit, b_true, rho_true, confusion, prior, truncate = args
    seed = 42 + rep * 97 + int(b_true * 1000) + int(rho_true * 1000)
    sim_dlgs = simulate_ar1_from_fitted(
        dlgs, n_ann, a_fit, b_true, rho_true, confusion, prior, seed=seed
    )
    try:
        fit = fit_ar1_model(sim_dlgs, n_ann,
                            truncate=truncate,
                            a_init=a_fit, b_init=b_true, rho_init=rho_true,
                            verbose=False)
        return {"rep": rep, "b_true": b_true, "rho_true": rho_true,
                "b_hat": fit["b"], "rho_hat": fit["rho"], "ok": True}
    except Exception as e:
        return {"rep": rep, "b_true": b_true, "rho_true": rho_true,
                "b_hat": float("nan"), "rho_hat": float("nan"),
                "ok": False, "err": str(e)}


def run_g3_prime(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_fit: float, b_fit: float, rho_fit: float,
    confusion: np.ndarray, prior: np.ndarray,
    truncate: bool = False,
    n_reps: int = G3_N_REPS,
    mults: List[float] = G3_MULTS,
    n_workers: int = MAX_WORKERS,
    ckpt_path: Optional[Path] = None,
) -> Dict:
    all_results: List[Dict] = []
    done_keys: set = set()
    if ckpt_path and ckpt_path.exists():
        try:
            ck = json.loads(ckpt_path.read_text())
            all_results = ck.get("g3_reps", [])
            done_keys = {(r["rep"], round(r["b_true"], 6), round(r["rho_true"], 6))
                         for r in all_results if r.get("ok")}
            log.info(f"G3' checkpoint: {len(done_keys)} reps done")
        except Exception:
            pass

    args_list = []
    for m in mults:
        b_test   = m * b_fit
        rho_test = float(np.clip(m * rho_fit, 0.0, RHO_MAX))
        for rep in range(n_reps):
            key = (rep, round(b_test, 6), round(rho_test, 6))
            if key not in done_keys:
                args_list.append(
                    (rep, dlgs, n_ann, a_fit, b_test, rho_test,
                     confusion, prior, truncate)
                )

    if args_list:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for res in pool.imap_unordered(_g3_worker, args_list):
                all_results.append(res)
                if len(all_results) % 10 == 0:
                    log.info(f"G3': {len(all_results)} reps done")
                    if ckpt_path:
                        ckpt_path.write_text(json.dumps({"g3_reps": all_results}))

    # Analysis
    ok = [r for r in all_results if r.get("ok")]
    true_b   = np.array([r["b_true"]   for r in ok])
    rec_b    = np.array([r["b_hat"]    for r in ok])
    true_rho = np.array([r["rho_true"] for r in ok])
    rec_rho  = np.array([r["rho_hat"]  for r in ok])

    corr_b, _ = pearsonr(true_b, rec_b)   if len(ok) >= 3 else (float("nan"), 1.0)
    corr_rho, _ = pearsonr(true_rho, rec_rho) if len(ok) >= 3 else (float("nan"), 1.0)

    per_setting: Dict = {}
    for m in mults:
        b_test   = m * b_fit
        rho_test = float(np.clip(m * rho_fit, 0.0, RHO_MAX))
        reps_s   = [r for r in ok
                    if abs(r["b_true"] - b_test) < 1e-6
                    and abs(r["rho_true"] - rho_test) < 1e-6]
        b_hats   = np.array([r["b_hat"]   for r in reps_s])
        rho_hats = np.array([r["rho_hat"] for r in reps_s])

        bias_b   = float(np.mean(b_hats) - b_test)
        rel_bias_b   = abs(bias_b) / b_test if b_test > 1e-9 else float("nan")
        bias_rho     = float(np.mean(rho_hats) - rho_test)
        rel_bias_rho = abs(bias_rho) / rho_test if rho_test > 1e-9 else float("nan")

        per_setting[f"{m}x"] = {
            "b_true":   float(b_test),
            "b_hat_mean": float(np.mean(b_hats)),
            "b_hat_sd":   float(np.std(b_hats, ddof=1)) if len(b_hats) > 1 else float("nan"),
            "bias_b":   bias_b, "rel_bias_b":   rel_bias_b,
            "rho_true": rho_test,
            "rho_hat_mean": float(np.mean(rho_hats)),
            "rho_hat_sd":   float(np.std(rho_hats, ddof=1)) if len(rho_hats) > 1 else float("nan"),
            "bias_rho": bias_rho, "rel_bias_rho": rel_bias_rho,
            "n_ok": len(reps_s),
        }

    max_rel_bias_b   = max(
        (per_setting[f"{m}x"]["rel_bias_b"]   for m in mults
         if not math.isnan(per_setting[f"{m}x"]["rel_bias_b"])), default=float("nan")
    )
    max_rel_bias_rho = max(
        (per_setting[f"{m}x"]["rel_bias_rho"] for m in mults
         if not math.isnan(per_setting[f"{m}x"]["rel_bias_rho"])), default=float("nan")
    )

    corr_b_ok   = (not math.isnan(corr_b))   and corr_b   > 0.8
    corr_rho_ok = (not math.isnan(corr_rho)) and corr_rho > 0.8
    bias_b_ok   = (not math.isnan(max_rel_bias_b))   and max_rel_bias_b   < 0.20
    bias_rho_ok = (not math.isnan(max_rel_bias_rho)) and max_rel_bias_rho < 0.20

    verdict = "PASS" if (corr_b_ok and corr_rho_ok and bias_b_ok and bias_rho_ok) else "FAIL"
    log.info(
        f"G3': corr_b={corr_b:.4f}(>0.8?{corr_b_ok}), "
        f"corr_rho={corr_rho:.4f}(>0.8?{corr_rho_ok}), "
        f"max_rel_bias_b={max_rel_bias_b:.4f}(<0.20?{bias_b_ok}), "
        f"max_rel_bias_rho={max_rel_bias_rho:.4f}(<0.20?{bias_rho_ok}) -> {verdict}"
    )

    return {
        "n_ok": len(ok), "n_total": len(all_results),
        "corr_b": float(corr_b), "corr_rho": float(corr_rho),
        "corr_b_pass": bool(corr_b_ok), "corr_rho_pass": bool(corr_rho_ok),
        "max_rel_bias_b": float(max_rel_bias_b),
        "max_rel_bias_rho": float(max_rel_bias_rho),
        "bias_b_pass": bool(bias_b_ok), "bias_rho_pass": bool(bias_rho_ok),
        "verdict": verdict,
        "per_setting": per_setting,
    }


# ===============================================================================
# 8. JSON serialisation helper
# ===============================================================================

def _json_safe(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, dict):            return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [_json_safe(v) for v in obj]
    return obj


# ===============================================================================
# 9. Main
# ===============================================================================

def main() -> None:
    t0 = time.time()
    log.info("=" * 68)
    log.info("E30: AR(1) Annotator Model  (start)")
    log.info("=" * 68)

    # -- 1. Load Data ----------------------------------------------------------
    log.info("\n[1] Loading datasets ...")
    el_friends, el_ep = load_el_raw_split()
    m3ed_dlgs = load_m3ed_votes()
    datasets = {
        "el_friends":     (el_friends, 5, False),
        "el_emotionpush": (el_ep,      5, False),
        "m3ed":           (m3ed_dlgs,  3, True),   # truncate=True
    }

    # -- 2. Fit AR(1) models ---------------------------------------------------
    log.info("\n[2] Fitting AR(1) annotator model (EM + bootstrap) ...")
    fit_results: Dict = {}
    for dname, (dlgs, n_ann, truncate) in datasets.items():
        log.info(f"  Fitting {dname} (n_ann={n_ann}, truncate={truncate}) ...")
        t_fit = time.time()
        fit = fit_ar1_model(dlgs, n_ann, truncate=truncate,
                            a_init=1.0, b_init=0.5, rho_init=0.3, verbose=True)
        fit["dataset"] = dname
        log.info(f"    {dname}: a={fit['a']:.4f}, b={fit['b']:.4f}, "
                 f"rho={fit['rho']:.4f}, logL={fit['logL']:.1f}, "
                 f"n_iter={fit['n_iter']}, t={fit['fit_time_s']}s")

        log.info(f"  Bootstrap SEs for {dname} ({N_BOOT} reps, {MAX_WORKERS} workers) ...")
        ckpt = OUT_DIR / f"E30_boot_{dname}_ckpt.json"
        boot = compute_bootstrap_ses(
            dlgs, n_ann, fit["a"], fit["b"], fit["rho"],
            truncate=truncate, n_boot=N_BOOT, n_workers=MAX_WORKERS, ckpt_path=ckpt,
        )
        fit["bootstrap"] = boot
        log.info(f"    a={fit['a']:.4f}+/-{boot['a_se']:.4f}  "
                 f"b={fit['b']:.4f}+/-{boot['b_se']:.4f}  "
                 f"rho={fit['rho']:.4f}+/-{boot['rho_se']:.4f}")
        fit_results[dname] = fit

    fit_path = OUT_DIR / "E30_fit.json"
    fit_path.write_text(json.dumps(_json_safe(fit_results), ensure_ascii=False, indent=2))
    log.info(f"  Fit results saved -> {fit_path}")

    # -- 3. Gate G3' -----------------------------------------------------------
    log.info("\n[3] Gate G3': parameter recovery ...")
    g3_dname = "el_friends"
    g3_dlgs, g3_nann, g3_trunc = datasets[g3_dname]
    g3_fit = fit_results[g3_dname]
    g3_conf = np.array(g3_fit["confusion"])
    g3_prior = np.array(g3_fit["prior"])

    ckpt_g3 = OUT_DIR / "E30_g3_ckpt.json"
    g3 = run_g3_prime(
        g3_dlgs, g3_nann,
        a_fit=g3_fit["a"], b_fit=g3_fit["b"], rho_fit=g3_fit["rho"],
        confusion=g3_conf, prior=g3_prior,
        truncate=g3_trunc,
        n_reps=G3_N_REPS, mults=G3_MULTS,
        n_workers=MAX_WORKERS, ckpt_path=ckpt_g3,
    )
    g3["source_dataset"] = g3_dname

    g3_path = OUT_DIR / "E30_g3.json"
    g3_path.write_text(json.dumps(_json_safe(g3), ensure_ascii=False, indent=2))
    log.info(f"  G3' saved -> {g3_path}")
    log.info(f"  G3' VERDICT: {g3['verdict']}")

    # -- 4. Posterior Predictive Check -----------------------------------------
    log.info("\n[4] Posterior predictive check (20 copies from fitted AR(1)) ...")
    ppc_results: Dict = {}
    for dname, (dlgs, n_ann, _) in datasets.items():
        fd = fit_results[dname]
        conf_d  = np.array(fd["confusion"])
        prior_d = np.array(fd["prior"])
        ppc_copies = []
        for rep in range(N_SIM):
            sim_dlgs = simulate_ar1_from_fitted(
                dlgs, n_ann, fd["a"], fd["b"], fd["rho"],
                conf_d, prior_d,
                seed=rep * 17 + _name_offset(dname),
            )
            cs = compute_vote_stats(sim_dlgs, n_ann, label=f"ppc_{rep}")
            ppc_copies.append(cs)
        ppc_results[dname] = aggregate_sim_stats(ppc_copies)
        log.info(f"  PPC {dname}: done")

    ppc_path = OUT_DIR / "E30_ppc.json"
    ppc_path.write_text(json.dumps(_json_safe(ppc_results), ensure_ascii=False, indent=2))
    log.info(f"  PPC saved -> {ppc_path}")

    # -- 5. Compute real stats and PPC gate check -------------------------------
    log.info("\n[5] Real vote stats for gate check ...")
    real_stats: Dict = {}
    for dname, (dlgs, n_ann, _) in datasets.items():
        real_stats[dname] = compute_vote_stats(dlgs, n_ann, label="real")

    # Save real stats and PPC comparison
    ppc_gate: Dict = {}
    for dname in datasets:
        real = real_stats[dname]
        ppc  = ppc_results[dname]
        gate: Dict = {}
        # consec H corr gate: |pred - real| < 0.05
        pred_consec = ppc["consec_H_corr_mean"]
        real_consec = real["consec_H_corr"]
        diff_consec = abs(pred_consec - real_consec)
        gate["consec_H_corr"] = {
            "real": real_consec, "pred": pred_consec,
            "diff": diff_consec, "pass": bool(diff_consec < 0.05),
        }
        # H mean gate: |pred - real| < 0.03
        pred_h_mean = ppc["H_obs_mean_mean"]
        diff_h_mean = abs(pred_h_mean - real["H_obs_mean"])
        gate["H_obs_mean"] = {
            "real": real["H_obs_mean"], "pred": pred_h_mean,
            "diff": diff_h_mean, "pass": bool(diff_h_mean < 0.03),
        }
        # H sd gate: |pred - real| < 0.03
        pred_h_sd = ppc["H_obs_sd_mean"]
        diff_h_sd = abs(pred_h_sd - real["H_obs_sd"])
        gate["H_obs_sd"] = {
            "real": real["H_obs_sd"], "pred": pred_h_sd,
            "diff": diff_h_sd, "pass": bool(diff_h_sd < 0.03),
        }
        # Plurality gate: each bin < 0.03
        plur_pass = True
        plur_checks: Dict = {}
        for pk, rv in real["plurality_dist"].items():
            pv = ppc.get(f"plurality_{pk}_mean", float("nan"))
            diff = abs(pv - rv) if not math.isnan(pv) else float("nan")
            pk_pass = (not math.isnan(diff)) and diff < 0.03
            if not pk_pass:
                plur_pass = False
            plur_checks[pk] = {"real": rv, "pred": pv, "diff": diff, "pass": bool(pk_pass)}
        gate["plurality"] = plur_checks
        gate["plurality_all_pass"] = bool(plur_pass)
        gate["overall_pass"] = bool(
            gate["consec_H_corr"]["pass"]
            and gate["H_obs_mean"]["pass"]
            and gate["H_obs_sd"]["pass"]
            and plur_pass
        )
        ppc_gate[dname] = gate
        log.info(f"  PPC gate {dname}: "
                 f"consec_r={'PASS' if gate['consec_H_corr']['pass'] else 'FAIL'} "
                 f"(|{diff_consec:.3f}|<0.05), "
                 f"H_mean={'PASS' if gate['H_obs_mean']['pass'] else 'FAIL'} "
                 f"(|{diff_h_mean:.3f}|<0.03), "
                 f"H_sd={'PASS' if gate['H_obs_sd']['pass'] else 'FAIL'} "
                 f"(|{diff_h_sd:.3f}|<0.03), "
                 f"plur={'PASS' if plur_pass else 'FAIL'} -> "
                 f"{'PASS' if gate['overall_pass'] else 'FAIL'}")

    gate_path = OUT_DIR / "E30_ppc_gate.json"
    gate_path.write_text(
        json.dumps(_json_safe({"gate": ppc_gate, "real_stats": real_stats}),
                   ensure_ascii=False, indent=2))
    log.info(f"  PPC gate saved -> {gate_path}")

    elapsed = time.time() - t0
    log.info(f"\n=== E30 complete ({elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()
