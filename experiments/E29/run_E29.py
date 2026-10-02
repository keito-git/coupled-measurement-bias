"""
E29: i.i.d. latent-difficulty annotator model.

Model: true category c ~ Cat(prior), item difficulty u ~ N(0, 1), annotator accuracy q = sigmoid(a - b*u);
each annotator votes c with probability q, otherwise draws from row c of a zero-diagonal confusion matrix.

1. E29_simcheck.json  vote statistics of the real corpora (plurality distribution, vote entropy,
                      lag-1 entropy correlation) and of the E27e-style simulators (20 copies each)
2. E29_fit.json       EM fit (20-point Gauss-Hermite quadrature) with dialogue-bootstrap SEs (100 reps)
                      for EL Friends, EL EmotionPush and M3ED
3. E29_g3.json        gate G3, parameter recovery: votes simulated from the fitted model with
                      b in {0.5, 1, 2} x b_hat, 50 refits each; pass if corr(b_true, b_hat) > 0.8 and
                      every |relative bias| < 0.20
4. E29_ppc.json       posterior predictive check (20 copies from the fitted model)
"""
from __future__ import annotations

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
from scipy.optimize import minimize  # noqa: E402
from scipy.special import roots_hermite, logsumexp  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

OUT_DIR = config.results_dir("E29")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "E29_run.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

K = 7               # emotion categories
N_GH = 20           # Gauss-Hermite quadrature points
N_SIM = 20          # simulated copies for simcheck / PPC
N_BOOT = 100        # bootstrap replications
G3_N_REPS = 50      # replications per b setting in G3
G3_B_MULTS = [0.5, 1.0, 2.0]
EM_MAX_ITER = 300
EM_TOL = 1e-6       # convergence tolerance on the log-likelihood
MAX_WORKERS = config.n_workers(4)
Q_NOISE_STD = 0.05  # noise on the linked accuracy (as in E27e)
LAPLACE = 1e-5      # smoothing for confusion and prior


def _name_offset(name: str) -> int:
    """Deterministic per-dataset seed offset."""
    return zlib.crc32(name.encode()) % 999


# ============================================================================
# Section 1: Data Loading
# ============================================================================

def load_el_raw_split() -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]]]:
    """
    Load EmotionLines raw data.

    Returns
    -------
    friends_dlgs, emotionpush_dlgs
    Each is a list of dialogues; each dialogue is a list of vote arrays (K,) int.
    """
    files = {
        "friends":     config.FRIENDS_JSON,
        "emotionpush": config.EMOTIONPUSH_JSON,
    }
    splits: Dict[str, List[List[np.ndarray]]] = {}
    for split_name, fpath in files.items():
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
        splits[split_name] = dlgs
        log.info(f"EL {split_name}: {len(dlgs)} dialogues, "
                 f"{sum(len(d) for d in dlgs)} items")
    return splits["friends"], splits["emotionpush"]


def load_m3ed_votes() -> List[List[np.ndarray]]:
    """
    Load M3ED votes from processed parquet (n_raters == 3 only).

    Returns list of dialogues; each dialogue is a list of vote arrays (K,) int.
    Category order: neutral, joy, sadness, fear, anger, surprise, disgust.
    """
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
                # Adjust: add to largest fractional component
                diff = 3 - int(votes.sum())
                fracs = row["p_dist"] * 3 - votes
                idx = np.argsort(fracs)[::-1]
                for i in range(abs(diff)):
                    votes[idx[i]] += 1 if diff > 0 else -1
            evs.append(votes)
        if len(evs) >= 2:
            dlgs.append(evs)

    n_items = sum(len(d) for d in dlgs)
    log.info(f"M3ED: {len(dlgs)} dialogues, {n_items} items (n_raters=3 only)")
    return dlgs


# ============================================================================
# Section 2: Statistics Computation
# ============================================================================

def compute_vote_stats(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    label: str = "real",
) -> Dict:
    """
    Compute vote distribution statistics for a dataset.

    Returns dict with:
    - plurality_dist: fraction of items at each plurality level
    - H_obs_mean, H_obs_sd, H_obs_quantiles (10,25,50,75,90)
    - mean_distinct_cats: mean number of categories with >=1 vote per item
    - consec_H_corr: Pearson r of H_obs[i] vs H_obs[i+1] (pooled across dialogues)
    """
    all_votes: List[np.ndarray] = []
    H_obs: List[float] = []
    consec_pairs_h: List[Tuple[float, float]] = []

    for dlg in dlgs:
        h_dlg: List[float] = []
        for votes in dlg:
            total = votes.sum()
            p = votes / total
            h = float(-(p * np.log(p + 1e-15)).sum())
            H_obs.append(h)
            h_dlg.append(h)
            all_votes.append(votes)
        for i in range(len(h_dlg) - 1):
            consec_pairs_h.append((h_dlg[i], h_dlg[i + 1]))

    H_arr = np.array(H_obs)
    votes_mat = np.stack(all_votes)  # (N, K)
    plur = votes_mat.max(axis=1)      # (N,) plurality counts

    # Plurality distribution (as fractions)
    plur_levels = list(range(n_ann, 0, -1))  # e.g. [5,4,3,2,1] for EL
    plur_dist: Dict[str, float] = {}
    n_items = len(all_votes)
    for pl in plur_levels:
        frac = float((plur == pl).sum()) / n_items
        plur_dist[f"{pl}/{n_ann}"] = round(frac, 6)

    # H_obs stats
    q10, q25, q50, q75, q90 = np.percentile(H_arr, [10, 25, 50, 75, 90])

    # Distinct categories
    distinct = (votes_mat > 0).sum(axis=1)  # (N,)
    mean_distinct = float(distinct.mean())

    # Consecutive H correlation
    if len(consec_pairs_h) >= 3:
        h1 = np.array([p[0] for p in consec_pairs_h])
        h2 = np.array([p[1] for p in consec_pairs_h])
        if h1.std() > 1e-9 and h2.std() > 1e-9:
            consec_corr = float(pearsonr(h1, h2)[0])
        else:
            consec_corr = float("nan")
    else:
        consec_corr = float("nan")

    return {
        "n_dialogues": len(dlgs),
        "n_items": n_items,
        "n_ann": n_ann,
        "plurality_dist": plur_dist,
        "H_obs_mean": float(H_arr.mean()),
        "H_obs_sd": float(H_arr.std()),
        "H_obs_q10": float(q10),
        "H_obs_q25": float(q25),
        "H_obs_q50": float(q50),
        "H_obs_q75": float(q75),
        "H_obs_q90": float(q90),
        "mean_distinct_cats": mean_distinct,
        "consec_H_corr": consec_corr,
        "label": label,
    }


def aggregate_sim_stats(
    sim_stats_list: List[Dict],
) -> Dict:
    """Compute mean +/- SD across multiple simulated copies."""
    keys = [
        "H_obs_mean", "H_obs_sd",
        "H_obs_q10", "H_obs_q25", "H_obs_q50", "H_obs_q75", "H_obs_q90",
        "mean_distinct_cats", "consec_H_corr",
    ]
    result: Dict = {"n_copies": len(sim_stats_list)}

    # Plurality dist keys
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


# ============================================================================
# Section 3: Vote Simulator (E27e-style)
# ============================================================================

def simulate_votes_for_dlgs(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    confusion: np.ndarray,
    q_pool: np.ndarray,
    H_pool: Optional[np.ndarray],
    mode: str,   # "indep" or "linked"
    seed: int,
    q_noise_std: float = Q_NOISE_STD,
) -> List[List[np.ndarray]]:
    """
    Simulate votes for all dialogues using E27e-style models.

    For "indep": q sampled i.i.d. from q_pool.
    For "linked": q = clip(rank_interp(H_pool, q_pool)(H_real) + noise, 0.2, 1.0)

    Returns new dialogues where each event's vote array is replaced by
    the simulated vote array.
    """
    rng = np.random.default_rng(seed)
    K_local = confusion.shape[0]
    n_items = sum(len(d) for d in dlgs)

    # Precompute flat arrays
    flat_true_cats = np.array(
        [int(v.argmax()) for d in dlgs for v in d], dtype=np.int64
    )
    flat_H_real = np.array(
        [float(-(v / v.sum() * np.log(v / v.sum() + 1e-15)).sum())
         for d in dlgs for v in d],
        dtype=np.float64,
    )

    # Compute q values for all items
    if mode == "indep":
        q_vals = rng.choice(q_pool, size=n_items, replace=True)
    else:  # linked
        H_sorted = np.sort(H_pool)
        q_sorted = np.sort(q_pool)[::-1]  # higher H -> lower q
        ranks = np.searchsorted(H_sorted, flat_H_real) / len(H_sorted)
        q_idx = np.clip((ranks * len(q_sorted)).astype(int), 0, len(q_sorted) - 1)
        q_base = q_sorted[q_idx]
        q_vals = np.clip(q_base + rng.normal(0, q_noise_std, n_items), 0.2, 1.0)

    # Simulate votes
    sim_flat: List[np.ndarray] = []
    for idx in range(n_items):
        tc = int(flat_true_cats[idx])
        q = float(q_vals[idx])
        votes_sim = np.zeros(K_local, dtype=np.int64)
        conf_row = confusion[tc].copy()
        conf_row[tc] = 0.0
        row_sum = conf_row.sum()
        if row_sum > 1e-12:
            conf_row /= row_sum
        else:
            conf_row = np.ones(K_local) / (K_local - 1)
            conf_row[tc] = 0.0
            conf_row /= conf_row.sum()

        for _ in range(n_ann):
            if rng.random() < q:
                votes_sim[tc] += 1
            else:
                chosen = rng.choice(K_local, p=conf_row)
                votes_sim[chosen] += 1
        sim_flat.append(votes_sim)

    # Reconstruct dialogue structure
    sim_dlgs: List[List[np.ndarray]] = []
    idx = 0
    for dlg in dlgs:
        sim_dlg: List[np.ndarray] = []
        for _ in dlg:
            sim_dlg.append(sim_flat[idx])
            idx += 1
        sim_dlgs.append(sim_dlg)

    return sim_dlgs


def empirical_confusion(dlgs: List[List[np.ndarray]], K_local: int) -> np.ndarray:
    """Row-normalised confusion matrix from vote data (using plurality as true cat)."""
    C = np.zeros((K_local, K_local))
    for dlg in dlgs:
        for votes in dlg:
            tc = int(votes.argmax())
            for k in range(K_local):
                if k != tc:
                    C[tc, k] += votes[k]
    for i in range(K_local):
        row_sum = C[i].sum()
        if row_sum > 0:
            C[i] /= row_sum
        else:
            C[i] = np.ones(K_local) / (K_local - 1)
            C[i, i] = 0.0
    return C


def empirical_q_pool_from_dlgs(
    dlgs: List[List[np.ndarray]], n_ann: int
) -> np.ndarray:
    """q = plurality / n_ann pool."""
    return np.array(
        [float(v.max()) / n_ann for d in dlgs for v in d]
    )


def empirical_H_pool_from_dlgs(dlgs: List[List[np.ndarray]]) -> np.ndarray:
    """Pool of raw H values."""
    Hs = []
    for dlg in dlgs:
        for v in dlg:
            p = v / v.sum()
            Hs.append(float(-(p * np.log(p + 1e-15)).sum()))
    return np.array(Hs)


# ============================================================================
# Section 4: Annotator Model (EM with Gauss-Hermite Quadrature)
# ============================================================================

# Precompute GH quadrature nodes and weights once at module load
_x_gh, _w_gh = roots_hermite(N_GH)
U_GH: np.ndarray = np.sqrt(2) * _x_gh      # quadrature nodes for N(0,1)
W_GH: np.ndarray = _w_gh / np.sqrt(np.pi)   # normalised weights


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _e_step(
    votes: np.ndarray,
    a: float,
    b: float,
    confusion: np.ndarray,
    prior: np.ndarray,
    n_ann: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E-step: compute posterior W_norm[i, c, j] and log-marginal log_Z[i].

    Parameters
    ----------
    votes : (N, K) int  vote count arrays
    a, b  : float       annotator model parameters
    confusion : (K, K)  error confusion matrix (zero diagonal)
    prior : (K,)        prior over true categories
    n_ann : int         number of annotators

    Returns
    -------
    W_norm : (N, K, N_GH)   normalised posterior weights
    log_Z  : (N,)           log marginal likelihood per item
    """
    N, K_loc = votes.shape
    n_gh = len(U_GH)

    q_j = _sigmoid(a - b * U_GH)                         # (n_gh,)
    log_q = np.log(np.maximum(q_j, 1e-300))              # (n_gh,)
    log_1mq = np.log(np.maximum(1 - q_j, 1e-300))        # (n_gh,)
    log_wgh = np.log(np.maximum(W_GH, 1e-300))           # (n_gh,)
    log_prior = np.log(np.maximum(prior, 1e-300))         # (K,)

    # Confusion log-probs: set diagonal to 0 (excluded from sum)
    log_conf = np.log(np.maximum(confusion, 1e-300))      # (K, K)
    log_conf_nd = log_conf.copy()
    np.fill_diagonal(log_conf_nd, 0.0)

    # conf_part[i, c] = sum_{k!=c} votes[i,k] * log_conf[c,k]
    # votes @ log_conf_nd.T -> result[i,c] = sum_k votes[i,k] * log_conf_nd[c,k]
    # (diagonal entries are 0, so the k == c term vanishes)
    conf_part = votes.astype(float) @ log_conf_nd.T       # (N, K)

    # log L_i(c, u_j) = v[i,c]*log(q_j) + (n_ann-v[i,c])*log(1-q_j) + conf_part[i,c]
    vc    = votes[:, :, np.newaxis].astype(float)         # (N, K, 1)
    n_err = (n_ann - votes)[:, :, np.newaxis].astype(float)  # (N, K, 1)
    logq  = log_q[np.newaxis, np.newaxis, :]              # (1, 1, n_gh)
    l1mq  = log_1mq[np.newaxis, np.newaxis, :]            # (1, 1, n_gh)

    loglik = vc * logq + n_err * l1mq + conf_part[:, :, np.newaxis]  # (N, K, n_gh)

    # log W_i(c,j) = log_prior[c] + log_wgh[j] + loglik[i,c,j]
    log_W = (loglik
             + log_prior[np.newaxis, :, np.newaxis]
             + log_wgh[np.newaxis, np.newaxis, :])       # (N, K, n_gh)

    # Normalise with logsumexp
    log_W_flat = log_W.reshape(N, -1)                    # (N, K*n_gh)
    log_Z = logsumexp(log_W_flat, axis=1)                # (N,)
    W_norm = np.exp(log_W - log_Z[:, np.newaxis, np.newaxis])  # (N, K, n_gh)

    return W_norm, log_Z


def _m_step_conf_prior(
    votes: np.ndarray,
    W_norm: np.ndarray,
    n_ann: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M-step: update prior and confusion matrix.

    prior[c] propto sum_i sum_j W_norm[i,c,j]
    confusion[c,k] propto sum_i W_c[i,c] * votes[i,k]  for k!=c
    """
    K_loc = votes.shape[1]
    W_c = W_norm.sum(axis=2)        # (N, K) marginal posterior over categories

    new_prior = W_c.sum(axis=0) + LAPLACE      # (K,)
    new_prior /= new_prior.sum()

    # new_conf[c,k] = sum_i W_c[i,c] * votes[i,k]
    new_conf = W_c.T @ votes.astype(float)     # (K, K)
    np.fill_diagonal(new_conf, 0.0)
    new_conf = np.maximum(new_conf, LAPLACE)
    row_sums = new_conf.sum(axis=1, keepdims=True)
    new_conf = new_conf / np.maximum(row_sums, 1e-300)

    return new_prior, new_conf


def _m_step_ab(
    votes: np.ndarray,
    W_norm: np.ndarray,
    a0: float,
    b0: float,
    n_ann: int,
) -> Tuple[float, float]:
    """
    M-step: update a, b by maximising E_W[log L(a,b)].

    Precompute V_j and W_j sufficient stats from W_norm.
    Then optimise Q(a,b) = sum_j [V_j*log(q_j) + (n_ann*W_j-V_j)*log(1-q_j)]
    via L-BFGS-B on 2 parameters.
    """
    # Sufficient stats: V_j[j] = sum_{i,c} W_norm[i,c,j] * votes[i,c]
    #                  W_j[j] = sum_{i,c} W_norm[i,c,j]
    votes_f = votes.astype(float)
    V_j = (W_norm * votes_f[:, :, np.newaxis]).sum(axis=(0, 1))  # (n_gh,)
    W_j = W_norm.sum(axis=(0, 1))                                # (n_gh,)

    def neg_Q_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        a_, b_ = params
        q_j = _sigmoid(a_ - b_ * U_GH)
        log_q = np.log(np.maximum(q_j, 1e-300))
        log_1mq = np.log(np.maximum(1 - q_j, 1e-300))
        n_err_j = n_ann * W_j - V_j
        Q = float(np.dot(V_j, log_q) + np.dot(n_err_j, log_1mq))
        gA = float(np.sum(V_j - n_ann * q_j * W_j))
        gB = float(-np.dot(U_GH, V_j - n_ann * q_j * W_j))
        return -Q, np.array([-gA, -gB])

    result = minimize(
        neg_Q_grad, [a0, b0], method="L-BFGS-B", jac=True,
        bounds=[(-10.0, 10.0), (0.0, 20.0)],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    return float(result.x[0]), float(result.x[1])


def fit_annotator_model_uniform_error(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_init: Optional[float] = None,
    b_init: float = 1.0,
    max_iter: int = EM_MAX_ITER,
    tol: float = EM_TOL,
) -> Dict:
    """
    Fit the annotator model with FIXED uniform error confusion matrix.
    confusion[c, k] = 1/(K-1) for k!=c (uniform errors).
    Only fits a, b and prior.
    """
    K_loc = K
    uniform_conf = np.ones((K_loc, K_loc)) / (K_loc - 1)
    np.fill_diagonal(uniform_conf, 0.0)

    votes_all = np.vstack([np.stack(dlg) for dlg in dlgs])
    N, _ = votes_all.shape

    prior = votes_all.sum(axis=0).astype(float)
    prior = np.maximum(prior, LAPLACE)
    prior /= prior.sum()

    mean_q = float(votes_all.max(axis=1).mean()) / n_ann
    mean_q = np.clip(mean_q, 0.01, 0.99)
    if a_init is None:
        a = float(np.log(mean_q / (1 - mean_q)))
    else:
        a = float(a_init)
    b = float(b_init)

    prev_logL = -np.inf
    n_iter = 0

    for iteration in range(max_iter):
        W_norm, log_Z = _e_step(votes_all, a, b, uniform_conf, prior, n_ann)
        logL = float(log_Z.sum())
        n_iter = iteration + 1

        if abs(logL - prev_logL) < tol and iteration >= 5:
            break
        prev_logL = logL

        # Update only prior (confusion is fixed to uniform)
        W_c = W_norm.sum(axis=2)
        new_prior = W_c.sum(axis=0) + LAPLACE
        new_prior /= new_prior.sum()
        prior = new_prior

        # Update a, b
        a, b = _m_step_ab(votes_all, W_norm, a, b, n_ann)

    W_norm, log_Z = _e_step(votes_all, a, b, uniform_conf, prior, n_ann)
    logL = float(log_Z.sum())

    return {
        "a": float(a), "b": float(b),
        "confusion": "uniform",
        "prior": prior.tolist(),
        "logL": float(logL),
        "n_items": int(N),
        "n_iter": n_iter,
        "variant": "uniform_error",
    }


def fit_annotator_model(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_init: Optional[float] = None,
    b_init: float = 1.0,
    max_iter: int = EM_MAX_ITER,
    tol: float = EM_TOL,
    verbose: bool = True,
) -> Dict:
    """
    Fit the annotator model by EM with GH quadrature.

    Model: true_cat c ~ Cat(prior),  u ~ N(0,1),  q = sigmoid(a - b*u)
    P(vote_j = k | c, u) = q * 1[k==c] + (1-q) * confusion[c,k] * 1[k!=c]

    Parameters
    ----------
    dlgs    : list of dialogues, each a list of (K,) vote arrays
    n_ann   : number of annotators per item
    a_init  : initial a (default: logit of mean empirical q)
    b_init  : initial b
    max_iter, tol : EM stopping criteria

    Returns dict with: a, b, confusion, prior, logL, n_items, n_iter
    """
    # Flatten
    votes_all = np.vstack([np.stack(dlg) for dlg in dlgs])  # (N, K)
    N, K_loc = votes_all.shape

    # Initialise prior from data
    prior = votes_all.sum(axis=0).astype(float)
    prior = np.maximum(prior, LAPLACE)
    prior /= prior.sum()

    # Initialise confusion from plurality (proxy for true cat)
    hard_cats = votes_all.argmax(axis=1)
    confusion = np.zeros((K_loc, K_loc))
    for i in range(N):
        c = int(hard_cats[i])
        for k in range(K_loc):
            if k != c:
                confusion[c, k] += votes_all[i, k]
    for c in range(K_loc):
        row_s = confusion[c].sum()
        if row_s > 0:
            confusion[c] /= row_s
        else:
            confusion[c] = np.ones(K_loc) / (K_loc - 1)
            confusion[c, c] = 0.0

    # Initialise a from empirical mean q
    mean_q = float(votes_all.max(axis=1).mean()) / n_ann
    mean_q = np.clip(mean_q, 0.01, 0.99)
    if a_init is None:
        a = float(np.log(mean_q / (1 - mean_q)))
    else:
        a = float(a_init)
    b = float(b_init)

    prev_logL = -np.inf
    history: List[float] = []
    n_iter = 0

    for iteration in range(max_iter):
        W_norm, log_Z = _e_step(votes_all, a, b, confusion, prior, n_ann)
        logL = float(log_Z.sum())
        history.append(logL)
        n_iter = iteration + 1

        if verbose and (iteration % 20 == 0 or iteration < 5):
            log.info(f"    EM iter {iteration:4d}: logL = {logL:.4f}, "
                     f"a={a:.4f}, b={b:.4f}")

        if abs(logL - prev_logL) < tol and iteration >= 5:
            if verbose:
                log.info(f"    Converged at iter {iteration} "
                         f"(|delta logL| = {abs(logL-prev_logL):.2e})")
            break
        prev_logL = logL

        prior, confusion = _m_step_conf_prior(votes_all, W_norm, n_ann)
        a, b = _m_step_ab(votes_all, W_norm, a, b, n_ann)

    # Final E-step
    W_norm, log_Z = _e_step(votes_all, a, b, confusion, prior, n_ann)
    logL = float(log_Z.sum())
    history.append(logL)

    return {
        "a": float(a),
        "b": float(b),
        "confusion": confusion.tolist(),
        "prior": prior.tolist(),
        "logL": float(logL),
        "n_items": int(N),
        "n_iter": n_iter,
        "history_last5": history[-5:],
    }


# ============================================================================
# Section 5: Bootstrap SEs
# ============================================================================

def _bootstrap_worker(args: tuple) -> Dict:
    """
    One bootstrap rep: resample dialogues, fit model, return (seed, a, b).
    """
    (seed, dlgs, n_ann, a_warm, b_warm) = args
    rng = np.random.default_rng(seed)
    n_dlg = len(dlgs)
    idx = rng.integers(0, n_dlg, size=n_dlg)
    boot_dlgs = [dlgs[i] for i in idx]
    try:
        fit = fit_annotator_model(
            boot_dlgs, n_ann,
            a_init=a_warm, b_init=b_warm,
            max_iter=200, tol=1e-5, verbose=False,
        )
        return {"seed": seed, "a": fit["a"], "b": fit["b"], "ok": True}
    except Exception as exc:
        return {"seed": seed, "a": float("nan"), "b": float("nan"),
                "ok": False, "error": str(exc)}


def compute_bootstrap_ses(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_warm: float,
    b_warm: float,
    n_boot: int = N_BOOT,
    n_workers: int = MAX_WORKERS,
    ckpt_path: Optional[Path] = None,
) -> Dict:
    """
    Bootstrap SEs for (a, b) by resampling dialogues.
    """
    # Resume from checkpoint
    done_seeds: set = set()
    results: List[Dict] = []
    if ckpt_path and ckpt_path.exists():
        try:
            ck = json.loads(ckpt_path.read_text())
            results = ck.get("bootstrap_reps", [])
            done_seeds = {r.get("seed", i) for i, r in enumerate(results)}
            log.info(f"Bootstrap checkpoint: {len(done_seeds)} reps done")
        except Exception:
            pass

    pending = [
        (seed, dlgs, n_ann, a_warm, b_warm)
        for seed in range(n_boot)
        if seed not in done_seeds
    ]

    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for result in pool.imap_unordered(_bootstrap_worker, pending):
                results.append(result)
                if len(results) % 10 == 0:
                    log.info(f"Bootstrap: {len(results)}/{n_boot} done")
                    if ckpt_path:
                        ckpt_path.write_text(
                            json.dumps({"bootstrap_reps": results},
                                       ensure_ascii=False))
    else:
        log.info("Bootstrap: all reps already done")

    a_vals = [r["a"] for r in results if r.get("ok", False)]
    b_vals = [r["b"] for r in results if r.get("ok", False)]

    return {
        "n_reps": len(results),
        "n_ok": len(a_vals),
        "a_mean": float(np.mean(a_vals)) if a_vals else float("nan"),
        "a_se": float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else float("nan"),
        "b_mean": float(np.mean(b_vals)) if b_vals else float("nan"),
        "b_se": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else float("nan"),
    }


# ============================================================================
# Section 6: Gate G3 - Parameter Recovery
# ============================================================================

def _g3_worker(args: tuple) -> Dict:
    """
    One G3 rep: simulate votes with true (a_true, b_true), refit, return b_hat.
    Uses real item structure (plurality categories from dlgs).
    """
    (rep, dlgs, n_ann, a_true, b_true, confusion_true, prior_true) = args
    rng = np.random.default_rng(rep * 91723 + 13)

    # Use real item structure: plurality of actual votes as true category
    flat_true_cats = np.array(
        [int(v.argmax()) for d in dlgs for v in d],
        dtype=np.int64,
    )

    # Sample u for each item, compute q
    n_items = int(flat_true_cats.shape[0])
    u_vals = rng.normal(0, 1, size=n_items)
    q_vals = _sigmoid(a_true - b_true * u_vals)

    # Simulate votes
    K_loc = confusion_true.shape[0]
    sim_flat: List[np.ndarray] = []
    for idx in range(n_items):
        tc = int(flat_true_cats[idx])
        q = float(q_vals[idx])
        votes_sim = np.zeros(K_loc, dtype=np.int64)
        conf_row = confusion_true[tc].copy()
        conf_row[tc] = 0.0
        rs = conf_row.sum()
        if rs > 1e-12:
            conf_row /= rs
        else:
            conf_row = np.ones(K_loc) / (K_loc - 1)
            conf_row[tc] = 0.0
            conf_row /= conf_row.sum()
        for _ in range(n_ann):
            if rng.random() < q:
                votes_sim[tc] += 1
            else:
                votes_sim[int(rng.choice(K_loc, p=conf_row))] += 1
        sim_flat.append(votes_sim)

    # Reconstruct dialogue structure
    idx = 0
    sim_dlgs: List[List[np.ndarray]] = []
    for dlg in dlgs:
        sdlg: List[np.ndarray] = []
        for _ in dlg:
            sdlg.append(sim_flat[idx])
            idx += 1
        sim_dlgs.append(sdlg)

    # Refit
    try:
        fit = fit_annotator_model(
            sim_dlgs, n_ann,
            a_init=a_true, b_init=b_true,
            max_iter=200, tol=1e-5, verbose=False,
        )
        return {
            "rep": rep, "a_true": a_true, "b_true": b_true,
            "a_hat": fit["a"], "b_hat": fit["b"], "ok": True,
        }
    except Exception as exc:
        return {
            "rep": rep, "a_true": a_true, "b_true": b_true,
            "a_hat": float("nan"), "b_hat": float("nan"),
            "ok": False, "error": str(exc),
        }


def run_g3(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_fitted: float,
    b_fitted: float,
    confusion_fitted: np.ndarray,
    prior_fitted: np.ndarray,
    n_reps: int = G3_N_REPS,
    b_mults: List[float] = G3_B_MULTS,
    n_workers: int = MAX_WORKERS,
    ckpt_path: Optional[Path] = None,
) -> Dict:
    """
    Gate G3: parameter recovery check.
    """
    # Load checkpoint
    all_results: List[Dict] = []
    done_keys: set = set()
    if ckpt_path and ckpt_path.exists():
        try:
            ck = json.loads(ckpt_path.read_text())
            all_results = ck.get("g3_reps", [])
            done_keys = {(r["rep"], r["b_true"]) for r in all_results}
            log.info(f"G3 checkpoint: {len(done_keys)} reps done")
        except Exception:
            pass

    args_list = []
    for bm in b_mults:
        b_true = bm * b_fitted
        for rep in range(n_reps):
            if (rep, b_true) not in done_keys:
                args_list.append(
                    (rep, dlgs, n_ann, a_fitted, b_true,
                     confusion_fitted, prior_fitted)
                )

    if args_list:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for res in pool.imap_unordered(_g3_worker, args_list):
                all_results.append(res)
                if len(all_results) % 20 == 0:
                    log.info(f"G3: {len(all_results)} reps done")
                    if ckpt_path:
                        ckpt_path.write_text(
                            json.dumps({"g3_reps": all_results},
                                       ensure_ascii=False))

    # Analyse results
    ok_results = [r for r in all_results if r.get("ok", False)]
    true_b_all = np.array([r["b_true"] for r in ok_results])
    rec_b_all = np.array([r["b_hat"] for r in ok_results])

    corr, corr_p = pearsonr(true_b_all, rec_b_all) if len(ok_results) >= 3 else (float("nan"), float("nan"))

    per_setting: Dict = {}
    for bm in b_mults:
        b_true = bm * b_fitted
        reps_s = [r for r in ok_results
                  if abs(r["b_true"] - b_true) < 1e-9]
        b_hats = np.array([r["b_hat"] for r in reps_s])
        bias = float(np.mean(b_hats) - b_true)
        rel_bias = abs(bias) / b_true if b_true > 1e-9 else float("nan")
        per_setting[f"b_{bm}x"] = {
            "b_true": float(b_true),
            "b_hat_mean": float(np.mean(b_hats)),
            "b_hat_sd": float(np.std(b_hats, ddof=1)) if len(b_hats) > 1 else float("nan"),
            "bias": bias,
            "rel_bias": rel_bias,
            "n_ok": len(reps_s),
        }

    # G3 verdict
    max_rel_bias = max(
        per_setting[f"b_{bm}x"]["rel_bias"] for bm in b_mults
    )
    corr_ok = (not math.isnan(corr)) and corr > 0.8
    bias_ok = (not math.isnan(max_rel_bias)) and max_rel_bias < 0.20
    verdict = "PASS" if (corr_ok and bias_ok) else "FAIL"

    log.info(
        f"G3: corr={corr:.4f} (>0.8? {corr_ok}), "
        f"max_rel_bias={max_rel_bias:.4f} (<0.20? {bias_ok}) -> {verdict}"
    )

    return {
        "n_ok": len(ok_results),
        "n_total": len(all_results),
        "pearson_corr": float(corr),
        "pearson_p": float(corr_p),
        "corr_pass": corr_ok,
        "max_rel_bias": float(max_rel_bias),
        "bias_pass": bias_ok,
        "verdict": verdict,
        "per_setting": per_setting,
    }


# ============================================================================
# Section 7: Simulate from Fitted Model (PPC)
# ============================================================================

def simulate_votes_from_fitted(
    dlgs: List[List[np.ndarray]],
    n_ann: int,
    a_fit: float,
    b_fit: float,
    confusion_fit: np.ndarray,
    prior_fit: np.ndarray,
    seed: int,
) -> List[List[np.ndarray]]:
    """
    Simulate votes from fitted annotator model (posterior predictive).
    Uses MAP true categories (plurality) for each item.
    """
    rng = np.random.default_rng(seed)
    K_loc = confusion_fit.shape[0]
    n_items = sum(len(d) for d in dlgs)

    # Use plurality as MAP true category
    flat_true_cats = np.array(
        [int(v.argmax()) for d in dlgs for v in d], dtype=np.int64
    )

    # Sample u, compute q
    u_vals = rng.normal(0, 1, size=n_items)
    q_vals = _sigmoid(a_fit - b_fit * u_vals)

    sim_flat: List[np.ndarray] = []
    for idx in range(n_items):
        tc = int(flat_true_cats[idx])
        q = float(q_vals[idx])
        votes_sim = np.zeros(K_loc, dtype=np.int64)
        conf_row = confusion_fit[tc].copy()
        conf_row[tc] = 0.0
        rs = conf_row.sum()
        if rs > 1e-12:
            conf_row /= rs
        else:
            conf_row = np.ones(K_loc) / (K_loc - 1)
            conf_row[tc] = 0.0
            conf_row /= conf_row.sum()
        for _ in range(n_ann):
            if rng.random() < q:
                votes_sim[tc] += 1
            else:
                votes_sim[int(rng.choice(K_loc, p=conf_row))] += 1
        sim_flat.append(votes_sim)

    # Reconstruct
    idx = 0
    sim_dlgs: List[List[np.ndarray]] = []
    for dlg in dlgs:
        sdlg: List[np.ndarray] = []
        for _ in dlg:
            sdlg.append(sim_flat[idx])
            idx += 1
        sim_dlgs.append(sdlg)

    return sim_dlgs


# ============================================================================
# Section 8: Main
# ============================================================================

def _safe_float(x) -> float:
    if isinstance(x, (int, np.integer)):
        return float(x)
    if isinstance(x, (float, np.floating)):
        return float("nan") if math.isnan(x) else float(x)
    return x


def _json_safe(obj):
    """Recursively convert numpy types to Python types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    t0_total = time.time()
    log.info("=" * 68)
    log.info("E29: Annotator Model  (start)")
    log.info("=" * 68)

    # -- 1. Load Data ----------------------------------------------------------
    log.info("\n[1] Loading datasets ...")
    el_friends, el_ep = load_el_raw_split()
    m3ed_dlgs = load_m3ed_votes()

    datasets = {
        "el_friends":     (el_friends,  5),
        "el_emotionpush": (el_ep,       5),
        "m3ed":           (m3ed_dlgs,   3),
    }

    # -- 2. Simcheck: real statistics ------------------------------------------
    log.info("\n[2] Computing real vote statistics ...")
    simcheck: Dict = {}
    for dname, (dlgs, n_ann) in datasets.items():
        log.info(f"  {dname} ...")
        real_stats = compute_vote_stats(dlgs, n_ann, label="real")
        simcheck[dname] = {
            "n_ann": n_ann,
            "real": real_stats,
            "sim_indep": None,
            "sim_linked": None,
        }

    # -- 3. Simcheck: simulated copies -----------------------------------------
    log.info("\n[3] Simulating votes (20 copies x 2 models x 3 datasets) ...")
    for dname, (dlgs, n_ann) in datasets.items():
        log.info(f"  {dname}: building E27e-style simulator ...")
        confusion = empirical_confusion(dlgs, K)
        q_pool = empirical_q_pool_from_dlgs(dlgs, n_ann)
        H_pool = empirical_H_pool_from_dlgs(dlgs)

        for mode in ("indep", "linked"):
            log.info(f"    {dname} {mode} ...")
            copy_stats: List[Dict] = []
            for rep in range(N_SIM):
                sim_dlgs = simulate_votes_for_dlgs(
                    dlgs, n_ann, confusion, q_pool, H_pool,
                    mode=mode, seed=rep * 7 + _name_offset(dname),
                )
                cs = compute_vote_stats(sim_dlgs, n_ann, label=f"sim_{mode}_{rep}")
                copy_stats.append(cs)
            agg = aggregate_sim_stats(copy_stats)
            simcheck[dname][f"sim_{mode}"] = agg

    # Compute discrepancies (real vs sim_indep and real vs sim_linked)
    for dname in datasets:
        real = simcheck[dname]["real"]
        disc: Dict = {}
        for model in ("indep", "linked"):
            sim = simcheck[dname][f"sim_{model}"]
            d: Dict = {}
            # H_obs_mean difference
            d["H_obs_mean_diff"] = real["H_obs_mean"] - sim["H_obs_mean_mean"]
            d["H_obs_sd_diff"] = real["H_obs_sd"] - sim["H_obs_sd_mean"]
            d["consec_H_corr_diff"] = (
                real["consec_H_corr"] - sim["consec_H_corr_mean"]
            )
            d["mean_distinct_cats_diff"] = (
                real["mean_distinct_cats"] - sim["mean_distinct_cats_mean"]
            )
            # Plurality distribution differences
            pl_keys = list(real["plurality_dist"].keys())
            pl_diffs = {}
            for pk in pl_keys:
                sim_key = f"plurality_{pk}_mean"
                if sim_key in sim:
                    pl_diffs[pk] = real["plurality_dist"][pk] - sim[sim_key]
            d["plurality_diffs"] = pl_diffs
            disc[model] = d
        simcheck[dname]["discrepancies"] = disc

    simcheck_path = OUT_DIR / "E29_simcheck.json"
    simcheck_path.write_text(
        json.dumps(_json_safe(simcheck), ensure_ascii=False, indent=2))
    log.info(f"  Simcheck saved -> {simcheck_path}")

    # -- 4. Fit annotator model ------------------------------------------------
    log.info("\n[4] Fitting annotator model (EM) ...")
    fit_results: Dict = {}
    for dname, (dlgs, n_ann) in datasets.items():
        log.info(f"  Fitting {dname} (n_ann={n_ann}) ...")
        t_fit = time.time()
        fit = fit_annotator_model(dlgs, n_ann, verbose=True)
        fit["fit_time_s"] = round(time.time() - t_fit, 2)
        fit["dataset"] = dname
        log.info(f"    {dname}: a={fit['a']:.4f}, b={fit['b']:.4f}, "
                 f"logL={fit['logL']:.1f}, "
                 f"n_iter={fit['n_iter']}, t={fit['fit_time_s']}s")

        # Uniform-error variant
        log.info(f"  Fitting uniform-error variant for {dname} ...")
        fit_unif = fit_annotator_model_uniform_error(
            dlgs, n_ann, a_init=fit["a"], b_init=fit["b"],
            max_iter=300, tol=1e-6,
        )
        fit["uniform_error_variant"] = fit_unif
        log.info(f"    Uniform: a={fit_unif['a']:.4f}, b={fit_unif['b']:.4f}, "
                 f"logL={fit_unif['logL']:.1f}")

        # Bootstrap SEs
        log.info(f"  Bootstrap SEs for {dname} ({N_BOOT} reps) ...")
        ckpt_boot = OUT_DIR / f"E29_boot_{dname}_ckpt.json"
        boot = compute_bootstrap_ses(
            dlgs, n_ann, fit["a"], fit["b"],
            n_boot=N_BOOT, n_workers=MAX_WORKERS,
            ckpt_path=ckpt_boot,
        )
        fit["bootstrap"] = boot
        log.info(f"    a = {fit['a']:.4f} +/- {boot['a_se']:.4f}  "
                 f"b = {fit['b']:.4f} +/- {boot['b_se']:.4f}")
        fit_results[dname] = fit

    fit_path = OUT_DIR / "E29_fit.json"
    fit_path.write_text(
        json.dumps(_json_safe(fit_results), ensure_ascii=False, indent=2))
    log.info(f"  Fit results saved -> {fit_path}")

    # -- 5. Gate G3 (Friends) --------------------------------------------------
    log.info("\n[5] Gate G3: parameter recovery ...")
    g3_dname = "el_friends"
    g3_dlgs, g3_nann = datasets[g3_dname]
    g3_fit = fit_results[g3_dname]
    confusion_np = np.array(g3_fit["confusion"])
    prior_np = np.array(g3_fit["prior"])

    ckpt_g3 = OUT_DIR / "E29_g3_ckpt.json"
    g3 = run_g3(
        g3_dlgs, g3_nann,
        a_fitted=g3_fit["a"],
        b_fitted=g3_fit["b"],
        confusion_fitted=confusion_np,
        prior_fitted=prior_np,
        n_reps=G3_N_REPS,
        b_mults=G3_B_MULTS,
        n_workers=MAX_WORKERS,
        ckpt_path=ckpt_g3,
    )
    g3["source_dataset"] = g3_dname

    g3_path = OUT_DIR / "E29_g3.json"
    g3_path.write_text(json.dumps(_json_safe(g3), ensure_ascii=False, indent=2))
    log.info(f"  G3 saved -> {g3_path}")
    log.info(f"  G3 VERDICT: {g3['verdict']}")

    # -- 6. Posterior Predictive Check -----------------------------------------
    log.info("\n[6] Posterior predictive check (20 copies from fitted model) ...")
    ppc_results: Dict = {}
    for dname, (dlgs, n_ann) in datasets.items():
        fit_d = fit_results[dname]
        confusion_d = np.array(fit_d["confusion"])
        prior_d = np.array(fit_d["prior"])
        ppc_copies: List[Dict] = []
        for rep in range(N_SIM):
            sim_dlgs_ppc = simulate_votes_from_fitted(
                dlgs, n_ann,
                fit_d["a"], fit_d["b"],
                confusion_d, prior_d,
                seed=rep * 13 + _name_offset(dname),
            )
            cs = compute_vote_stats(sim_dlgs_ppc, n_ann, label=f"ppc_{rep}")
            ppc_copies.append(cs)
        ppc_results[dname] = aggregate_sim_stats(ppc_copies)

    ppc_path = OUT_DIR / "E29_ppc.json"
    ppc_path.write_text(
        json.dumps(_json_safe(ppc_results), ensure_ascii=False, indent=2))
    log.info(f"  PPC saved -> {ppc_path}")

    elapsed = time.time() - t0_total
    log.info(f"\n=== E29 complete ({elapsed:.1f}s) ===")


if __name__ == "__main__":
    main()
