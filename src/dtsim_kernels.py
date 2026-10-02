"""
numba kernels for the observation-pipeline simulators.

All kernels draw from numba's internal random state; seed it with _seed_nb() before use.
Accuracy models:
    independent  q ~ empirical pool of plurality / 5
    linked       q = q_sorted[rank(H)] + N(0, noise_std), clipped to [0.2, 1]; rank is the lower-bound
                 rank of H among the empirical entropies, q_sorted is the q pool in descending order.
Each annotator votes for the true category with probability q, otherwise draws from the confusion row.
The observed label is the plurality vote (random tie-break) and the observed H is the vote entropy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit

from estimator_dt import _dt_single_nll_grad


@njit(cache=True)
def _seed_nb(seed: int) -> None:
    """Seed numba's internal random state."""
    np.random.seed(seed)


@njit(cache=True)
def _sample_q_indep_nb(N_ev: int, q_pool: np.ndarray) -> np.ndarray:
    """Draw N_ev accuracies uniformly (with replacement) from q_pool."""
    n_q = len(q_pool)
    q_vals = np.empty(N_ev, dtype=np.float64)
    for i in range(N_ev):
        idx = int(np.random.random() * n_q)
        if idx >= n_q:
            idx = n_q - 1
        q_vals[i] = q_pool[idx]
    return q_vals


@njit(cache=True)
def _compute_q_linked_nb(
    H_vals: np.ndarray,
    H_sorted: np.ndarray,
    q_sorted: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    """Linked accuracies: rank-matched, decreasing in H, plus clipped Gaussian noise."""
    N = len(H_vals)
    n_h = len(H_sorted)
    n_q = len(q_sorted)
    q_vals = np.empty(N, dtype=np.float64)

    for i in range(N):
        # lower-bound binary search (np.searchsorted, side="left")
        lo, hi = 0, n_h
        while lo < hi:
            mid = (lo + hi) // 2
            if H_sorted[mid] < H_vals[i]:
                lo = mid + 1
            else:
                hi = mid
        rank_frac = lo / n_h
        q_idx = int(rank_frac * n_q)
        if q_idx >= n_q:
            q_idx = n_q - 1
        q_base = q_sorted[q_idx]

        q_noise = np.random.normal(0.0, noise_std)
        q_m = q_base + q_noise
        if q_m < 0.2:
            q_m = 0.2
        if q_m > 1.0:
            q_m = 1.0
        q_vals[i] = q_m

    return q_vals


@njit(cache=True)
def _simulate_votes_all_events_nb(
    all_true_cats: np.ndarray,
    all_q_vals: np.ndarray,
    confusion: np.ndarray,
    K: int,
    n_ann: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Vote simulation for all events.
    Returns (obs_cats, H_obs, plurality, flip_count), where flip_count counts obs_cat != true_cat.
    """
    N = len(all_true_cats)
    obs_cats = np.empty(N, dtype=np.int64)
    H_obs_arr = np.empty(N, dtype=np.float64)
    plur_arr = np.empty(N, dtype=np.int64)
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

        max_v = 0.0
        for j in range(K):
            if votes[j] > max_v:
                max_v = votes[j]

        n_tied = 0
        for j in range(K):
            if votes[j] == max_v:
                n_tied += 1

        # random tie-break (one uniform draw per event)
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

        total = votes.sum()
        h_val = 0.0
        for j in range(K):
            p_j = votes[j] / total
            if p_j > 1e-12:
                h_val -= p_j * np.log(p_j + 1e-12)
        H_obs_arr[idx] = h_val
        plur_arr[idx] = int(max_v)

    return obs_cats, H_obs_arr, plur_arr, flip_count


@njit(cache=True)
def _generate_cats_from_H_nb(
    dlg_len: int,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma_inj: float,
    H_cov: np.ndarray,
    K: int,
) -> np.ndarray:
    """Sequential DT-AMHP category generation with modifier H_cov."""
    cats = np.empty(dlg_len, dtype=np.int64)
    R = np.zeros(K, dtype=np.float64)
    ef = np.exp(-beta)

    for m in range(dlg_len):
        Lam = mu.copy()
        for j in range(K):
            for i in range(K):
                Lam[i] += alpha[i, j] * R[j]

        for j in range(K):
            if Lam[j] < 1e-300:
                Lam[j] = 1e-300

        total_lam = 0.0
        for j in range(K):
            total_lam += Lam[j]

        r = np.random.random()
        cum = 0.0
        cat = K - 1
        for j in range(K):
            cum += Lam[j] / total_lam
            if r <= cum:
                cat = j
                break
        cats[m] = cat

        clip_val = gamma_inj * H_cov[m]
        if clip_val > 30.0:
            clip_val = 30.0
        elif clip_val < -30.0:
            clip_val = -30.0
        gain = np.exp(clip_val)

        for j in range(K):
            R[j] = ef * R[j]
        R[cat] += ef * gain

    return cats


@njit(cache=True)
def _compute_within_cell_H_nb(
    cats: np.ndarray,
    H_raw: np.ndarray,
    K: int,
) -> np.ndarray:
    """Within-cell residuals (size-1 cells set to 0), then globally re-centred."""
    L = len(cats)
    H_wc = H_raw.copy()

    for c in range(K):
        count = 0
        total_h = 0.0
        for m in range(L):
            if cats[m] == c:
                count += 1
                total_h += H_raw[m]

        if count > 1:
            mean_c = total_h / count
            for m in range(L):
                if cats[m] == c:
                    H_wc[m] -= mean_c
        elif count == 1:
            for m in range(L):
                if cats[m] == c:
                    H_wc[m] = 0.0
                    break

    total = 0.0
    for m in range(L):
        total += H_wc[m]
    mean_g = total / L
    for m in range(L):
        H_wc[m] -= mean_g

    return H_wc


@njit(cache=True)
def _process_one_dlg_scheme_c_nb(
    dlg_len: int,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma_inj: float,
    H_dist: np.ndarray,
    H_gm: float,
    H_sorted: np.ndarray,
    q_sorted: np.ndarray,
    confusion: np.ndarray,
    q_noise_std: float,
    n_refine: int,
    K: int,
    n_ann: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray,
]:
    """
    One dialogue of the Scheme C' generator: latent H drawn from H_dist, categories generated with
    modifier H (pass 0: globally centred H; then n_refine passes with the within-cell residual of H),
    votes with linked accuracy.
    Returns (oracle_cats, oracle_H, oracle_plur, obs_cats, obs_H, obs_plur, H_raw).
    """
    n_h = len(H_dist)
    n_q = len(q_sorted)
    n_hs = len(H_sorted)

    H_raw = np.empty(dlg_len, dtype=np.float64)
    for m in range(dlg_len):
        idx = int(np.random.random() * n_h)
        if idx >= n_h:
            idx = n_h - 1
        H_raw[m] = H_dist[idx]

    H_cov = np.empty(dlg_len, dtype=np.float64)
    for m in range(dlg_len):
        H_cov[m] = H_raw[m] - H_gm
    cats = _generate_cats_from_H_nb(dlg_len, mu, alpha, beta, gamma_inj, H_cov, K)

    for _ in range(n_refine):
        H_cov = _compute_within_cell_H_nb(cats, H_raw, K)
        cats = _generate_cats_from_H_nb(dlg_len, mu, alpha, beta, gamma_inj, H_cov, K)

    oracle_cats = np.empty(dlg_len, dtype=np.int64)
    oracle_H = np.empty(dlg_len, dtype=np.float64)
    oracle_plur = np.empty(dlg_len, dtype=np.int64)
    obs_cats = np.empty(dlg_len, dtype=np.int64)
    obs_H = np.empty(dlg_len, dtype=np.float64)
    obs_plur = np.empty(dlg_len, dtype=np.int64)

    for m in range(dlg_len):
        tc = cats[m]

        lo, hi = 0, n_hs
        while lo < hi:
            mid = (lo + hi) // 2
            if H_sorted[mid] < H_raw[m]:
                lo = mid + 1
            else:
                hi = mid
        rank_frac = lo / n_hs
        q_idx = int(rank_frac * n_q)
        if q_idx >= n_q:
            q_idx = n_q - 1
        q_base = q_sorted[q_idx]
        q_noise = np.random.normal(0.0, q_noise_std)
        q_m = q_base + q_noise
        if q_m < 0.2:
            q_m = 0.2
        if q_m > 1.0:
            q_m = 1.0

        votes = np.zeros(K, dtype=np.float64)
        for _ in range(n_ann):
            if np.random.random() < q_m:
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

        max_v = 0.0
        for j in range(K):
            if votes[j] > max_v:
                max_v = votes[j]

        n_tied = 0
        for j in range(K):
            if votes[j] == max_v:
                n_tied += 1

        r3 = np.random.random()
        obs_cat_m = 0
        cum_p = 0.0
        for j in range(K):
            if votes[j] == max_v:
                cum_p += 1.0 / n_tied
                if r3 <= cum_p:
                    obs_cat_m = j
                    break

        total = votes.sum()
        h_val = 0.0
        for j in range(K):
            p_j = votes[j] / total
            if p_j > 1e-12:
                h_val -= p_j * np.log(p_j + 1e-12)

        oracle_cats[m] = tc
        oracle_H[m] = H_raw[m]
        oracle_plur[m] = int(max_v)

        obs_cats[m] = obs_cat_m
        obs_H[m] = h_val
        obs_plur[m] = int(max_v)

    return oracle_cats, oracle_H, oracle_plur, obs_cats, obs_H, obs_plur, H_raw


def _warmup_numba(K: int = 7) -> None:
    """Compile all kernels on small synthetic inputs."""
    _dt_single_nll_grad(
        np.array([0, 1, 2, 0, 1], dtype=np.int64),
        np.array([0.1, -0.2, 0.05, -0.1, 0.3]),
        np.ones(K) / K, np.ones((K, K)) * 0.05, 1.0, -0.3, K,
    )

    _seed_nb(42)
    dummy_cats = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    dummy_H = np.array([0.3, 0.8, 0.1, 0.6, 0.4])
    dummy_q = np.array([0.7, 0.8, 0.6, 0.7, 0.7])
    dummy_conf = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dummy_conf[i, j] = 0.0 if i == j else 1.0 / (K - 1)
    dummy_pool = np.array([0.6, 0.7, 0.8, 0.9])
    H_sorted_d = np.sort(dummy_H)
    q_sorted_d = np.sort(dummy_pool)[::-1]
    H_dist_d = dummy_H.copy()

    _sample_q_indep_nb(5, dummy_pool)
    _compute_q_linked_nb(dummy_H, H_sorted_d, q_sorted_d, 0.05)
    _simulate_votes_all_events_nb(dummy_cats, dummy_q, dummy_conf, K, 5)
    _generate_cats_from_H_nb(
        5, np.ones(K) / K, np.ones((K, K)) * 0.05, 1.0, -0.4,
        np.array([0.1, -0.2, 0.05, -0.1, 0.3]), K,
    )
    _compute_within_cell_H_nb(dummy_cats, dummy_H, K)
    _process_one_dlg_scheme_c_nb(
        5, np.ones(K) / K, np.ones((K, K)) * 0.05, 1.0, -0.4,
        H_dist_d, float(H_dist_d.mean()), H_sorted_d, q_sorted_d,
        dummy_conf, 0.05, 1, K, 5,
    )
