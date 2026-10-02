"""
Latent-mark correction and the matched data-generating process of experiments E34e, E34f and E37.

Matched DGP: for each event a latent difficulty u ~ N(0, 1) drives both the excitation modifier of the
DT-AMHP (pass 0: u; then N_REFINE passes with the within-cell residual of u) and the annotator accuracy
q = sigmoid(a - b*u) of the i.i.d. annotator model (E29, Friends parameters used for EL-wide). Five
annotators vote; the observed label is the plurality vote (random tie-break) and the observed modifier is
the vote entropy.

Latent-mark correction: the history uses the posterior expected marks P(c | votes) and the current event is
marginalised with the annotator likelihood P(votes | c), both under the annotator model with known
parameters (Gauss-Hermite integration over u).
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import roots_hermite

import config
from dtsim_core import (  # noqa: F401
    K, load_el_raw, residualize_within_cell, global_H_mean, prepare_dt_dlgs, fit_dt_bounded,
)
from dtsim_fits import N_REFINE, fit_within_cell  # noqa: F401
from dtsim_kernels import (  # noqa: F401
    _seed_nb, _warmup_numba, _generate_cats_from_H_nb, _compute_within_cell_H_nb,
)
from estimator_dt import unpack_v_dt  # noqa: F401

N_GH = 20
N_ANN_EL = 5
N_ANN_M3D = 3
K_EL = K
DT_L1 = 0.001
GAMMA_BOUND = 5.0
GAMMA_CONV = 4.5
N_RESTARTS = 3
MAXITER = 3000

# Gauss-Hermite quadrature for u ~ N(0, 1)
_xgh, _wgh = roots_hermite(N_GH)
U_GH: np.ndarray = np.sqrt(2) * _xgh
W_GH: np.ndarray = _wgh / np.sqrt(np.pi)
W_GH = W_GH / W_GH.sum()


# ---------------------------------------------------------------------------
# Corrected likelihood
# ---------------------------------------------------------------------------

@njit(cache=True)
def _ds_nll_grad_fixed_c(
    post: np.ndarray,
    log_lik: np.ndarray,
    Hs_c: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma: float,
    K: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    Corrected NLL and gradients for one dialogue.
    post (M, K): P(c | votes) used as history marks; log_lik (M, K): log P(votes | c) for the current event.
    Lambda_c(m) = mu_c + sum_j alpha[c,j] * R_j(m).
    """
    M  = len(Hs_c)
    ef = np.exp(-beta)
    g_mu    = np.zeros(K)
    g_alpha = np.zeros((K, K))
    g_beta  = 0.0
    g_gamma = 0.0
    R = np.zeros(K)
    P = np.zeros(K)
    Q = np.zeros(K)
    nll = 0.0

    for m in range(M):
        gain_m = np.exp(gamma * Hs_c[m])

        lam = np.zeros(K)
        for c in range(K):
            lam[c] = mu[c]
            for j in range(K):
                lam[c] += alpha[c, j] * R[j]
            if lam[c] < 1e-300:
                lam[c] = 1e-300

        S_m = 0.0
        for c in range(K):
            S_m += lam[c]

        max_ll = log_lik[m, 0]
        for c in range(K):
            if log_lik[m, c] > max_ll:
                max_ll = log_lik[m, c]
        lik = np.zeros(K)
        W_u = 0.0
        for c in range(K):
            lik[c] = np.exp(log_lik[m, c] - max_ll)
            W_u   += lam[c] * lik[c]

        nll += np.log(S_m) - (np.log(max(W_u, 1e-300)) + max_ll)

        inv_S = 1.0 / S_m
        inv_W = 1.0 / max(W_u, 1e-300)
        for c in range(K):
            w_mc = inv_S - lik[c] * inv_W
            g_mu[c] += w_mc
            for j in range(K):
                g_alpha[c, j] += R[j] * w_mc
            for j in range(K):
                g_beta  += w_mc * alpha[c, j] * (-P[j])
                g_gamma += w_mc * alpha[c, j] * Q[j]

        R_new = np.zeros(K)
        P_new = np.zeros(K)
        Q_new = np.zeros(K)
        for j in range(K):
            R_new[j] = ef * (R[j] + post[m, j] * gain_m)
            P_new[j] = ef * P[j] + R_new[j]
            Q_new[j] = ef * (Q[j] + Hs_c[m] * post[m, j] * gain_m)
        R = R_new; P = P_new; Q = Q_new

    return nll, g_mu, g_alpha, g_beta, g_gamma


@njit(cache=True)
def _compute_ds_posterior_nb(
    votes: np.ndarray,        # (K,) vote counts
    a: float, b: float,
    confusion: np.ndarray,    # (K, K)
    prior: np.ndarray,        # (K,)
    n_ann: int,
    K: int,
    U_gh: np.ndarray,
    W_gh: np.ndarray,
    N_gh: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Annotator-model (Dawid-Skene with difficulty) posterior for one event.
    Returns (post, log_lik), each of shape (K,):
    post[c] = P(cat = c | votes), log_lik[c] = log P(votes | cat = c) marginalised over u.
    """
    # For each true category c, compute log P(votes | c, u) integrated over u
    log_marg = np.zeros(K)
    for c in range(K):
        acc = 0.0
        for g in range(N_gh):
            u = U_gh[g]; w = W_gh[g]
            logit_q = a - b * u
            q = 1.0 / (1.0 + np.exp(-logit_q))
            # p_correct = q; p_wrong[k] = (1-q) * confusion[c,k]  (k!=c)
            log_p = 0.0
            for k in range(K):
                if votes[k] < 0.5:
                    continue
                n_k = int(votes[k] + 0.5)
                if k == c:
                    p_k = q
                else:
                    p_k = (1.0 - q) * confusion[c, k]
                if p_k < 1e-300:
                    p_k = 1e-300
                log_p += n_k * np.log(p_k)
            acc += w * np.exp(log_p)
        log_marg[c] = np.log(max(acc, 1e-300))

    # Posterior: prior * likelihood
    log_post_unnorm = log_marg + np.log(np.maximum(prior, 1e-300))
    log_max = log_post_unnorm.max()
    post_unnorm = np.exp(log_post_unnorm - log_max)
    post = post_unnorm / max(post_unnorm.sum(), 1e-300)
    return post, log_marg


def _softmax(v: np.ndarray) -> np.ndarray:
    v = v - v.max(); e = np.exp(v); return e / e.sum()


def _softmax_inv(mu: np.ndarray) -> np.ndarray:
    v = np.log(np.maximum(mu, 1e-9)); return v - v.mean()


def _softplus(v): return np.where(v > 20, v, np.log1p(np.exp(np.clip(v,-500,20))))


def _softplus_inv(x): return np.where(x>20, x, np.log(np.expm1(np.maximum(x, 1e-9))))


def _sigmoid(v): return 1.0 / (1.0 + np.exp(-np.clip(v, -30, 30)))


def _n_params(): return K + K * K + 2


def _unpack_v(v):
    mu    = _softmax(v[:K])
    alpha = _softplus(v[K:K+K*K]).reshape(K, K)
    beta  = float(np.exp(v[K+K*K]))
    gamma = float(v[K+K*K+1])
    return mu, alpha, beta, gamma


def _pack_v(mu, alpha, beta, gamma):
    return np.concatenate([
        _softmax_inv(np.maximum(mu, 1e-9)),
        _softplus_inv(np.maximum(alpha.ravel(), 1e-6)),
        [np.log(max(beta, 1e-6))],
        [gamma],
    ])


def _total_obj_c(v, dlg_data):
    mu, alpha, beta, gamma = _unpack_v(v)
    g_mu_n  = np.zeros(K)
    g_a_n   = np.zeros((K, K))
    g_b_n   = 0.0
    g_g     = 0.0
    tot     = DT_L1 * float(alpha.sum())
    for post, log_lik, Hs_c in dlg_data:
        nll_d, gm, ga, gb, gg = _ds_nll_grad_fixed_c(
            post, log_lik, Hs_c, mu, alpha, beta, gamma, K)
        tot += nll_d; g_mu_n += gm; g_a_n += ga; g_b_n += gb; g_g += gg
    g_a_n += DT_L1 * np.ones((K, K))
    grad_v = np.zeros_like(v)
    mu_dot = float(np.dot(mu, g_mu_n))
    grad_v[:K] = mu * (g_mu_n - mu_dot)
    sig_a = _sigmoid(v[K:K+K*K])
    grad_v[K:K+K*K] = g_a_n.ravel() * sig_a
    grad_v[K+K*K]   = g_b_n * beta
    grad_v[K+K*K+1] = g_g
    return tot, grad_v


def fit_ds_fixed_c(dlg_data, seed):
    """Fit the latent-mark corrected estimator (multi-restart L-BFGS-B, gamma bounded)."""
    n_p = _n_params()
    bounds = [(None, None)] * n_p
    bounds[K+K*K]   = (np.log(0.05), np.log(5.0))
    bounds[K+K*K+1] = (-GAMMA_BOUND, GAMMA_BOUND)
    rng = np.random.default_rng(seed)
    best_f = np.inf; best_x = None; best_ok = False
    for _ in range(N_RESTARTS):
        v0 = np.zeros(n_p)
        v0[:K]       = rng.uniform(-0.5, 0.5, K)
        v0[K:K+K*K]  = rng.uniform(-2.0, -0.5, K * K)
        v0[K+K*K]    = rng.uniform(np.log(0.3), np.log(1.5))
        v0[K+K*K+1]  = rng.uniform(-0.5, 0.5)
        try:
            res = minimize(
                _total_obj_c, v0, args=(dlg_data,), method="L-BFGS-B",
                jac=True, bounds=bounds,
                options={"maxiter": MAXITER, "ftol": 1e-12, "gtol": 1e-7},
            )
            if res.fun < best_f:
                best_f = res.fun; best_x = res.x.copy(); best_ok = bool(res.success)
        except Exception:
            pass
    if best_x is None:
        return {"gamma_hat": float("nan"), "converged": False, "at_bound": False}
    _, _, _, gamma_hat = _unpack_v(best_x)
    at_bound  = abs(gamma_hat) >= GAMMA_CONV
    converged = not at_bound and best_ok
    return {"gamma_hat": float(gamma_hat), "converged": bool(converged), "at_bound": bool(at_bound)}


def _within_cell_H(cats: np.ndarray, H_raw: np.ndarray) -> np.ndarray:
    H_wc = H_raw.copy()
    for c in range(K):
        idx = np.where(cats == c)[0]
        if len(idx) > 1:
            H_wc[idx] -= H_raw[idx].mean()
        elif len(idx) == 1:
            H_wc[idx[0]] = 0.0
    return H_wc


# ---------------------------------------------------------------------------
# Annotator parameters, simulated events and the two estimators
# ---------------------------------------------------------------------------

def load_annotator_params() -> Dict:
    """i.i.d. annotator-model parameters used for EL-wide (E29 fit on Friends)."""
    e29 = json.loads((config.RESULTS_ROOT / "E29" / "E29_fit.json").read_text())["el_friends"]
    return {"a": e29["a"], "b": e29["b"],
            "confusion": np.array(e29["confusion"]),
            "prior": np.array(e29["prior"])}


def _votes_to_event(votes: np.ndarray, rng: np.random.Generator) -> dict:
    """Event dict from a vote vector (plurality label with random tie-break, H = vote entropy)."""
    p     = votes / votes.sum()
    H_obs = float(-(p * np.log(p + 1e-12)).sum())
    max_v = votes.max()
    tied  = np.where(votes == max_v)[0]
    obs_cat = int(rng.choice(tied))  # random tie-break
    return {
        "votes": votes.copy(), "p": p, "H": H_obs,
        "cat": obs_cat, "plurality": int(max_v),
    }


def _build_correction_dlg_data(
    dlgs: List[List[dict]],
    a: float, b: float,
    confusion: np.ndarray, prior: np.ndarray,
    n_ann: int,
) -> List[Tuple]:
    """
    Inputs (post, log_lik, Hs_c) of the corrected estimator; the modifier is the observed vote entropy,
    residualised within observed cells and globally centred.
    """
    all_Hs_wc = []
    valid_dlgs = []

    for dlg in dlgs:
        filt = [e for e in dlg if len(dlg) >= 2]
        if len(filt) < 2:
            continue
        cats  = np.array([e["cat"] for e in filt], dtype=np.int64)
        H_obs = np.array([e["H"]   for e in filt], dtype=np.float64)
        H_wc  = _within_cell_H(cats, H_obs)
        all_Hs_wc.append(H_wc)
        valid_dlgs.append(filt)

    if not all_Hs_wc:
        return []

    H_bar = float(np.concatenate(all_Hs_wc).mean())
    confusion_nb = np.ascontiguousarray(confusion, dtype=np.float64)
    prior_nb     = np.ascontiguousarray(prior, dtype=np.float64)

    dlg_data = []
    for i, filt in enumerate(valid_dlgs):
        Hs_c = all_Hs_wc[i] - H_bar
        M    = len(filt)
        post_arr    = np.zeros((M, K_EL), dtype=np.float64)
        log_lik_arr = np.zeros((M, K_EL), dtype=np.float64)
        for m, ev in enumerate(filt):
            votes_m = np.ascontiguousarray(ev["votes"], dtype=np.float64)
            pm, llm = _compute_ds_posterior_nb(
                votes_m, a, b, confusion_nb, prior_nb,
                n_ann, K_EL, U_GH, W_GH, N_GH)
            post_arr[m]    = pm
            log_lik_arr[m] = llm
        dlg_data.append((post_arr, log_lik_arr, Hs_c.astype(np.float64)))

    return dlg_data


def _fit_both(
    rep: int, dlgs: List[List[dict]],
    n_ann: int, ann_iid: dict,
    label: str, gamma_inj: float,
) -> dict:
    """Fit the hard-mark and the latent-mark corrected estimators (modifier = vote entropy for both)."""
    result = {"rep": rep, "label": label, "gamma_inj": gamma_inj}

    filt = [d for d in dlgs if len(d) >= 2]
    if len(filt) >= 5:
        try:
            r = fit_within_cell(filt, seed=rep)
            result["hard"] = {"gamma_hat": r["gamma_hat"],
                              "converged": r["converged"],
                              "at_bound":  r.get("at_bound", False)}
        except Exception as ex:
            result["hard"] = {"gamma_hat": float("nan"), "converged": False, "err": str(ex)}
    else:
        result["hard"] = {"gamma_hat": float("nan"), "converged": False, "err": "too few dlgs"}

    try:
        dd = _build_correction_dlg_data(
            dlgs, ann_iid["a"], ann_iid["b"],
            ann_iid["confusion"], ann_iid["prior"], n_ann)
        if len(dd) >= 5:
            r = fit_ds_fixed_c(dd, seed=rep + 5000)
            result["ds_corr"] = {"gamma_hat": r["gamma_hat"],
                                 "converged": r["converged"],
                                 "at_bound":  r["at_bound"]}
        else:
            result["ds_corr"] = {"gamma_hat": float("nan"), "converged": False, "err": "too few dlgs"}
    except Exception as ex:
        result["ds_corr"] = {"gamma_hat": float("nan"), "converged": False, "err": str(ex)}

    return result



def simulate_matched(rep, gamma, dlg_lens, mu, alpha, beta, ann):
    """
    One data set of the matched DGP. Events carry the votes, the observed label and entropy,
    the true category ('true_cat') and the latent difficulty ('u').
    """
    rng = np.random.default_rng(rep * 104729 + int(abs(gamma) * 1000) + 7)
    _seed_nb(rep * 130363 + int(abs(gamma) * 1000) + 11)
    a, b = ann["a"], ann["b"]
    conf = np.asarray(ann["confusion"], dtype=np.float64)
    out = []
    for dl in dlg_lens:
        if dl < 2:
            continue
        u = rng.normal(size=dl)
        h_cov = u.copy()
        cats = _generate_cats_from_H_nb(dl, mu, alpha, beta, gamma, h_cov.astype(np.float64), K_EL)
        for _ in range(N_REFINE):
            h_cov = _compute_within_cell_H_nb(cats, u.astype(np.float64), K_EL)
            cats = _generate_cats_from_H_nb(dl, mu, alpha, beta, gamma, h_cov, K_EL)
        evs = []
        for m in range(dl):
            tc = int(cats[m])
            row = conf[tc].copy()
            row[tc] = 0.0
            row = row / row.sum() if row.sum() > 1e-12 else np.where(np.arange(K_EL) == tc, 0.0, 1.0 / (K_EL - 1))
            q = 1.0 / (1.0 + np.exp(-(a - b * u[m])))
            votes = np.zeros(K_EL)
            for _ in range(N_ANN_EL):
                if rng.random() < q:
                    votes[tc] += 1.0
                else:
                    votes[rng.choice(K_EL, p=row)] += 1.0
            ev = _votes_to_event(votes, rng)
            ev["true_cat"] = tc
            ev["u"] = float(u[m])
            evs.append(ev)
        out.append(evs)
    return out


def fit_el_wide_generator(n_restarts: int = 3):
    """DT-AMHP fitted to EL-wide (within-cell, hard-mark): returns (dialogue lengths, mu, alpha, beta)."""
    raw = load_el_raw()
    lens = [len(x) for x in raw]
    raw_dt = [{"cats": np.array([q["cat"] for q in x], dtype=np.int64),
               "Hs_raw": np.array([q["H"] for q in x])} for x in raw]
    dr = residualize_within_cell(raw_dt)
    hb = global_H_mean(dr)
    fit = fit_dt_bounded(prepare_dt_dlgs(dr, hb), hb, n_restarts=n_restarts, seed=0)
    mu, alpha, beta, _ = unpack_v_dt(np.array(fit["v_hat"]), K_EL)
    return lens, mu, alpha, float(beta)
