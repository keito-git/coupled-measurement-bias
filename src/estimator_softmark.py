"""
Soft-mark estimator.

The excitation contributed by event k uses its vote distribution p_k instead of its majority label:
alpha[i, argmax_k] is replaced by sum_j p_kj * alpha[i, j]. The target of the likelihood term is the
majority label. Events are placed on the turn index (times 0, 1, 2, ...).

Model:
    lambda_i(t) = mu_i + beta * sum_j alpha[i,j] * S_j(t)
    S_j(t)      = sum_{k: t_k < t} gain_k * p_kj * exp(-beta * (t - t_k)),   gain_k = exp(gamma * H_tilde_k)

Running sums:
    S[j]       = sum_{k<m} gain_k * p_kj * exp(-beta*Delta)
    P_s[j]     = sum_{k<m} gain_k * p_kj * Delta * exp(-beta*Delta)
    S_gamma[j] = sum_{k<m} Ht_k * gain_k * p_kj * beta * exp(-beta*Delta)

Unconstrained parameter vector (length K + K^2 + 2 + K): softplus(mu), softplus(alpha), log(beta), gamma,
softplus(b); b (shared-shock loadings) is not used by the likelihood below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus log(1 + exp(x))."""
    return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    """Inverse softplus: log(exp(y) - 1)."""
    return np.where(y > 30, y, np.log(np.expm1(np.clip(y, 1e-6, None))))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (derivative of softplus)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def pack_v(mu: np.ndarray, alpha: np.ndarray, beta: float,
           gamma_s: float, b: np.ndarray) -> np.ndarray:
    """Pack constrained parameters into the unconstrained vector."""
    return np.concatenate([
        _softplus_inv(np.maximum(mu, 1e-6)),
        _softplus_inv(np.maximum(alpha.ravel(), 1e-6)),
        [np.log(max(beta, 1e-6))],
        [gamma_s],
        _softplus_inv(np.maximum(b, 1e-6)),
    ])


def unpack_v(v: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Unpack the unconstrained vector to (mu, alpha, beta, gamma, b)."""
    idx = 0
    mu = _softplus(v[idx:idx + K]); idx += K
    alpha = _softplus(v[idx:idx + K * K]).reshape(K, K); idx += K * K
    beta = float(np.exp(v[idx])); idx += 1
    gamma_s = float(v[idx]); idx += 1
    b = _softplus(v[idx:idx + K])
    return mu, alpha, beta, gamma_s, b


def n_params(K: int) -> int:
    return K + K * K + 1 + 1 + K


@dataclass
class SoftMarkResult:
    mu_hat: np.ndarray
    alpha_hat: np.ndarray
    beta_hat: float
    gamma_hat: float
    b_hat: np.ndarray
    neg_loglik: float
    n_events: int
    n_threads: int
    success: bool
    message: str
    H_bar: float
    K: int


# ---------------------------------------------------------------------------
# numba kernels: NLL and natural-parameter gradients
# ---------------------------------------------------------------------------

@njit(cache=True)
def _soft_mark_nll_and_grad(
    times: np.ndarray,      # (N,) sorted event times
    cats: np.ndarray,       # (N,) int64 majority labels (likelihood targets)
    Hs_c: np.ndarray,       # (N,) centred modifier
    p_dist: np.ndarray,     # (N, K) vote distributions
    T: float,
    mu: np.ndarray,         # (K,)
    alpha: np.ndarray,      # (K, K)
    beta: float,
    gamma_s: float,
    K: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Return (nll, g_mu, g_alpha, g_beta, g_gamma, g_b) for one thread; L1 not included."""
    N = len(times)

    g_mu = np.zeros(K)
    g_alpha = np.zeros((K, K))
    g_beta = 0.0
    g_gamma = 0.0
    g_b = np.zeros(K)

    # baseline compensator
    comp = 0.0
    for i in range(K):
        comp += mu[i] * T
        g_mu[i] += T

    if N == 0:
        return comp, g_mu, g_alpha, g_beta, g_gamma, g_b

    S = np.zeros(K)
    P_s = np.zeros(K)
    S_gamma = np.zeros(K)

    prev_t = 0.0
    ll = 0.0

    for k in range(N):
        dt = times[k] - prev_t
        ef = math.exp(-beta * dt)

        for j in range(K):
            P_s[j] = ef * (P_s[j] + dt * S[j])
            S[j] *= ef
            S_gamma[j] *= ef

        ik = cats[k]
        R_ik = 0.0
        for j in range(K):
            R_ik += alpha[ik, j] * S[j]
        R_ik *= beta

        lam = mu[ik] + R_ik
        if lam < 1e-300:
            lam = 1e-300
        ll += math.log(lam)

        inv_lam = 1.0 / lam

        g_mu[ik] -= inv_lam
        for j in range(K):
            g_alpha[ik, j] -= beta * S[j] * inv_lam

        dR_dbeta = 0.0
        for j in range(K):
            dR_dbeta += alpha[ik, j] * (S[j] - beta * P_s[j])
        g_beta -= dR_dbeta * inv_lam

        # S_gamma already carries the beta factor
        dR_dgamma = 0.0
        for j in range(K):
            dR_dgamma += alpha[ik, j] * S_gamma[j]
        g_gamma -= dR_dgamma * inv_lam

        gs_h = gamma_s * Hs_c[k]
        if gs_h > 30.0:
            gs_h = 30.0
        elif gs_h < -30.0:
            gs_h = -30.0
        gain = math.exp(gs_h)

        for j in range(K):
            S[j] += gain * p_dist[k, j]
            S_gamma[j] += Hs_c[k] * gain * p_dist[k, j] * beta

        prev_t = times[k]

    # excitation compensator and its gradient
    alpha_colsum = np.zeros(K)
    for i in range(K):
        for j in range(K):
            alpha_colsum[j] += alpha[i, j]

    for k in range(N):
        gs_h = gamma_s * Hs_c[k]
        if gs_h > 30.0:
            gs_h = 30.0
        elif gs_h < -30.0:
            gs_h = -30.0
        gain = math.exp(gs_h)
        rem_t = T - times[k]
        exp_rem = math.exp(-beta * rem_t)
        remain = 1.0 - exp_rem

        for i in range(K):
            for j in range(K):
                term = gain * p_dist[k, j] * alpha[i, j] * remain
                comp += term
                g_alpha[i, j] += gain * p_dist[k, j] * remain

        soft_colsum = 0.0
        for j in range(K):
            soft_colsum += p_dist[k, j] * alpha_colsum[j]

        g_beta += gain * soft_colsum * rem_t * exp_rem
        g_gamma += Hs_c[k] * gain * soft_colsum * remain

    nll = -(ll - comp)
    return nll, g_mu, g_alpha, g_beta, g_gamma, g_b


@njit(cache=True)
def _soft_mark_total_nll_and_grad(
    times_all: np.ndarray,
    cats_all: np.ndarray,
    Hs_c_all: np.ndarray,
    p_dist_all: np.ndarray,     # (N_total, K)
    thread_starts: np.ndarray,  # (M + 1,)
    T_threads: np.ndarray,      # (M,)
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma_s: float,
    K: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, float, np.ndarray]:
    M = len(T_threads)
    total = 0.0
    g_mu = np.zeros(K)
    g_alpha = np.zeros((K, K))
    g_beta = 0.0
    g_gamma = 0.0
    g_b = np.zeros(K)

    for m in range(M):
        start = thread_starts[m]
        end = thread_starts[m + 1]
        nll_m, gmu_m, galpha_m, gbeta_m, ggamma_m, gb_m = _soft_mark_nll_and_grad(
            times_all[start:end],
            cats_all[start:end],
            Hs_c_all[start:end],
            p_dist_all[start:end, :],
            T_threads[m],
            mu, alpha, beta, gamma_s, K,
        )
        total += nll_m
        g_beta += gbeta_m
        g_gamma += ggamma_m
        for i in range(K):
            g_mu[i] += gmu_m[i]
            g_b[i] += gb_m[i]
            for j in range(K):
                g_alpha[i, j] += galpha_m[i, j]

    return total, g_mu, g_alpha, g_beta, g_gamma, g_b


def soft_mark_neg_loglik_and_grad(
    v: np.ndarray,
    K: int,
    l1_alpha: float,
    times_all: np.ndarray,
    cats_all: np.ndarray,
    Hs_c_all: np.ndarray,
    p_dist_all: np.ndarray,
    thread_starts: np.ndarray,
    T_threads: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """NLL + L1 penalty and its gradient w.r.t. the unconstrained vector."""
    mu, alpha, beta, gamma_s, b = unpack_v(v, K)

    nll, g_mu, g_alpha, g_beta_nat, g_gamma, g_b = _soft_mark_total_nll_and_grad(
        times_all, cats_all, Hs_c_all, p_dist_all,
        thread_starts, T_threads,
        mu, alpha, beta, gamma_s, K,
    )

    f = nll + l1_alpha * float(alpha.sum())
    g_alpha_total = g_alpha + l1_alpha

    idx_mu = slice(0, K)
    idx_alpha = slice(K, K + K * K)
    idx_beta = K + K * K
    idx_gamma = K + K * K + 1
    idx_b = slice(K + K * K + 2, K + K * K + 2 + K)

    sig_mu = _sigmoid(v[idx_mu])
    sig_alpha = _sigmoid(v[idx_alpha]).reshape(K, K)
    sig_b = _sigmoid(v[idx_b])

    grad_v = np.empty_like(v)
    grad_v[idx_mu] = g_mu * sig_mu
    grad_v[idx_alpha] = (g_alpha_total * sig_alpha).ravel()
    grad_v[idx_beta] = g_beta_nat * beta
    grad_v[idx_gamma] = g_gamma
    grad_v[idx_b] = g_b * sig_b

    return f, grad_v


def gradient_check_softmark(K: int = 4, n_threads: int = 6, l1_alpha: float = 0.001,
                            seed: int = 42, eps: float = 1e-5) -> dict:
    """Compare the analytic gradient with central finite differences on synthetic data."""
    rng = np.random.default_rng(seed)
    n_p = n_params(K)

    mu_true = rng.uniform(0.1, 0.5, K)
    alpha_true = rng.uniform(0.01, 0.15, (K, K))
    b_true = rng.uniform(0.05, 0.2, K)
    v0 = pack_v(mu_true, alpha_true, 1.2, -0.5, b_true)

    all_times, all_cats, all_Hs_c, all_p = [], [], [], []
    T_list, starts = [], [0]

    for _ in range(n_threads):
        n_m = int(rng.integers(5, 20))
        t = np.sort(rng.uniform(0.0, 10.0, n_m))
        t = (t - t[0]).astype(np.float64)
        c = rng.integers(0, K, n_m).astype(np.int64)
        H = rng.beta(2.0, 2.0, n_m).astype(np.float64)
        p = rng.dirichlet(np.ones(K), n_m).astype(np.float64)
        all_times.extend(t.tolist())
        all_cats.extend(c.tolist())
        all_Hs_c.extend((H - 0.5).tolist())
        all_p.append(p)
        T_list.append(float(t[-1]) + 1.0)
        starts.append(starts[-1] + n_m)

    kwargs = dict(K=K, l1_alpha=l1_alpha,
                  times_all=np.array(all_times, dtype=np.float64),
                  cats_all=np.array(all_cats, dtype=np.int64),
                  Hs_c_all=np.array(all_Hs_c, dtype=np.float64),
                  p_dist_all=np.vstack(all_p).astype(np.float64),
                  thread_starts=np.array(starts, dtype=np.int64),
                  T_threads=np.array(T_list, dtype=np.float64))

    f0, grad_analytic = soft_mark_neg_loglik_and_grad(v0, **kwargs)

    def obj(v):
        return soft_mark_neg_loglik_and_grad(v, **kwargs)[0]

    grad_numerical = np.zeros(n_p)
    for i in range(n_p):
        vp = v0.copy(); vp[i] += eps
        vm = v0.copy(); vm[i] -= eps
        grad_numerical[i] = (obj(vp) - obj(vm)) / (2.0 * eps)

    rel_err = np.abs(grad_analytic - grad_numerical) / (np.abs(grad_numerical) + 1e-10)
    max_rel = float(rel_err.max())
    return {
        "passed": bool(max_rel < 1e-4),
        "passed_strict": bool(max_rel < 1e-5),
        "max_rel_err": max_rel,
        "mean_rel_err": float(rel_err.mean()),
        "f0": float(f0),
        "K": K,
    }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class SoftMarkEstimator:
    """Maximum-likelihood fit of the soft-mark model."""

    def __init__(
        self,
        threads_data: List[dict],  # each dict: times_h, cats, Hs_c, p_dist, T
        K: int,
        H_bar: float = 0.0,
        l1_alpha: float = 0.001,
    ):
        self.K = K
        self.H_bar = H_bar
        self.l1_alpha = l1_alpha
        self._build_jit_arrays(threads_data)

    def _build_jit_arrays(self, threads_data: List[dict]):
        all_times, all_cats, all_Hs_c, all_p = [], [], [], []
        T_list, starts = [], [0]

        for td in threads_data:
            n = len(td["times_h"])
            all_times.extend(td["times_h"].tolist())
            all_cats.extend(td["cats"].tolist())
            all_Hs_c.extend(td["Hs_c"].tolist())
            all_p.append(np.asarray(td["p_dist"], dtype=np.float64))
            T_list.append(float(td["T"]))
            starts.append(starts[-1] + n)

        self._times = np.array(all_times, dtype=np.float64)
        self._cats = np.array(all_cats, dtype=np.int64)
        self._Hs_c = np.array(all_Hs_c, dtype=np.float64)
        self._p_dist = np.vstack(all_p) if all_p else np.zeros((0, self.K))
        self._thread_starts = np.array(starts, dtype=np.int64)
        self._T_threads = np.array(T_list, dtype=np.float64)
        self.n_events = len(self._times)
        self.n_threads = len(T_list)

    def neg_loglik_and_grad(self, v: np.ndarray) -> Tuple[float, np.ndarray]:
        return soft_mark_neg_loglik_and_grad(
            v, self.K, self.l1_alpha,
            self._times, self._cats, self._Hs_c, self._p_dist,
            self._thread_starts, self._T_threads,
        )

    def fit(
        self,
        n_restarts: int = 5,
        maxiter: int = 5000,
        seed: int = 0,
        init_from: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> SoftMarkResult:
        K = self.K
        n_p = n_params(K)
        rng = np.random.default_rng(seed)

        best_nll = np.inf
        best_x = None
        best_res = None

        starts_list = [init_from] if init_from is not None else []
        for _ in range(n_restarts - len(starts_list)):
            v0 = np.zeros(n_p)
            idx = 0
            v0[idx:idx + K] = rng.uniform(-1, 0, K); idx += K
            v0[idx:idx + K * K] = rng.uniform(-3, -1, K * K); idx += K * K
            v0[idx] = rng.uniform(-1, 0); idx += 1
            v0[idx] = rng.uniform(-0.5, 0.5); idx += 1
            v0[idx:idx + K] = rng.uniform(-2, -0.5, K)
            starts_list.append(v0)

        bounds = [(None, None)] * n_p
        bounds[K + K * K] = (np.log(0.1), np.log(3.0))

        for restart_i, v0 in enumerate(starts_list):
            try:
                res = minimize(
                    self.neg_loglik_and_grad, v0, method="L-BFGS-B",
                    jac=True, bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-7},
                )
                if verbose:
                    print(f"  [soft-mark] restart {restart_i}: nll={res.fun:.4f} success={res.success}")
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_x = res.x.copy()
                    best_res = res
            except Exception as e:
                if verbose:
                    print(f"  [soft-mark] restart {restart_i} failed: {e}")

        if best_x is None:
            return SoftMarkResult(
                np.zeros(K), np.zeros((K, K)), 1.0, 0.0, np.zeros(K),
                np.inf, self.n_events, self.n_threads, False,
                "all soft-mark restarts failed", self.H_bar, K,
            )

        mu_h, alpha_h, beta_h, gamma_h, b_h = unpack_v(best_x, K)
        return SoftMarkResult(
            mu_hat=mu_h, alpha_hat=alpha_h, beta_hat=beta_h,
            gamma_hat=gamma_h, b_hat=b_h,
            neg_loglik=float(best_nll),
            n_events=self.n_events, n_threads=self.n_threads,
            success=best_res.success, message=best_res.message,
            H_bar=self.H_bar, K=K,
        )
