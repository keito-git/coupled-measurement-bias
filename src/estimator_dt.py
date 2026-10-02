"""
DT-AMHP: discrete-time, arrival-conditioned ambiguity-modulated Hawkes process (hard-mark estimator).

One event per turn; the category of turn m follows the arrival-conditioned multinomial

    P(d_m = i | history) = Lambda_i(m) / sum_j Lambda_j(m)
    Lambda_i(m) = mu_i + sum_j alpha[i,j] * R_j(m)
    R_j(m) = sum_{k<m: d_k=j} exp(gamma * H_tilde_k) * exp(-beta * (m - k))

where H_tilde_k = H_k - H_bar is the centred modifier. The scale of (mu, alpha) is identified by
the softmax parameterisation of mu (sum_i mu_i = 1).

Unconstrained parameter vector v (length K + K^2 + 2):
    v_mu    (K):   mu = softmax(v_mu)
    v_alpha (K*K): alpha = softplus(v_alpha)
    v_beta  (1):   beta = exp(v_beta)
    v_gamma (1):   gamma = v_gamma

Running sums per source category j, updated after each turn (ef = exp(-beta)):
    R_j <- ef * (R_j + 1[j == d_m] * gain_m)
    P_j <- ef * P_j + R_j                      (beta gradient)
    Q_j <- ef * (Q_j + 1[j == d_m] * H_tilde_m * gain_m)   (gamma gradient)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numba import njit
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

def n_params_dt(K: int) -> int:
    """Total number of unconstrained parameters."""
    return K + K * K + 2


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -500, 20))))


def _softplus_inv(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Inverse softplus: log(exp(x) - 1)."""
    x = np.maximum(x, eps)
    return np.where(x > 20.0, x, np.log(np.expm1(x)))


def _stable_softmax(v: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def _stable_softmax_inv(mu: np.ndarray) -> np.ndarray:
    """Inverse softmax up to an additive constant (log, then centre)."""
    v = np.log(np.maximum(mu, 1e-9))
    return v - v.mean()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def pack_v_dt(mu: np.ndarray, alpha: np.ndarray, beta: float,
              gamma: float) -> np.ndarray:
    """Pack constrained parameters into the unconstrained vector."""
    return np.concatenate([
        _stable_softmax_inv(np.maximum(mu, 1e-9)),
        _softplus_inv(np.maximum(alpha.ravel(), 1e-6)),
        [np.log(max(beta, 1e-6))],
        [gamma],
    ])


def unpack_v_dt(v: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Unpack the unconstrained vector to (mu, alpha, beta, gamma)."""
    idx = 0
    mu = _stable_softmax(v[idx:idx + K]); idx += K
    alpha = _softplus(v[idx:idx + K * K]).reshape(K, K); idx += K * K
    beta = float(np.exp(v[idx])); idx += 1
    gamma = float(v[idx])
    return mu, alpha, beta, gamma


# ---------------------------------------------------------------------------
# numba kernels: negative log-likelihood and natural-parameter gradients
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dt_single_nll_grad(
    cats: np.ndarray,    # (M,) int64 categories
    Hs_c: np.ndarray,    # (M,) float64 centred modifier
    mu: np.ndarray,      # (K,)
    alpha: np.ndarray,   # (K, K)
    beta: float,
    gamma: float,
    K: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """
    NLL and gradients for one dialogue. Returns (nll, g_mu, g_alpha, g_beta, g_gamma), where the
    g_* are gradients of the log-likelihood (not of the NLL).
    """
    M = len(cats)
    g_mu = np.zeros(K)
    g_alpha = np.zeros((K, K))
    g_beta = 0.0
    g_gamma = 0.0

    R = np.zeros(K)
    P = np.zeros(K)
    Q = np.zeros(K)

    ef = math.exp(-beta)
    ll = 0.0

    for m in range(M):
        Lam = np.empty(K)
        for i in range(K):
            s = mu[i]
            for j in range(K):
                s += alpha[i, j] * R[j]
            if s < 1e-300:
                s = 1e-300
            Lam[i] = s

        S = 0.0
        for i in range(K):
            S += Lam[i]
        if S < 1e-300:
            S = 1e-300

        ik = cats[m]
        ll += math.log(Lam[ik] / S)

        # rho_i = d log P(d_m | history) / d Lambda_i
        inv_S = 1.0 / S
        rho = np.empty(K)
        for i in range(K):
            rho[i] = -inv_S
        rho[ik] += 1.0 / Lam[ik]

        for i in range(K):
            g_mu[i] += rho[i]

        for i in range(K):
            for j in range(K):
                g_alpha[i, j] += rho[i] * R[j]

        v_rho_alpha = np.zeros(K)
        for j in range(K):
            s_j = 0.0
            for i in range(K):
                s_j += rho[i] * alpha[i, j]
            v_rho_alpha[j] = s_j

        for j in range(K):
            g_beta -= v_rho_alpha[j] * P[j]
            g_gamma += v_rho_alpha[j] * Q[j]

        gs_h = gamma * Hs_c[m]
        if gs_h > 30.0:
            gs_h = 30.0
        elif gs_h < -30.0:
            gs_h = -30.0
        gain_m = math.exp(gs_h)

        for j in range(K):
            R_new_j = ef * (R[j] + (gain_m if j == ik else 0.0))
            P[j] = ef * P[j] + R_new_j
            Q[j] = ef * (Q[j] + (Hs_c[m] * gain_m if j == ik else 0.0))
            R[j] = R_new_j

    return -ll, g_mu, g_alpha, g_beta, g_gamma


@njit(cache=True)
def _dt_total_nll_grad(
    cats_all: np.ndarray,       # (N_total,) int64
    Hs_c_all: np.ndarray,       # (N_total,) float64
    thread_starts: np.ndarray,  # (n_dlg + 1,) int64
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: float,
    gamma: float,
    K: int,
) -> Tuple[float, np.ndarray, np.ndarray, float, float]:
    """Sum of NLL and natural-parameter gradients over all dialogues."""
    n_dlg = len(thread_starts) - 1
    total_nll = 0.0
    g_mu = np.zeros(K)
    g_alpha = np.zeros((K, K))
    g_beta = 0.0
    g_gamma = 0.0

    for d in range(n_dlg):
        start = thread_starts[d]
        end = thread_starts[d + 1]
        if end - start < 1:
            continue
        nll_d, gmu_d, galpha_d, gbeta_d, ggamma_d = _dt_single_nll_grad(
            cats_all[start:end],
            Hs_c_all[start:end],
            mu, alpha, beta, gamma, K,
        )
        total_nll += nll_d
        g_beta += gbeta_d
        g_gamma += ggamma_d
        for i in range(K):
            g_mu[i] += gmu_d[i]
            for j in range(K):
                g_alpha[i, j] += galpha_d[i, j]

    return total_nll, g_mu, g_alpha, g_beta, g_gamma


# ---------------------------------------------------------------------------
# NLL + L1 penalty and gradient with respect to the unconstrained vector
# ---------------------------------------------------------------------------

def neg_loglik_and_grad_dt(
    v: np.ndarray,
    K: int,
    l1_alpha: float,
    cats_all: np.ndarray,
    Hs_c_all: np.ndarray,
    thread_starts: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Return (NLL + l1_alpha * sum(alpha), gradient w.r.t. v)."""
    mu, alpha, beta, gamma = unpack_v_dt(v, K)

    nll, g_mu, g_alpha, g_beta_nat, g_gamma_nat = _dt_total_nll_grad(
        cats_all, Hs_c_all, thread_starts, mu, alpha, beta, gamma, K,
    )

    f = nll + l1_alpha * float(alpha.sum())

    n_p = n_params_dt(K)
    grad_v = np.empty(n_p)

    # softmax parameterisation of mu
    dotprod = float(np.dot(g_mu, mu))
    for j in range(K):
        grad_v[j] = mu[j] * (dotprod - g_mu[j])

    # softplus parameterisation of alpha
    idx_alpha_start = K
    sig_alpha = _sigmoid(v[idx_alpha_start:idx_alpha_start + K * K]).reshape(K, K)
    g_alpha_nat_total = -g_alpha + l1_alpha
    grad_v[idx_alpha_start:idx_alpha_start + K * K] = (g_alpha_nat_total * sig_alpha).ravel()

    # exp parameterisation of beta
    idx_beta = K + K * K
    grad_v[idx_beta] = -g_beta_nat * beta

    idx_gamma = K + K * K + 1
    grad_v[idx_gamma] = -g_gamma_nat

    return f, grad_v


def gradient_check_dt(
    K: int = 4,
    n_dlg: int = 10,
    l1_alpha: float = 0.001,
    seed: int = 42,
    eps: float = 1e-5,
) -> dict:
    """Compare the analytic gradient with central finite differences on synthetic data."""
    rng = np.random.default_rng(seed)
    n_p = n_params_dt(K)

    mu_true = rng.dirichlet(np.ones(K))
    alpha_true = rng.uniform(0.01, 0.1, (K, K))
    v0 = pack_v_dt(mu_true, alpha_true, 0.8, -0.5)

    all_cats, all_Hs_c = [], []
    starts = [0]
    for _ in range(n_dlg):
        M = int(rng.integers(5, 20))
        cats = rng.integers(0, K, M).astype(np.int64)
        Hs = rng.beta(2.0, 2.0, M).astype(np.float64) - 0.5
        all_cats.extend(cats.tolist())
        all_Hs_c.extend(Hs.tolist())
        starts.append(starts[-1] + M)

    kwargs = dict(K=K, l1_alpha=l1_alpha,
                  cats_all=np.array(all_cats, dtype=np.int64),
                  Hs_c_all=np.array(all_Hs_c, dtype=np.float64),
                  thread_starts=np.array(starts, dtype=np.int64))

    f0, grad_analytic = neg_loglik_and_grad_dt(v0, **kwargs)
    grad_numerical = np.zeros(n_p)
    for i in range(n_p):
        vp = v0.copy(); vp[i] += eps
        vm = v0.copy(); vm[i] -= eps
        grad_numerical[i] = (neg_loglik_and_grad_dt(vp, **kwargs)[0]
                             - neg_loglik_and_grad_dt(vm, **kwargs)[0]) / (2.0 * eps)

    rel_err = np.abs(grad_analytic - grad_numerical) / (np.abs(grad_numerical) + 1e-10)
    return {"passed": bool(rel_err.max() < 1e-5), "max_rel_err": float(rel_err.max()),
            "mean_rel_err": float(rel_err.mean()), "f0": float(f0), "n_params": n_p}


# ---------------------------------------------------------------------------
# Multi-restart L-BFGS-B fitter
# ---------------------------------------------------------------------------

@dataclass
class DTResult:
    mu_hat: np.ndarray
    alpha_hat: np.ndarray
    beta_hat: float
    gamma_hat: float
    neg_loglik: float
    n_events: int
    n_dialogues: int
    success: bool
    message: str
    H_bar: float
    K: int
    v_hat: np.ndarray = field(default_factory=lambda: np.array([]))


class DTEstimator:
    """DT-AMHP estimator with analytic gradient and multiple restarts."""

    def __init__(
        self,
        dialogues: List[dict],  # each: {"cats": ndarray, "Hs_c": ndarray}
        K: int = 7,
        H_bar: float = 0.0,
        l1_alpha: float = 0.001,
    ) -> None:
        self.K = K
        self.H_bar = H_bar
        self.l1_alpha = l1_alpha
        self._dialogues = dialogues

        all_cats, all_Hs_c = [], []
        starts = [0]
        n_ev = 0
        for d in dialogues:
            cats = np.asarray(d["cats"], dtype=np.int64)
            Hs_c = np.asarray(d["Hs_c"], dtype=np.float64)
            all_cats.append(cats)
            all_Hs_c.append(Hs_c)
            starts.append(starts[-1] + len(cats))
            n_ev += len(cats)
        self._cats_all = np.concatenate(all_cats).astype(np.int64)
        self._Hs_c_all = np.concatenate(all_Hs_c).astype(np.float64)
        self._thread_starts = np.array(starts, dtype=np.int64)
        self.n_events = n_ev
        self.n_dialogues = len(dialogues)

    def _obj(self, v: np.ndarray) -> Tuple[float, np.ndarray]:
        return neg_loglik_and_grad_dt(
            v, self.K, self.l1_alpha,
            self._cats_all, self._Hs_c_all, self._thread_starts,
        )

    def neg_loglik(self, v: np.ndarray, l1: bool = False) -> float:
        """NLL under v (L1 penalty included only if l1=True)."""
        la = self.l1_alpha if l1 else 0.0
        return neg_loglik_and_grad_dt(
            v, self.K, la,
            self._cats_all, self._Hs_c_all, self._thread_starts,
        )[0]

    def fit(
        self,
        n_restarts: int = 5,
        maxiter: int = 5000,
        seed: int = 0,
        init_from: Optional[np.ndarray] = None,
        bounds: Optional[list] = None,
        verbose: bool = False,
    ) -> DTResult:
        """Fit with L-BFGS-B from several random starts; keep the best."""
        K = self.K
        n_p = n_params_dt(K)
        rng = np.random.default_rng(seed)

        if bounds is None:
            bounds = [(None, None)] * n_p
            bounds[K + K * K] = (np.log(0.05), np.log(5.0))

        starts = []
        if init_from is not None:
            starts.append(init_from.copy())
        for _ in range(n_restarts - len(starts)):
            v0 = np.zeros(n_p)
            v0[:K] = rng.uniform(-0.5, 0.5, K)
            v0[K:K + K * K] = rng.uniform(-2.0, -0.5, K * K)
            v0[K + K * K] = rng.uniform(np.log(0.3), np.log(1.5))
            v0[K + K * K + 1] = rng.uniform(-0.5, 0.5)
            starts.append(v0)

        best_f = np.inf
        best_x = None
        best_res = None

        for i_r, v0 in enumerate(starts):
            try:
                res = minimize(
                    self._obj, v0, method="L-BFGS-B",
                    jac=True, bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7},
                )
                if verbose:
                    print(f"  restart {i_r}: nll={res.fun:.4f}  success={res.success}")
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    best_res = res
            except Exception as e:
                if verbose:
                    print(f"  restart {i_r} failed: {e}")

        if best_x is None:
            return DTResult(
                np.ones(K) / K, np.zeros((K, K)), 1.0, 0.0,
                np.inf, self.n_events, self.n_dialogues, False,
                "all restarts failed", self.H_bar, K,
                np.zeros(n_p),
            )

        mu_h, alpha_h, beta_h, gamma_h = unpack_v_dt(best_x, K)
        return DTResult(
            mu_hat=mu_h, alpha_hat=alpha_h, beta_hat=beta_h, gamma_hat=gamma_h,
            neg_loglik=float(best_f), n_events=self.n_events,
            n_dialogues=self.n_dialogues, success=best_res.success,
            message=best_res.message, H_bar=self.H_bar, K=K,
            v_hat=best_x,
        )
