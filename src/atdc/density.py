"""k-nearest-neighbour log-density estimate.

log p_hat(x) = psi(k) - log(n) - log(V_d) - d * log(r_k(x))

where r_k is the distance to the k-th neighbour among n sample points and
V_d the volume of the unit d-ball.  Under a locally constant density,
n * p * V_d * r_k^d is Gamma(k, 1) distributed, so this estimator of log p is
unbiased (psi(k) instead of log(k)) with variance trigamma(k) ~ 1/k.  The
smoothing bias from the curvature of p over the ball of radius r_k is NOT
removed and depends on the bandwidth, i.e. on k/n.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln, polygamma


def log_unit_ball_volume(d: int) -> float:
    return 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)


def knn_log_density(r_k: np.ndarray, k: int, n: int, d: int) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return digamma(k) - np.log(n) - log_unit_ball_volume(d) - d * np.log(r_k)


def knn_log_density_se(k: int) -> float:
    """Standard error of the estimate under local uniformity."""
    return float(np.sqrt(polygamma(1, k)))
