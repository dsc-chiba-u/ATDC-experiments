"""Prototype of the diagnostics of the specification draft (Phase A).

docs/phaseA_spec_draft.md, sections 3 and 10: default K, noise floor, stability
to resampling, recommendation.  Prototype only; the packages implement the
specification, not this file.
"""
from __future__ import annotations

import numpy as np
from scipy.special import polygamma
from sklearn.metrics import adjusted_rand_score

from .subsample import SubsampleTree, nearest_vertex


def default_K(n: int) -> int:
    return int(min(100, max(10, n // 20)))


def noise_floor(K: int, z: float = 3.5) -> float:
    return float(z * np.sqrt(2.0 * polygamma(1, K)))


def vertex_labels(tree, tau):
    peaks = tree.peaks(tau)
    pos = -np.ones(tree.n, dtype=np.int64)
    pos[peaks] = np.arange(len(peaks))
    return pos[tree.labels(tau)]


def tau_grid(fit, K, n_grid=25):
    p = fit.tree.persistence
    p = p[np.isfinite(p)]
    t0 = noise_floor(K)
    hi = max(float(p.max()) if p.size else t0, 2 * t0)
    return np.geomspace(t0 / 2, hi, n_grid)


def stability_path(X, K=None, m=4000, B=20, fraction=0.8, seed=0, taus=None, k_graph=10, backend=None):
    """Returns (fit, table): the original fit and, per threshold, the number of
    clusters of the original fit, the stability, the modal count and its
    agreement over B replicates on row subsamples."""
    X = np.ascontiguousarray(X, dtype=float)
    n, d = X.shape
    K = default_K(n) if K is None else K
    backend = backend or ("kdtree" if d <= 15 else "brute")
    fit = SubsampleTree(X, m=min(m, n), K=K, k_graph=k_graph, seed=seed, dim="auto", backend=backend)
    D = fit.dim if np.isfinite(fit.dim) and fit.dim >= 1 else float(d)
    taus = tau_grid(fit, K) if taus is None else np.asarray(taus, float)
    rng = np.random.default_rng([*np.atleast_1d(seed).astype(int).tolist(), 0x57AB])
    labs = np.empty((B, len(taus), n), dtype=np.int32)
    ns = int(np.ceil(fraction * n))
    for b in range(B):
        rows = np.sort(rng.choice(n, ns, replace=False))
        t = SubsampleTree(X[rows], m=min(m, ns), K=K, k_graph=k_graph, seed=[int(np.atleast_1d(seed)[0]), b + 1],
                          dim=D, backend=backend)
        nn = nearest_vertex(X, X[rows][t.idx], "kdtree" if d <= 15 else "brute")
        for j, tau in enumerate(taus):
            labs[b, j] = vertex_labels(t.tree, tau)[nn]
    out = []
    for j, tau in enumerate(taus):
        cnt = labs[:, j].max(axis=1) + 1
        vals, freq = np.unique(cnt, return_counts=True)
        st = float(np.mean([adjusted_rand_score(labs[b, j], labs[(b + 1) % B, j]) for b in range(B)]))
        out.append(dict(tau=float(tau), n_clusters=fit.n_clusters(tau), stability=st,
                        count_mode=int(vals[np.argmax(freq)]), count_agreement=float(freq.max() / B)))
    return fit, out


def recommend(table, K, min_stability=0.8, min_agreement=0.0):
    """Thresholds >= noise floor whose partition has at least two clusters and
    is stable, finest first (specification draft, 10.3).  Empty list: no
    partition into several clusters is stable at this resolution.

    History: the first version (stability >= 0.9, the same number of clusters
    in 80 % of the replicates, and the finest such threshold REPLACING the
    default partition) was rejected after the development runs; see the
    specification draft.  ``min_agreement`` is kept for those reruns.
    """
    t0 = noise_floor(K)
    return [r for r in table if r["tau"] >= t0 and r["count_mode"] >= 2 and r["stability"] >= min_stability
            and r["count_agreement"] >= min_agreement]
