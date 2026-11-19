"""Wrappers around existing implementations (no re-implementation).

Each wrapper returns a :class:`atdc.core.TreeEstimate` so that the same
metrics apply.  What can be evaluated differs by method:

* ToMATo (GUDHI): labels and persistence diagram.  GUDHI does not expose
  merge heights between retained clusters, so ``M`` is NaN off the diagonal;
  the tree metrics for "ToMATo with a log-kNN density" are those of
  ``core.full_tree``, which is tested to give the same diagram and labels as
  GUDHI on identical inputs.
* HDBSCAN (hdbscan package): labels from its own cluster selection.  Its
  single-linkage tree on the mutual reachability distance is converted to
  log-density units (a kNN radius r corresponds to psi(k) - log n - log V_d -
  d log r), which gives peak heights and merge heights for its clusters.
* DADApy / ADP runs in a separate environment (needs numpy < 2); see
  benchmarks/dadapy_worker.py.
"""
from __future__ import annotations

import time

import numpy as np

from .core import TreeEstimate
from .density import knn_log_density


def tomato_gudhi(X, K, k_graph, tau, density="logDTM") -> TreeEstimate:
    from gudhi.clustering.tomato import Tomato
    t0 = time.perf_counter()
    # GUDHI's k counts the point itself
    tm = Tomato(graph_type="knn", density_type=density, k=k_graph + 1, k_DTM=K + 1,
                merge_threshold=tau, n_jobs=1)
    tm.fit(X)
    labels = np.asarray(tm.labels_)
    dt = time.perf_counter() - t0
    w = np.asarray(tm.weights_)
    nc = tm.n_clusters_
    peaks = np.array([np.flatnonzero(labels == c)[np.argmax(w[labels == c])] for c in range(nc)])
    M = np.full((nc, nc), np.nan)
    M[np.arange(nc), np.arange(nc)] = w[peaks]
    return TreeEstimate(method=f"tomato_gudhi_{density}", peak_idx=peaks, peak_h=w[peaks],
                        peak_pers=np.full(nc, np.nan), M=M, labels=labels,
                        cost_units=float(len(X) * K), times={"total": dt},
                        info={"density": density})


def _linkage_merge_heights(Z, n, q):
    """Merge distance between query points from a SciPy-style linkage."""
    nq = len(q)
    M = np.full((nq, nq), np.inf)
    members = {int(v): [i] for i, v in enumerate(q)}
    for r, (a, b, dist, _) in enumerate(Z):
        ma = members.pop(int(a), None)
        mb = members.pop(int(b), None)
        if ma and mb:
            for i in ma:
                for j in mb:
                    M[i, j] = M[j, i] = dist
        if ma or mb:
            members[n + r] = (ma or []) + (mb or [])
    return M


def hdbscan_tree(X, min_samples, min_cluster_size, method="hdbscan",
                 cluster_selection_method="eom") -> TreeEstimate:
    import hdbscan
    from scipy.spatial import cKDTree
    n, d = X.shape
    t0 = time.perf_counter()
    cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                         cluster_selection_method=cluster_selection_method,
                         core_dist_n_jobs=1, gen_min_span_tree=False).fit(X)
    labels = cl.labels_.astype(np.int64)
    dt = time.perf_counter() - t0
    # conversion to log-density units (not timed: evaluation only)
    ms = min_samples if min_samples is not None else min_cluster_size   # package rule
    core = cKDTree(X).query(X, [ms])[0][:, 0]            # hdbscan counts the point itself
    k = max(ms - 1, 1)
    logden = knn_log_density(core, k, n, d)
    nc = int(labels.max()) + 1
    if nc == 0:
        # everything is noise / a single cluster: report one peak
        peaks = np.array([int(np.argmax(logden))])
        return TreeEstimate(method=method, peak_idx=peaks, peak_h=logden[peaks],
                            peak_pers=np.array([np.inf]), M=logden[peaks][None, :].copy(),
                            labels=labels, cost_units=float(n * ms), times={"total": dt},
                            info={"n_clusters": 0})
    peaks = np.array([np.flatnonzero(labels == c)[np.argmax(logden[labels == c])] for c in range(nc)])
    Z = cl.single_linkage_tree_.to_numpy()
    Dm = _linkage_merge_heights(Z, n, peaks)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = knn_log_density(Dm, k, n, d)
    M[np.arange(nc), np.arange(nc)] = logden[peaks]
    M = np.minimum(M, np.minimum.outer(logden[peaks], logden[peaks]))
    pers = np.array([logden[peaks[i]] - max([M[i, j] for j in range(nc)
                                              if j != i and logden[peaks[j]] >= logden[peaks[i]]],
                                             default=-np.inf) for i in range(nc)])
    return TreeEstimate(method=method, peak_idx=peaks, peak_h=logden[peaks], peak_pers=pers, M=M,
                        labels=labels, cost_units=float(n * ms), times={"total": dt},
                        info={"n_clusters": nc})


def automato_fit(X, n_bootstrap=1000, seed=0) -> TreeEstimate:
    from automato import Automato
    t0 = time.perf_counter()
    a = Automato(random_state=seed, n_bootstrap=n_bootstrap, parallelize=1).fit(X)
    labels = np.asarray(a.labels_).astype(np.int64)
    dt = time.perf_counter() - t0
    nc = int(labels.max()) + 1
    # no density exposed in a documented way: peak = cluster medoid proxy
    # (the point closest to the cluster mean); used for peak matching only
    peaks = []
    for c in range(nc):
        idx = np.flatnonzero(labels == c)
        peaks.append(idx[np.argmin(np.sum((X[idx] - X[idx].mean(0)) ** 2, axis=1))])
    peaks = np.array(peaks, dtype=np.int64)
    return TreeEstimate(method="automato", peak_idx=peaks, peak_h=np.full(nc, np.nan),
                        peak_pers=np.full(nc, np.nan), M=np.full((nc, nc), np.nan), labels=labels,
                        times={"total": dt}, info={"n_bootstrap": n_bootstrap})
