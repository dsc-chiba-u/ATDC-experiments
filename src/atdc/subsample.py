"""Uniform subsample density cluster tree (Phase A reference implementation).

This is the estimator that neither selective design could beat
(docs/evaluation_v2.md) and the candidate core of the software:

  1. m vertices drawn uniformly without replacement from the n data points;
  2. kNN log-density of every vertex against ALL data (K neighbours, the
     vertex itself excluded);
  3. undirected kNN graph among the vertices (k_graph neighbours);
  4. merge tree (0-dim superlevel persistence) of the vertex densities on that
     graph, an edge carrying min(f_u, f_v);
  5. peaks with log-prominence >= tau are the clusters; a vertex gets the label
     of the peak its component is absorbed into (ToMATo rule); every data
     point gets the label of its nearest vertex.

Nothing here is new: 4-5 are ToMATo, 1-2 are the subsample-density idea of
DBSCAN++.  The numbers agree with ``ATDC2`` without refinement when the same
vertices are used (tests/test_subsample.py).

Neighbour search is pluggable so that its cost and its accuracy can be
measured separately (benchmarks/13_dimension.py):
  kdtree   scipy cKDTree (exact)
  brute    exact, blocked BLAS distances; no index (cost n*m*d)
  hnsw     hnswlib, approximate (index on all data)
  nndescent pynndescent, approximate
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from .core import TreeEstimate, knn_graph_edges
from .density import knn_log_density
from .mergetree import merge_tree


# ------------------------------------------------------------ neighbour search

# Work arrays of the exact search are kept below this many elements (64 MB of
# doubles), whatever n, m and d are; see results/raw/16_cost_brute2.csv.
_BLOCK = 8_000_000


def _brute_knn(Q, X, k, exclude=None, block=_BLOCK):
    """Exact k nearest rows of X for every row of Q.  ``exclude[i]`` is the row
    of X that is Q[i] itself (dropped).  Returns (dist, ind), sorted."""
    Q = np.asarray(Q)
    x2 = np.einsum("ij,ij->i", X, X)
    D = np.empty((len(Q), k)); I = np.empty((len(Q), k), dtype=np.int64)
    step = max(1, int(block / max(len(X), 1)))
    for s in range(0, len(Q), step):
        q = Q[s:s + step]
        d2 = q @ X.T                                    # one work array, updated in place
        d2 *= -2.0
        d2 += x2[None, :]
        d2 += np.einsum("ij,ij->i", q, q)[:, None]
        if exclude is not None:
            d2[np.arange(len(q)), exclude[s:s + step]] = np.inf
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dd = np.take_along_axis(d2, part, axis=1)
        o = np.argsort(dd, axis=1)
        I[s:s + step] = np.take_along_axis(part, o, axis=1)
        D[s:s + step] = np.sqrt(np.maximum(np.take_along_axis(dd, o, axis=1), 0.0))
    return D, I


def _exact_radius(X, q, ind, block=_BLOCK):
    """Distances from q[i] to X[ind[i]] computed directly (no cancellation)."""
    ind = np.asarray(ind)
    out = np.empty(ind.shape)
    step = max(1, int(block / max(ind.shape[1] * X.shape[1], 1)))
    for s in range(0, len(q), step):
        diff = X[ind[s:s + step]] - q[s:s + step, None, :]
        out[s:s + step] = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return out


class Neighbours:
    """kNN of query points among the data, with the query's own row removed."""

    def __init__(self, X, backend="kdtree", seed=0, **opt):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.backend = backend
        self.opt = opt
        n, d = self.X.shape
        if backend == "kdtree":
            self.index = cKDTree(self.X)
        elif backend == "brute":
            self.index = None
        elif backend == "hnsw":
            import hnswlib
            p = hnswlib.Index(space="l2", dim=d)
            p.init_index(max_elements=n, ef_construction=opt.get("ef_construction", 200),
                         M=opt.get("M", 16), random_seed=int(seed))
            p.set_num_threads(1)
            p.add_items(self.X.astype(np.float32), np.arange(n))
            self.index = p
        elif backend == "nndescent":
            from pynndescent import NNDescent
            self.index = NNDescent(self.X.astype(np.float32), n_neighbors=opt.get("n_neighbors", 30),
                                   random_state=int(seed), n_jobs=1)
            self.index.prepare()
        else:
            raise ValueError(backend)

    def kth_distance(self, pts, k):
        """Distance from arbitrary locations to their k-th nearest data point
        (exact back-ends only)."""
        if self.backend == "kdtree":
            return self.index.query(pts, [k])[0][:, 0]
        if self.backend == "brute":
            d, i = _brute_knn(pts, self.X, k)
            return _exact_radius(self.X, np.asarray(pts), i[:, -1:])[:, 0]
        raise NotImplementedError(self.backend)

    def query_data(self, idx, k):
        """k nearest data points of data points ``idx`` (self excluded)."""
        q = self.X[idx]
        if self.backend == "kdtree":
            d, i = self.index.query(q, k + 1)
            return d[:, 1:], i[:, 1:]
        if self.backend == "brute":
            d, i = _brute_knn(q, self.X, k, exclude=np.asarray(idx))
            return _exact_radius(self.X, q, i), i
        if self.backend == "hnsw":
            self.index.set_ef(max(self.opt.get("ef", 2 * k), k + 1))
            i, _ = self.index.knn_query(q.astype(np.float32), k=k + 1)
        else:
            i, _ = self.index.query(q.astype(np.float32), k=k + 1, epsilon=self.opt.get("epsilon", 0.1))
        i = i.astype(np.int64)
        out = np.empty((len(idx), k), dtype=np.int64)
        for r, (row, me) in enumerate(zip(i, idx)):         # drop self if returned
            row = row[row != me]
            out[r] = row[:k]
        d = _exact_radius(self.X, q, out)
        o = np.argsort(d, axis=1)
        return np.take_along_axis(d, o, axis=1), np.take_along_axis(out, o, axis=1)


def nearest_vertex(X, V, backend="kdtree"):
    """Index of the nearest row of V for every row of X (exact)."""
    if backend == "kdtree":
        return cKDTree(V).query(X, 1)[1]
    v2 = np.einsum("ij,ij->i", V, V)
    out = np.empty(len(X), dtype=np.int64)
    step = max(1, int(_BLOCK / max(len(V), 1)))
    for s in range(0, len(X), step):
        d2 = X[s:s + step] @ V.T
        d2 *= -2.0
        d2 += v2[None, :]
        out[s:s + step] = np.argmin(d2, axis=1)
    return out


def intrinsic_dim_mle(dist):
    """Intrinsic dimension from sorted kNN distances (rows = points).

    Maximum-likelihood estimator of Levina & Bickel (2005) with the averaging
    of MacKay & Ghahramani (2005): the inverse of the mean over points of
    (1 / (K - 1)) * sum_j log(r_K / r_j).  Uses distances that the density
    step has computed anyway.  Zero distances (duplicates) are skipped.
    """
    dist = np.asarray(dist, dtype=float)
    K = dist.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(dist[:, -1:] / dist[:, :-1])
    ok = np.isfinite(lr)
    s = np.where(ok, lr, 0.0).sum(axis=1) / np.maximum(ok.sum(axis=1), 1)
    s = s[np.isfinite(s) & (s > 0)]
    return float(1.0 / s.mean()) if s.size else float("nan")


# ------------------------------------------------------------------ estimator

class SubsampleTree:
    """Fit once, prune at any tau."""

    def __init__(self, X, m=1000, K=100, k_graph=10, seed=0, backend="kdtree", vertex_idx=None,
                 label_backend=None, edge_check=False, dim=None, **opt):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.K, self.k_graph, self.backend = int(K), int(k_graph), backend
        self.label_backend = label_backend or ("kdtree" if self.d <= 15 else "brute")
        self.times = {}
        if vertex_idx is None:
            rng = np.random.default_rng([*np.atleast_1d(seed).astype(int).tolist(), 0x1217])  # as ATDC2
            m = min(max(int(m), self.k_graph + 2), self.n)
            vertex_idx = rng.choice(self.n, size=m, replace=False)
        self.idx = np.asarray(vertex_idx, dtype=np.int64)
        self.m = len(self.idx)
        t0 = time.perf_counter()
        nb = Neighbours(self.X, backend, seed=int(np.atleast_1d(seed)[0]), **opt)
        self.times["index_full"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dist, self.nb_ind = nb.query_data(self.idx, self.K)
        self.r = dist[:, -1]
        # ``dim``: dimension used in the density formula.  None = ambient d;
        # "auto" = intrinsic-dimension estimate; or a number.  The tree as a
        # combinatorial object does not depend on it (f is a decreasing
        # function of r for any dim > 0); the prominence VALUES scale with it.
        self.d_hat = intrinsic_dim_mle(dist)
        self.dim = float(self.d if dim is None else (self.d_hat if dim == "auto" else dim))
        self.f = knn_log_density(self.r, self.K, self.n, self.dim)
        self.times["query"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        V = self.X[self.idx]
        k = min(self.k_graph, self.m - 1)
        if self.d <= 15:
            ind = cKDTree(V).query(V, k + 1)[1]
        else:
            _, ind = _brute_knn(V, V, k, exclude=np.arange(self.m))
            ind = np.concatenate([np.arange(self.m)[:, None], ind], axis=1)
        self.edges = knn_graph_edges(ind, k)
        self.times["graph"] = time.perf_counter() - t0
        self.w = None
        if edge_check:
            # optional: an edge cannot carry more than the density at its
            # midpoint (one extra density evaluation per edge), so that a long
            # edge of the sparse graph does not jump over a valley
            t0 = time.perf_counter()
            mid = 0.5 * (V[self.edges[:, 0]] + V[self.edges[:, 1]])
            fm = knn_log_density(nb.kth_distance(mid, self.K), self.K, self.n, self.dim)
            self.w = np.minimum(np.minimum(self.f[self.edges[:, 0]], self.f[self.edges[:, 1]]), fm)
            self.times["edge_check"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.tree = merge_tree(self.f, self.edges, self.w)
        self.times["tree"] = time.perf_counter() - t0
        self._nn = None

    def nearest(self):
        if self._nn is None:
            t0 = time.perf_counter()
            self._nn = nearest_vertex(self.X, self.X[self.idx], self.label_backend)
            self.times["label"] = time.perf_counter() - t0
        return self._nn

    def prominence(self):
        """Log-prominence of every local maximum, sorted decreasing (the
        global maximum of each graph component has prominence inf)."""
        p = self.tree.persistence
        return np.sort(np.r_[p, np.full(len(self.tree.roots), np.inf)])[::-1]

    def n_clusters(self, tau):
        return int(np.sum(self.tree.persistence >= tau)) + len(self.tree.roots)

    def estimate(self, tau, method="subsample", with_labels=True) -> TreeEstimate:
        tree = self.tree
        peaks = tree.peaks(tau)
        death = tree.death_level()
        labels = None
        if with_labels:
            pos = -np.ones(tree.n, dtype=np.int64)
            pos[peaks] = np.arange(len(peaks))
            labels = pos[tree.labels(tau)][self.nearest()]
        return TreeEstimate(method=method, peak_idx=self.idx[peaks], peak_h=tree.f[peaks],
                            peak_pers=tree.f[peaks] - death[peaks], M=tree.merge_heights(peaks),
                            labels=labels,
                            cost_units=float((self.m + (0 if self.w is None else len(self.edges)))
                                             * self.K + self.n),
                            times=dict(self.times),
                            info={"n_vertices": self.m, "m0": self.m, "n_edges": len(self.edges),
                                  "backend": self.backend, "d_hat": self.d_hat, "dim": self.dim})
