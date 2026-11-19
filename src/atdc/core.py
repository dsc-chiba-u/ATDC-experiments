"""ATDC proof of concept: coarse merge tree on a subsample (Stage A) and
selective refinement with full-data density estimates (Stage B).

See docs/algorithm_design.md for the rationale.  Summary of the objects:

* fidelity 0 ("coarse"): kNN log-density of a subsample point computed inside
  the subsample S (m = q n points) with k0 = round(q K) neighbours.
* fidelity 1 ("fine"): kNN log-density of the same point computed in the full
  data with K neighbours.

Both use (in expectation) the same ball radius, hence the same smoothing bias;
fidelity 1 has ~1/q times smaller variance.  Refinement therefore improves the
*computational* approximation of the full-data estimator at bandwidth K; it
does not change the bias-variance trade-off set by K.

The hierarchy is represented by the adjacency graph of the coarse ascent
basins: node value h_a (peak log-density of basin a), edge value s_ab (saddle
log-density between adjacent basins a and b).  The merge tree is the tree of
that small graph.  Stage B replaces selected h_a and s_ab by fidelity-1
estimates and recomputes the tree.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .density import knn_log_density, knn_log_density_se
from .mergetree import (MergeTree, basin_graph, edge_weights, merge_tree,
                        unique_undirected_edges)


@dataclass
class TreeEstimate:
    """Common output format of every tree-producing method (for metrics)."""

    method: str
    peak_idx: np.ndarray      # indices into the data X of retained peaks
    peak_h: np.ndarray        # estimated log-density at the peaks
    peak_pers: np.ndarray     # estimated log-prominence (inf for roots)
    M: np.ndarray             # merge heights between retained peaks
    labels: np.ndarray | None  # cluster label (peak position in peak_idx) per data point
    cost_units: float = 0.0   # sum over kNN queries of k (neighbour evaluations)
    times: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def time_total(self):
        return float(sum(self.times.values()))


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key, t0):
        self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


def knn_graph_edges(ind: np.ndarray, k_graph: int) -> np.ndarray:
    """Undirected kNN graph from a neighbour index array (self in column 0)."""
    n = ind.shape[0]
    nb = ind[:, 1:k_graph + 1]
    return unique_undirected_edges(np.repeat(np.arange(n), nb.shape[1]), nb.ravel())


def estimate_from_vertex_tree(method, tree: MergeTree, tau, vertex_to_data, n_data=None,
                              label_fn=None, **kw) -> TreeEstimate:
    peaks = tree.peaks(tau)
    death = tree.death_level()
    lab = tree.labels(tau)
    # labels as positions in peaks
    pos = -np.ones(tree.n, dtype=np.int64)
    pos[peaks] = np.arange(len(peaks))
    vlab = pos[lab]
    labels = label_fn(vlab) if label_fn is not None else vlab
    return TreeEstimate(method=method, peak_idx=vertex_to_data[peaks], peak_h=tree.f[peaks],
                        peak_pers=tree.f[peaks] - death[peaks], M=tree.merge_heights(peaks),
                        labels=labels, **kw)


def full_tree(X, K, k_graph, tau, method="full") -> TreeEstimate:
    """Reference without subsampling: kNN density (K) and kNN graph on all
    points, 0-dim persistence, prune at tau.  This is ToMATo with a log-kNN
    density; GUDHI's implementation is run separately as a cross-check."""
    tm = Timer()
    n, d = X.shape
    t0 = time.perf_counter(); tree = cKDTree(X); tm.add("index_full", t0)
    t0 = time.perf_counter(); dist, ind = tree.query(X, K + 1); tm.add("knn_full", t0)
    t0 = time.perf_counter()
    f = knn_log_density(dist[:, K], K, n, d)
    edges = knn_graph_edges(ind, k_graph)
    mt = merge_tree(f, edges)
    est = estimate_from_vertex_tree(method, mt, tau, np.arange(n))
    tm.add("graph_tree", t0)
    est.cost_units = float(n * K)
    est.times = tm.t
    est.info = {"n_vertices": n, "n_edges": len(edges), "f": f}
    return est


class ATDC:
    """Two-stage estimator.  ``stage_a`` must be called first."""

    def __init__(self, X, K=100, q=0.1, k_graph=None, t_top=5, k_min=4, seed=0,
                 midpoint=False):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.K = int(K)
        self.q = float(q)
        self.m = max(int(round(self.q * self.n)), k_min + 2)
        self.k0 = max(k_min, int(round(self.q * self.K)))
        self.k_graph = int(k_graph) if k_graph is not None else self.k0
        self.k_graph = min(self.k_graph, self.m - 1)
        self.t_top = int(t_top)
        self.midpoint = bool(midpoint)
        self.rng = np.random.default_rng(seed)
        self.tm = Timer()
        self.cost = 0.0
        self._full_index = None
        self.n_refined = 0
        self.iterations = 0

    # ------------------------------------------------------------------ A
    def stage_a(self):
        t0 = time.perf_counter()
        self.S = np.sort(self.rng.choice(self.n, size=self.m, replace=False)) if self.m < self.n \
            else np.arange(self.n)
        self.XS = self.X[self.S]
        self.index_sub = cKDTree(self.XS)
        self.tm.add("index_sub", t0)
        t0 = time.perf_counter()
        kq = max(self.k0, self.k_graph)
        dist, ind = self.index_sub.query(self.XS, kq + 1)
        self.tm.add("knn_sub", t0)
        self.cost += self.m * kq
        t0 = time.perf_counter()
        self.f0 = knn_log_density(dist[:, self.k0], self.k0, self.m, self.d)
        self.se0 = knn_log_density_se(self.k0)
        self.se1 = knn_log_density_se(self.K)
        self.edges = knn_graph_edges(ind, self.k_graph)
        self.w0 = edge_weights(self.f0, self.edges)
        self.bg = basin_graph(self.f0, self.edges, self.w0)
        B = len(self.bg.peak)
        # vertices of each basin sorted by decreasing coarse value
        order = np.lexsort((-self.f0, self.bg.basin))
        counts = np.bincount(self.bg.basin, minlength=B)
        self._basin_order = order
        self._basin_start = np.r_[0, np.cumsum(counts)]
        # state of the basin graph
        self.h = self.f0[self.bg.peak].copy()
        self.s = self.bg.saddle.copy()
        self.ver_h = np.zeros(B, dtype=bool)
        self.ver_s = np.zeros(len(self.s), dtype=bool)
        self.f1 = np.full(self.m, np.nan)           # fine values where computed
        self.h_ref = np.full(B, -np.inf)            # max fine value seen per basin
        self.tm.add("graph_tree", t0)
        return self

    # ------------------------------------------------------------ helpers
    @property
    def full_index(self):
        if self._full_index is None:
            t0 = time.perf_counter()
            self._full_index = cKDTree(self.X)
            self.tm.add("index_full", t0)
        return self._full_index

    def _fine_density(self, pts, is_data=True):
        """Fidelity-1 log-density at subsample vertices (leave-one-out) or at
        arbitrary locations."""
        t0 = time.perf_counter()
        k = self.K + 1 if is_data else self.K
        dist, _ = self.full_index.query(pts, [k])
        self.tm.add("refine_query", t0)
        self.cost += len(pts) * self.K
        return knn_log_density(dist[:, 0], self.K, self.n, self.d)

    def refine_vertices(self, idx):
        idx = np.unique(np.asarray(idx, dtype=np.int64))
        idx = idx[np.isnan(self.f1[idx])]
        if idx.size:
            self.f1[idx] = self._fine_density(self.XS[idx])
            np.maximum.at(self.h_ref, self.bg.basin[idx], self.f1[idx])
            self.n_refined += idx.size
        return idx.size

    def _top_vertices(self, a):
        s, e = self._basin_start[a], self._basin_start[a + 1]
        return self._basin_order[s:min(e, s + self.t_top)]

    def _top_edges(self, i):
        return self.bg.cross[i][:self.t_top]

    def _verify(self, peaks, adjs):
        """Bring the given basin peaks and adjacencies to fidelity 1."""
        peaks = [a for a in peaks if not self.ver_h[a]]
        adjs = [i for i in adjs if not self.ver_s[i]]
        need = [self._top_vertices(a) for a in peaks]
        need += [self.edges[self._top_edges(i)].ravel() for i in adjs]
        if not need:
            return 0
        n_new = self.refine_vertices(np.concatenate(need))
        for i in adjs:
            e = self.edges[self._top_edges(i)]
            w = np.minimum(self.f1[e[:, 0]], self.f1[e[:, 1]])
            if self.midpoint:
                mid = 0.5 * (self.XS[e[:, 0]] + self.XS[e[:, 1]])
                w = np.minimum(w, self._fine_density(mid, is_data=False))
            self.s[i] = w.max()
            self.ver_s[i] = True
        self.ver_h[peaks] = True
        # a verified peak is the highest fine value seen in its basin
        v = self.ver_h
        self.h[v] = self.h_ref[v]
        return n_new

    def basin_tree(self, z_saddle=0.0) -> MergeTree:
        """Tree of the basin graph from the current values.  With
        ``z_saddle`` > 0 the unverified saddles are replaced by the optimistic
        value s + z_saddle * se0 (upper confidence bound)."""
        a, b = self.bg.adj[:, 0], self.bg.adj[:, 1]
        s = self.s if z_saddle <= 0 else self.s + z_saddle * self.se0 * (~self.ver_s)
        w = np.minimum(s, np.minimum(self.h[a], self.h[b]))
        return merge_tree(self.h, self.bg.adj, w, use_mst=False)

    # ------------------------------------------------------------------ B
    def refine(self, tau, z=0.0, budget=None, mode="selective", max_iter=10_000, z_saddle=0.0):
        """Stage B.

        mode = "selective": verify the peaks and saddles of merge events whose
               log-prominence could still reach tau (R + z*se >= tau), most
               prominent first, recompute the tree, repeat.
               With z_saddle > 0, additionally: for every currently retained
               peak, the adjacency at which it would die if the unverified
               saddles took their optimistic value s + z_saddle*se0 is
               verified (an unverified saddle that was underestimated by the
               coarse stage can otherwise leave a false peak).
        mode = "random":    verify uniformly random peaks/adjacencies.
        mode = "all":       verify every peak and adjacency.
        ``budget`` is the maximal number of fine density evaluations (points);
        None means unlimited (selective then stops at its own fixpoint).
        """
        B, A = len(self.h), len(self.s)
        start = self.n_refined
        left = (lambda: np.inf) if budget is None else (lambda: budget - (self.n_refined - start))
        stop = "fixpoint"
        if mode == "all":
            self._verify(range(B), range(A))
        elif mode == "random":
            items = self.rng.permutation(B + A)
            # process in chunks so the budget is respected approximately
            per_item = 1.5 * self.t_top
            pos = 0
            while pos < len(items) and left() > 0:
                step = int(max(1, min(len(items) - pos, left() // per_item + 1)))
                chunk = items[pos:pos + step]
                self._verify(chunk[chunk < B].tolist(), (chunk[chunk >= B] - B).tolist())
                pos += step
            stop = "budget" if pos < len(items) else "all"
        elif mode == "selective":
            for it in range(max_iter):
                t0 = time.perf_counter()
                tb = self.basin_tree()
                R = tb.persistence
                dy, sv, ed = tb.dying, tb.surviving, tb.edge
                unv = (~self.ver_h[dy]).astype(float) + (~self.ver_s[ed]).astype(float)
                ucb = R + z * self.se0 * np.sqrt(unv)
                cand = np.flatnonzero((ucb >= tau) & (unv > 0))
                # survivors of events that are retained after verification
                done = np.flatnonzero((unv == 0) & (R >= tau) & ~self.ver_h[sv])
                # peaks that never die (one per connected component) are
                # always retained, so their height is verified too
                roots = tb.roots[~self.ver_h[tb.roots]]
                extra = np.empty(0, dtype=np.int64)
                if z_saddle > 0:
                    to = self.basin_tree(z_saddle)
                    kept = dy[R >= tau]
                    extra = to.edge[np.isin(to.dying, kept) & ~self.ver_s[to.edge]]
                self.tm.add("refine_tree", t0)
                if cand.size == 0 and done.size == 0 and roots.size == 0 and extra.size == 0:
                    break
                if left() <= 0:
                    stop = "budget"
                    break
                cand = cand[np.argsort(-ucb[cand], kind="stable")]
                if budget is not None:
                    # each event needs at most 3 * t_top new points
                    cap = int(max(1, left() // (3 * self.t_top)))
                    cand = cand[:cap]
                self._verify(list(dict.fromkeys(dy[cand].tolist() + sv[done].tolist() + roots.tolist())),
                             list(dict.fromkeys(ed[cand].tolist() + extra.tolist())))
                self.iterations += 1
            else:
                stop = "max_iter"
        else:
            raise ValueError(mode)
        self.stop_reason = stop
        return self

    # ------------------------------------------------------------- output
    def label_all(self, vertex_labels):
        """Propagate subsample labels to all data points (1-NN)."""
        t0 = time.perf_counter()
        _, nn = self.index_sub.query(self.X, 1)
        self.tm.add("label", t0)
        self.cost += self.n
        return vertex_labels[nn]

    def estimate(self, tau, method="atdc", with_labels=True) -> TreeEstimate:
        t0 = time.perf_counter()
        tb = self.basin_tree()
        peaks = tb.peaks(tau)
        death = tb.death_level()
        pos = -np.ones(tb.n, dtype=np.int64)
        pos[peaks] = np.arange(len(peaks))
        blab = pos[tb.labels(tau)]
        self.tm.add("refine_tree", t0)
        labels = self.label_all(blab[self.bg.basin]) if with_labels else None
        # location of a retained peak: best vertex of the basin at the best
        # available fidelity
        loc = []
        for a in peaks:
            top = self._top_vertices(a)
            if self.ver_h[a]:
                top = top[np.argmax(self.f1[top])]
            else:
                top = top[0]
            loc.append(self.S[top])
        events = dict(zip(tb.dying.tolist(), tb.edge.tolist()))
        unverified = sum(1 for a in peaks
                         if not self.ver_h[a] or (a in events and not self.ver_s[events[a]]))
        return TreeEstimate(
            method=method, peak_idx=np.array(loc, dtype=np.int64), peak_h=self.h[peaks],
            peak_pers=self.h[peaks] - death[peaks], M=tb.merge_heights(peaks), labels=labels,
            cost_units=self.cost, times=dict(self.tm.t),
            info={"n_vertices": self.m, "n_basins": len(self.h), "n_adj": len(self.s),
                  "n_refined": self.n_refined, "frac_refined": self.n_refined / self.m,
                  "iterations": self.iterations, "stop": getattr(self, "stop_reason", "stage_a"),
                  "n_unverified_retained": unverified, "k0": self.k0, "K": self.K, "m": self.m})

    def vertex_estimate(self, tau, fidelity, method, with_labels=True) -> TreeEstimate:
        """Vertex-level tree on the subsample graph with a single fidelity:
        0 = Stage A only, 1 = every subsample vertex refined (global refinement)."""
        if fidelity == 1:
            self.refine_vertices(np.arange(self.m))
            f = self.f1
        else:
            f = self.f0
        t0 = time.perf_counter()
        mt = merge_tree(f, self.edges)
        self.tm.add("refine_tree" if fidelity else "graph_tree", t0)
        est = estimate_from_vertex_tree(method, mt, tau, self.S,
                                        label_fn=self.label_all if with_labels else None)
        est.cost_units = self.cost
        est.times = dict(self.tm.t)
        est.info = {"n_vertices": self.m, "n_refined": self.n_refined,
                    "frac_refined": self.n_refined / self.m, "k0": self.k0, "K": self.K, "m": self.m}
        return est
