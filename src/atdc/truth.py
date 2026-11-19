"""Ground-truth merge tree of a known density.

The true tree is computed with the same (tested) merge-tree routine on a
dense reference graph whose vertex values are the TRUE log-density: the data
points plus a large extra sample, connected by a kNN graph, with every edge
weight capped by the true log-density at the edge midpoint (so that an edge
cannot jump over a valley).  This is a discretisation of the true tree; its
resolution is checked against an analytic saddle in tests/test_truth.py.

For distributions with noise coordinates the structure lives in the first
``core_dim`` coordinates: the true density is the product of the core density
and a centred Gaussian, so modes and saddles sit at 0 in the noise
coordinates, the tree is that of the core density, and all heights are shifted
by the log-density of the noise at 0.  Ground-truth labels are the basins of
the core density.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mergetree import MergeTree, merge_tree, unique_undirected_edges


@dataclass
class Truth:
    ref: np.ndarray        # reference points (core coordinates); first n_data are the data
    f: np.ndarray          # true log-density (full-dimensional) at the reference points
    tree: MergeTree
    n_data: int
    shift: float

    def modes(self, tau):
        """Reference-vertex ids, heights and prominences of true modes with
        log-prominence >= tau (the global mode has prominence inf)."""
        p = self.tree.peaks(tau)
        death = self.tree.death_level()
        return p, self.f[p], self.f[p] - death[p]

    def labels(self, tau):
        """True cluster label (position in modes(tau)) of every data point."""
        p = self.tree.peaks(tau)
        pos = -np.ones(self.tree.n, dtype=np.int64)
        pos[p] = np.arange(len(p))
        return pos[self.tree.labels(tau)][:self.n_data]

    def merge_matrix(self, tau):
        return self.tree.merge_heights(self.tree.peaks(tau))


def compute_truth(dist, X, n_extra=40000, k=16, seed=12345) -> Truth:
    rng = np.random.default_rng(seed)
    core = dist.core()
    c = dist.core_dim
    extra = [core.sample(n_extra, rng)]
    means = getattr(core, "means", None)
    if means is not None:
        extra.append(means)
    ref = np.concatenate([np.asarray(X)[:, :c]] + extra)
    shift = dist.noise_log_const()
    f = core.logpdf(ref) + shift
    _, ind = cKDTree(ref).query(ref, k + 1)
    edges = unique_undirected_edges(np.repeat(np.arange(len(ref)), k), ind[:, 1:].ravel())
    mid = 0.5 * (ref[edges[:, 0]] + ref[edges[:, 1]])
    w = np.minimum(np.minimum(f[edges[:, 0]], f[edges[:, 1]]), core.logpdf(mid) + shift)
    return Truth(ref=ref, f=f, tree=merge_tree(f, edges, w), n_data=len(X), shift=shift)
