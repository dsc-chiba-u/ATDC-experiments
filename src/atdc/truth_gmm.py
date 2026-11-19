"""Ground truth for isotropic Gaussian mixtures in any dimension (Phase A).

``truth.compute_truth`` discretises the true tree on a kNN graph of sample
points.  That fails when the structure fills many dimensions: samples sit on
shells around the component means and no sample is near a mode or a saddle.
For mixtures of ISOTROPIC Gaussians the critical points can be reached
without sampling:

* modes: fixed points of modal EM (Li, Ray & Lindsay 2007) started from the
  component means;
* saddles: for two isotropic components every critical point of their mixture
  lies on the segment between the means (ridgeline of Ray & Lindsay 2005 with
  covariances proportional to the identity).  The reference graph is therefore
  a skeleton: the modes plus a fine grid on the segment between every pair of
  modes, with the TRUE log-density of the whole mixture as vertex value.  With
  more than two components a saddle may lie off the segments (inside the
  convex hull of the means); the skeleton then gives a lower bound of the
  merge level.  The data sets used keep the components separated enough for
  this to be negligible (checked against ``compute_truth`` in 2-D,
  tests/test_truth_gmm.py);
* labels: every data point is moved to its mode by modal EM; its true cluster
  is the pruned-tree label of that mode.

The object has the interface of ``truth.Truth`` used by ``metrics``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .mergetree import merge_tree


def modal_em(dist, X, max_iter=500, tol=1e-7, chunk=5000):
    """Ascend the mixture density from every row of X (isotropic components)."""
    X = np.array(X, dtype=float)
    prec = 1.0 / dist.var[:, 0]                              # (C,)
    lognorm = np.log(dist.weights) - 0.5 * dist.dim * np.log(2 * np.pi * dist.var[:, 0])
    m2 = np.einsum("ij,ij->i", dist.means, dist.means)
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk]
        active = np.arange(len(x))
        for _ in range(max_iter):
            xa = x[active]
            d2 = np.einsum("ij,ij->i", xa, xa)[:, None] - 2 * xa @ dist.means.T + m2[None, :]
            lp = lognorm[None, :] - 0.5 * d2 * prec[None, :]
            w = np.exp(lp - logsumexp(lp, axis=1, keepdims=True)) * prec[None, :]
            new = (w @ dist.means) / w.sum(axis=1, keepdims=True)
            moved = np.linalg.norm(new - xa, axis=1)
            x[active] = new
            active = active[moved > tol]
            if active.size == 0:
                break
        X[s:s + chunk] = x
    return X


class TruthGMM:
    def __init__(self, dist, X, grid=101, merge_tol=1e-3):
        self.dist = dist
        self.n_data = len(X)
        top = modal_em(dist, dist.means)
        modes = []                                           # distinct modes
        for t in top:
            if not any(np.linalg.norm(t - q) < merge_tol for q in modes):
                modes.append(t)
        modes = np.array(modes)
        self.mode_pts = modes
        nm = len(modes)
        pts, edges = [modes], []
        nxt = nm
        ts = np.linspace(0, 1, grid + 2)[1:-1]
        for a in range(nm):
            for b in range(a + 1, nm):
                seg = modes[a][None, :] + ts[:, None] * (modes[b] - modes[a])[None, :]
                ids = np.arange(nxt, nxt + grid)
                pts.append(seg)
                edges.append(np.stack([np.r_[a, ids], np.r_[ids, b]], axis=1))
                nxt += grid
        self.ref = np.concatenate(pts)
        self.f = dist.logpdf(self.ref)
        e = np.concatenate(edges) if edges else np.empty((0, 2), dtype=np.int64)
        if len(self.ref) > nm:
            # A segment can cross the flank of a third mode and have a local
            # maximum that is not a mode.  Every skeleton point is joined to
            # the mode it ascends to: the ascent path lies in the superlevel
            # set at the point's own level, so the edge (weight = density of
            # the point) is exact.  All peaks of the skeleton are then modes.
            up = self._nearest_mode(modal_em(dist, self.ref[nm:]), modes)
            e = np.concatenate([e, np.stack([np.arange(nm, len(self.ref)), up], axis=1)])
        self.tree = merge_tree(self.f, e)
        self.shift = 0.0
        # data point -> mode
        end = modal_em(dist, X)
        self.point_mode = self._nearest_mode(end, modes)
        self.unconverged = float(np.mean(np.linalg.norm(end - modes[self.point_mode], axis=1) > 1e-2))

    @staticmethod
    def _nearest_mode(P, modes):
        d2 = (np.einsum("ij,ij->i", P, P)[:, None] - 2 * P @ modes.T
              + np.einsum("ij,ij->i", modes, modes)[None, :])
        return np.argmin(d2, axis=1)

    def modes(self, tau):
        p = self.tree.peaks(tau)
        death = self.tree.death_level()
        return p, self.f[p], self.f[p] - death[p]

    def labels(self, tau):
        p = self.tree.peaks(tau)
        pos = -np.ones(self.tree.n, dtype=np.int64)
        pos[p] = np.arange(len(p))
        return pos[self.tree.labels(tau)][self.point_mode]

    def merge_matrix(self, tau):
        return self.tree.merge_heights(self.tree.peaks(tau))
