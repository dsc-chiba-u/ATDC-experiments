"""ATDC v2: uniform refinement of a small subsample, then (i) adaptive graph
refinement at the events that matter and (ii) held-out verification of those
events on an independent part of the data.

Differences from v1 (core.py), each aimed at a measured weakness
(docs/refinement_v2.md):

* no coarse stage and no mixed fidelities: every vertex carries a density
  estimated from the (tree part of the) full data from the start;
* the graph is not frozen: vertices are added at peaks (mean-shift ascent in
  the full data) and long edges on the path between two peaks are verified by
  evaluating the density at interior points;
* optional sample splitting: the tree, the ascent and the edge checks use
  only the part H1 of the data; the prominence of every candidate event is
  then re-estimated at the selected peak and saddle locations from the
  disjoint part H2, so the estimate is free of the selection bias of
  maxima / minima over many noisy values.

Density access goes through an oracle object so that the graph logic can be
tested with the true density as a positive control.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order
from scipy.spatial import cKDTree

from .core import Timer, TreeEstimate
from .density import knn_log_density, knn_log_density_se
from .mergetree import merge_tree


class KNNOracle:
    """kNN log-density from the data.  With ``rho`` the data are split at
    random into H1 (fraction rho; used for everything that selects) and H2
    (the rest; used only by :meth:`held`).  Both use the same expected ball
    radius: K1 = round(rho K) neighbours in H1, K2 = K - K1 in H2."""

    def __init__(self, X, K, rho=None, seed=0, tm=None):
        self.X = np.ascontiguousarray(X, dtype=float)
        self.n, self.d = self.X.shape
        self.K = int(K)
        self.tm = tm or Timer()
        self.cost = 0.0
        rng = np.random.default_rng([int(seed), 0x51ED])   # not default_rng(seed): see tests
        t0 = time.perf_counter()
        if rho is None:
            self.pool = np.arange(self.n)            # candidate vertices
            self.K1, self.K2 = self.K, 0
            self.X1 = self.X
            self.i1 = cKDTree(self.X1)
            self.i2 = None
        else:
            in1 = rng.random(self.n) < rho
            self.pool = np.flatnonzero(in1)
            self.K1 = max(4, int(round(rho * self.K)))
            self.K2 = max(4, self.K - self.K1)
            self.X1 = self.X[in1]
            self.X2 = self.X[~in1]
            self.i1 = cKDTree(self.X1)
            self.i2 = cKDTree(self.X2)
        self.pos1 = -np.ones(self.n, dtype=np.int64)  # data index -> row of X1
        self.pos1[self.pool] = np.arange(len(self.pool))
        self.n1 = len(self.X1)
        self.se = knn_log_density_se(self.K1)
        self.se_held = knn_log_density_se(self.K2) if self.K2 else np.nan
        self.tm.add("index_full", t0)

    def tree_density(self, idx):
        """Leave-one-out density of data points (indices into X, all in H1).
        Returns (log-density, radius, neighbour data indices)."""
        t0 = time.perf_counter()
        dist, ind = self.i1.query(self.X[idx], self.K1 + 1)
        self.tm.add("query", t0)
        self.cost += len(idx) * self.K1
        return (knn_log_density(dist[:, self.K1], self.K1, self.n1, self.d), dist[:, self.K1],
                self.pool[ind[:, 1:]])

    def tree_density_at(self, pts):
        """Density at arbitrary locations (not data points)."""
        t0 = time.perf_counter()
        dist, _ = self.i1.query(pts, [self.K1])
        self.tm.add("query", t0)
        self.cost += len(pts) * self.K1
        return knn_log_density(dist[:, 0], self.K1, self.n1, self.d)

    def held(self, pts):
        """Density at locations, from H2 only (no point of H2 is a vertex)."""
        t0 = time.perf_counter()
        dist, _ = self.i2.query(pts, [self.K2])
        self.tm.add("query_held", t0)
        self.cost += len(pts) * self.K2
        return knn_log_density(dist[:, 0], self.K2, len(self.X2), self.d)


class TrueOracle(KNNOracle):
    """Positive control: the true log-density instead of an estimate.  The
    neighbour lists (used only for the mean-shift ascent) still come from the
    data."""

    def __init__(self, X, dist_obj, K=30, seed=0, tm=None):
        super().__init__(X, K, rho=None, seed=seed, tm=tm)
        self.dist_obj = dist_obj
        self.se = 0.0

    def tree_density(self, idx):
        _, r, nb = super().tree_density(idx)
        return self.dist_obj.logpdf(self.X[idx]), r, nb

    def tree_density_at(self, pts):
        return self.dist_obj.logpdf(pts)

    def held(self, pts):
        return self.dist_obj.logpdf(pts)


class ATDC2:
    BIG = 1 << 31

    def __init__(self, X, K=100, m0=None, q0=0.01, k_graph=10, rho=None, seed=0,
                 oracle=None, init_idx=None):
        self.tm = Timer()
        self.o = oracle if oracle is not None else KNNOracle(X, K, rho=rho, seed=seed, tm=self.tm)
        self.o.tm = self.tm
        self.X = self.o.X
        self.n, self.d = self.X.shape
        self.k_graph = int(k_graph)
        rng = np.random.default_rng([int(seed), 0x1217])
        if init_idx is None:
            m0 = int(m0) if m0 is not None else int(round(q0 * self.n))
            m0 = min(max(m0, self.k_graph + 2), len(self.o.pool))
            init_idx = rng.choice(self.o.pool, size=m0, replace=False)
        self.vid = {}                 # data index -> vertex id
        self.idx = []                 # vertex id -> data index
        self.f = []                   # tree density
        self.r = []                   # bandwidth radius
        self.nb = []                  # K nearest data points (for the ascent)
        self.cap = {}                 # edge key -> verified cap on the edge weight
        self.cap_at = {}              # edge key -> location of the minimum
        self.keys = np.empty(0, dtype=np.int64)   # sorted edge keys u * BIG + v, u < v
        self.ascended = set()
        self.n_ascent = self.n_edge_checks = self.n_inserted = 0
        self.rounds = 0
        self._add_vertices(np.asarray(init_idx))
        self.m0 = len(self.idx)
        self._rebuild_graph()

    # ------------------------------------------------------------ vertices
    def _add_vertices(self, data_idx):
        data_idx = np.array([i for i in dict.fromkeys(np.asarray(data_idx).tolist())
                             if i not in self.vid], dtype=np.int64)
        if data_idx.size == 0:
            return
        f, r, nb = self.o.tree_density(data_idx)
        for i, fi, ri, ni in zip(data_idx.tolist(), f.tolist(), r.tolist(), nb):
            self.vid[i] = len(self.idx)
            self.idx.append(i); self.f.append(fi); self.r.append(ri); self.nb.append(ni)

    def _rebuild_graph(self):
        t0 = time.perf_counter()
        V = self.X[np.array(self.idx)]
        k = min(self.k_graph, len(V) - 1)
        _, ind = cKDTree(V).query(V, k + 1)
        u = np.repeat(np.arange(len(V)), k)
        v = ind[:, 1:].ravel()
        new = np.minimum(u, v) * self.BIG + np.maximum(u, v)
        # edges are never removed: a verified edge stays valid information
        self.keys = np.unique(np.concatenate([self.keys, new]))
        self.edge_set = np.stack([self.keys // self.BIG, self.keys % self.BIG], axis=1)
        self.tm.add("graph", t0)

    def _tree(self):
        t0 = time.perf_counter()
        f = np.array(self.f)
        e = self.edge_set
        w = np.minimum(f[e[:, 0]], f[e[:, 1]])
        if self.cap:
            ck = np.fromiter(self.cap.keys(), np.int64, len(self.cap))
            cv = np.fromiter(self.cap.values(), float, len(self.cap))
            pos = np.searchsorted(self.keys, ck)
            w[pos] = np.minimum(w[pos], cv)
        t = merge_tree(f, e, w)
        self.tm.add("tree", t0)
        return t

    # -------------------------------------------------------------- ascent
    def _ascend(self, peaks, max_steps=8, n_cand=6):
        """Hill climbing in the data from the given peak vertices.

        Candidates for the next position are a few of the K neighbours of the
        current point: the one furthest along the mean-shift direction
        (centroid of the neighbours minus the point), the one closest to the
        centroid, and n_cand random ones from the outer half of the
        neighbourhood.  Their densities are evaluated and the best becomes the
        new position if it is higher.  (The mean-shift direction alone is too
        noisy: its length is of the order of the sampling error of the
        centroid.)  Visited candidates become vertices.  Batched over peaks:
        one density query per step."""
        cur = [p for p in dict.fromkeys(peaks) if p not in self.ascended]
        rng = np.random.default_rng([len(self.idx), 0xA5C])
        n_before = len(self.idx)
        for _ in range(max_steps):
            if not cur:
                break
            cands = []
            for p in cur:
                nbi = self.nb[p]
                x = self.X[self.idx[p]]
                D = self.X[nbi] - x
                c = D.mean(axis=0)
                pick = {int(nbi[np.argmax(D @ c)]), int(nbi[np.argmin(np.sum((D - c) ** 2, axis=1))])}
                outer = nbi[len(nbi) // 2:]                 # neighbours are sorted by distance
                pick.update(int(j) for j in rng.choice(outer, size=min(n_cand, len(outer)), replace=False))
                cands.append(list(pick))
            self._add_vertices(np.array([j for c in cands for j in c]))
            nxt = []
            for p, c in zip(cur, cands):
                self.ascended.add(p)
                q = max((self.vid[j] for j in c), key=lambda v: self.f[v])
                if self.f[q] > self.f[p] and q not in self.ascended:
                    nxt.append(q); self.n_ascent += 1
            cur = list(dict.fromkeys(nxt))
        self.ascended.update(cur)
        self.n_inserted += len(self.idx) - n_before
        return len(self.idx) > n_before

    # ------------------------------------------------------- edge checking
    def _check(self, todo, max_depth=4):
        """todo: edge key -> (number of interior points, resolution).

        Evaluates the density at equally spaced interior points of each edge
        and stores the minimum as a cap on the edge weight.  While the spacing
        of the points is coarser than ``resolution`` (a fraction of the edge),
        the neighbourhood of the lowest point is bisected (at most max_depth
        times), so that the minimum along a very long edge is located without
        sampling the whole edge at bandwidth resolution."""
        if not todo:
            return False
        V = self.X[np.array(self.idx)]
        keys = list(todo)
        P0 = np.array([V[k // self.BIG] for k in keys]); P1 = np.array([V[k % self.BIG] for k in keys])
        best = np.full(len(keys), np.inf)
        best_t = np.zeros(len(keys))
        h = np.array([1.0 / (todo[k][0] + 1) for k in keys])
        res = np.array([todo[k][1] for k in keys])
        ei, ts = [], []
        for i, k in enumerate(keys):
            n_int = todo[k][0]
            ei += [i] * n_int
            ts += (np.arange(1, n_int + 1) / (n_int + 1)).tolist()
        for depth in range(max_depth + 1):
            if not ei:
                break
            ei_a, ts_a = np.array(ei), np.array(ts)
            pts = P0[ei_a] + ts_a[:, None] * (P1[ei_a] - P0[ei_a])
            val = self.o.tree_density_at(pts)
            self.n_edge_checks += len(pts)
            for i, t, fv in zip(ei, ts, val.tolist()):
                if fv < best[i]:
                    best[i] = fv; best_t[i] = t
            ei, ts = [], []
            for i in np.flatnonzero(h > res).tolist():
                h[i] *= 0.5
                for t in (best_t[i] - h[i], best_t[i] + h[i]):
                    if 0.0 < t < 1.0:
                        ei.append(i); ts.append(t)
        for i, k in enumerate(keys):
            self.cap[k] = best[i]
            self.cap_at[k] = P0[i] + best_t[i] * (P1[i] - P0[i])
        return True

    def _edge_state(self, lam):
        """Per edge: weight (with caps), length, long-and-unverified flag."""
        e = self.edge_set
        f = np.array(self.f); r = np.array(self.r)
        V = self.X[np.array(self.idx)]
        L = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        w = np.minimum(f[e[:, 0]], f[e[:, 1]])
        ver = np.zeros(len(e), dtype=bool)
        if self.cap:
            ck = np.fromiter(self.cap.keys(), np.int64, len(self.cap))
            cv = np.fromiter(self.cap.values(), float, len(self.cap))
            pos = np.searchsorted(self.keys, ck)
            w[pos] = np.minimum(w[pos], cv); ver[pos] = True
        rr = np.maximum(r[e[:, 0]], r[e[:, 1]])
        return f, w, L, rr, (L > lam * rr) & ~ver

    def _verify_edges(self, tree, sel, lam=1.0, band=0.3, max_int=3):
        """Lazy verification of the merges that carry the selected events.

        The merge level of peak a with the peak b that absorbs it is the
        bottleneck of the path between them in the maximum spanning forest.
        Any long, unverified edge on that path may jump a valley -- not only
        the bottleneck edge: a slope vertex can be attached to the other
        cluster by a long edge and to its own cluster by a short one.  For
        every such path edge, all long unverified edges that leave the
        component it attached (the cut of that merge), down to ``band`` below
        its level, are verified in one batch.  Rounds are needed because the
        forest changes.  When no long unverified edge remains on the paths,
        the merge levels of the selected events are exact for the verified
        weights (lazy Kruskal).  An edge is long when it exceeds lam bandwidth
        radii: a shorter edge cannot hide a dip that the estimator, at this
        bandwidth, could resolve."""
        f, w, L, rr, long_ = self._edge_state(lam)
        if not long_.any() or len(sel) == 0:
            return False
        e = self.edge_set
        m = len(f)
        tu, tv = tree.edge_uv[:, 0], tree.edge_uv[:, 1]
        tlong = long_[tree.edge]                       # per merge: is its edge long & unverified
        if not tlong.any():
            return False
        forest = coo_matrix((np.arange(1, len(tu) + 1), (tu, tv)), shape=(m, m)).tocsr()
        forest = forest + forest.T                     # value = merge index + 1
        J = set()
        for i in sel.tolist():
            a_, b_ = int(tree.dying[i]), int(tree.surviving[i])
            _, pred = breadth_first_order(forest, a_, directed=False, return_predecessors=True)
            x = b_
            while x != a_ and pred[x] >= 0:
                j = int(forest[x, pred[x]]) - 1
                if tlong[j]:
                    J.add(j)
                x = int(pred[x])
        if not J:
            return False
        members = {}
        need = np.zeros(len(e), dtype=bool)
        inC = np.zeros(m, dtype=bool)
        for j, (d, s_, lv) in enumerate(zip(tree.dying.tolist(), tree.surviving.tolist(),
                                            tree.level.tolist())):
            md = members.pop(d, None) or [d]
            if j in J:
                inC[:] = False
                inC[md] = True
                need |= (inC[e[:, 0]] ^ inC[e[:, 1]]) & long_ & (w >= lv - band)
            ms = members.get(s_)
            if ms is None:
                members[s_] = [s_] + md
            else:
                ms.extend(md)
        todo = {int(self.keys[i]): (int(min(max(np.ceil(L[i] / (lam * rr[i])) - 1, 1), max_int)),
                                    lam * rr[i] / L[i])
                for i in np.flatnonzero(need).tolist()}
        return self._check(todo)

    def _verify_hidden_peaks(self, lam_a=2.0, max_int=3):
        """Look for peaks hidden by the sparsity of the graph.

        A vertex whose only higher neighbours are reached through long,
        unverified edges may be the top of a cluster that the graph joins to
        another one by jumping a valley; such a vertex has zero prominence
        and no event points to it.  For every vertex that is a local maximum
        among its short (or verified) edges, the long edge to its highest
        neighbour is verified.  One call checks one edge per such vertex."""
        e = self.edge_set
        if len(e) == 0:
            return False
        f, w, L, rr, long_ = self._edge_state(lam_a)
        rr = lam_a * rr
        src = np.r_[e[:, 0], e[:, 1]]; dst = np.r_[e[:, 1], e[:, 0]]
        ww = np.r_[w, w]; lg = np.r_[long_, long_]; ei = np.r_[np.arange(len(e)), np.arange(len(e))]
        up = ((f[dst] > f[src]) | ((f[dst] == f[src]) & (dst < src))) & (ww >= f[src] - 1e-12)
        has_short = np.zeros(len(f), dtype=bool)
        has_short[src[up & ~lg]] = True
        c = up & lg & ~has_short[src]
        if not c.any():
            return False
        s_, d_, e_ = src[c], dst[c], ei[c]
        o = np.lexsort((-f[d_], s_))
        first = np.r_[True, s_[o][1:] != s_[o][:-1]]
        todo = {}
        for i in e_[o][first].tolist():
            todo[int(self.keys[i])] = (int(min(max(np.ceil(L[i] / (rr[i] / lam_a)) - 1, 1), max_int)),
                                       rr[i] / lam_a / L[i])
        self.n_hidden_checks = getattr(self, "n_hidden_checks", 0) + len(todo)
        return self._check(todo)

    def _densify_saddles(self, tree, sel, k_ins=6, lam_s=0.5):
        """Add vertices where an event attains its merge level.

        A sparse graph has no vertex at the pass, so the merge level is too
        low.  For every selected event whose critical edge is longer than
        lam_s bandwidth radii, the k_ins data points closest to the location
        of the merge level (among the neighbours of the two endpoints) become
        vertices.  Below about half a bandwidth radius further vertices only
        add noise maxima, hence the stop."""
        if not hasattr(self, "densified"):
            self.densified = set()
        V = self.X
        want = []
        for i in sel.tolist():
            u, v = (int(x) for x in tree.edge_uv[i])
            key = min(u, v) * self.BIG + max(u, v)
            if key in self.densified:
                continue
            self.densified.add(key)
            xu, xv = V[self.idx[u]], V[self.idx[v]]
            if np.linalg.norm(xu - xv) <= lam_s * max(self.r[u], self.r[v]):
                continue
            lo = u if self.f[u] <= self.f[v] else v
            capped = key in self.cap and self.cap[key] < self.f[lo]
            mid = self.cap_at[key] if capped else 0.5 * (xu + xv)
            cand = np.unique(np.concatenate([self.nb[u], self.nb[v]]))
            cand = cand[[c not in self.vid for c in cand.tolist()]]
            if cand.size:
                d = np.sum((V[cand] - mid) ** 2, axis=1)
                want.append(cand[np.argsort(d)[:k_ins]])
        if not want:
            return False
        n_before = len(self.idx)
        self._add_vertices(np.concatenate(want))
        self.n_inserted += len(self.idx) - n_before
        self.n_saddle_ins = getattr(self, "n_saddle_ins", 0) + len(self.idx) - n_before
        return len(self.idx) > n_before

    # ---------------------------------------------------------------- main
    def refine(self, tau, margin=0.25, ascent=True, edges=True, saddles=True, hidden=True,
               max_rounds=4, max_events=40, lam=1.0, asc_margin=None, k_ins=4, max_steps=3,
               hidden_rounds=4, n_cand=4):
        """Adaptive graph refinement.

        ascent: mean-shift ascent from local maxima with prominence
                >= tau - asc_margin (default: asc_margin = margin).
        edges:  lazy verification of long edges on the forest paths of the
                events with prominence >= tau - margin, repeated while it
                changes the tree (at most max_rounds times).
        saddles: new vertices at the merge location of the same events.
        hidden: verification of long ascent edges that may hide a peak.
        """
        self.tau, self.margin = tau, margin
        if hidden:
            for _ in range(hidden_rounds):
                if not self._verify_hidden_peaks():
                    break
        tree = self._tree()
        if ascent:
            pers = tree.persistence
            lo = max(tau - (margin if asc_margin is None else asc_margin), 1e-9)
            sel = np.flatnonzero(pers >= lo)
            peaks = tree.dying[sel].tolist() + tree.roots.tolist()
            if self._ascend(peaks, max_steps=max_steps, n_cand=n_cand):
                self._rebuild_graph()
                if hidden:
                    self._verify_hidden_peaks()
                tree = self._tree()
        if edges or saddles:
            for _ in range(max_rounds):
                pers = tree.persistence
                sel = np.flatnonzero((pers >= tau - margin) & (pers > 1e-9))
                sel = sel[np.argsort(-pers[sel])][:max_events]
                changed = edges and self._verify_edges(tree, sel, lam=lam)
                if saddles and self._densify_saddles(tree, sel, k_ins=k_ins):
                    self._rebuild_graph()
                    changed = True
                if not changed:
                    break
                self.rounds += 1
                tree = self._tree()
        self._last_tree = tree
        return self

    def _saddle_location(self, tree, i):
        """Location where event i attains its merge level."""
        u, v = (int(x) for x in tree.edge_uv[i])
        key = min(u, v) * self.BIG + max(u, v)
        lo = u if self.f[u] <= self.f[v] else v
        if key in self.cap and self.cap[key] < self.f[lo]:
            return self.cap_at[key]
        return self.X[self.idx[lo]]

    def estimate(self, tau, method="atdc2", held_out=None, margin=None, with_labels=True,
                 z=0.0) -> TreeEstimate:
        """Prune and label.

        Without sample splitting: keep peaks with (in-sample) prominence >= tau.
        With sample splitting (``held_out``, default whenever the data were
        split): every event with in-sample prominence >= tau - margin is
        re-estimated on H2 at the selected peak and saddle locations and is
        retained iff that held-out prominence is >= tau + z * se.  The
        reported peak heights and merge levels of tested events are the
        held-out values.
        """
        tree = getattr(self, "_last_tree", None) or self._tree()
        self._last_tree = None
        held_out = (self.o.i2 is not None) if held_out is None else held_out
        margin = getattr(self, "margin", 0.25) if margin is None else margin
        pers = tree.persistence
        keep = np.zeros(tree.n, dtype=bool)
        info = {}
        level = tree.level.copy()
        fh = tree.f.copy()
        idx = np.array(self.idx)
        if held_out:
            cand = np.flatnonzero((pers >= tau - margin) & (pers > 1e-9))
            P = self.X[idx[np.r_[tree.dying[cand], tree.roots]]]
            S = np.array([self._saddle_location(tree, i) for i in cand]).reshape(-1, self.d)
            h = self.o.held(np.concatenate([P, S]))
            nc = len(cand)
            hp, hs = h[:nc], h[len(P):]
            r2 = hp - hs
            se2 = np.sqrt(2.0) * self.o.se_held
            keep[tree.dying[cand[r2 >= tau + z * se2]]] = True
            fh[tree.dying[cand]] = hp
            fh[tree.roots] = h[nc:len(P)]
            level[cand] = np.minimum(hs, hp)
            info.update(n_tested=int(nc), n_in_sample=int(np.sum(pers >= tau)),
                        held_r=r2.tolist(), in_sample_r=pers[cand].tolist())
        else:
            keep[tree.dying[pers >= tau]] = True
        keep[tree.roots] = True
        peaks = np.flatnonzero(keep)
        death = np.full(tree.n, -np.inf)
        death[tree.dying] = level
        t0 = time.perf_counter()
        labels = None
        if with_labels:
            pos = -np.ones(tree.n, dtype=np.int64)
            pos[peaks] = np.arange(len(peaks))
            vlab = pos[tree.labels_keep(keep)]
            _, nn = cKDTree(self.X[idx]).query(self.X, 1)
            labels = vlab[nn]
            self.o.cost += self.n
            self.tm.add("label", t0)
        # merge heights between retained peaks, with held-out levels where tested
        lv_orig = tree.level
        tree.level = level
        M = tree.merge_heights(peaks)
        tree.level = lv_orig
        M[np.arange(len(peaks)), np.arange(len(peaks))] = fh[peaks]
        M = np.minimum(M, np.minimum.outer(fh[peaks], fh[peaks]))
        info.update(n_vertices=len(self.idx), m0=self.m0, n_edges=len(self.keys), rounds=self.rounds,
                    n_ascent=self.n_ascent, n_edge_checks=self.n_edge_checks,
                    n_inserted=self.n_inserted, n_saddle_ins=getattr(self, "n_saddle_ins", 0),
                    n_hidden_checks=getattr(self, "n_hidden_checks", 0),
                    K1=self.o.K1, K2=self.o.K2)
        return TreeEstimate(method=method, peak_idx=idx[peaks], peak_h=fh[peaks],
                            peak_pers=fh[peaks] - death[peaks], M=M, labels=labels,
                            cost_units=self.o.cost, times=dict(self.tm.t), info=info)
