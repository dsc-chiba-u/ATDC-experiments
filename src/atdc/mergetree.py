"""Merge trees (0-dimensional superlevel-set persistence) on weighted graphs.

Definitions used throughout the repository
------------------------------------------
A *field* is a graph G = (V, E) with a value f(v) on every vertex and a
weight w(e) <= min(f(u), f(v)) on every edge e = (u, v).  By default
w(e) = min(f(u), f(v)) (the usual vertex-based upper-star filtration); a
smaller w(e) expresses that the density dips below both endpoints somewhere
along the edge.

Superlevel graph at level t: vertices with f >= t and edges with w >= t.

* merge height  m(u, v) = sup{t : u and v are connected in the superlevel
  graph at level t} = max over u-v paths of the minimum of the vertex values
  and edge weights along the path.  m(u, u) = f(u).
* peak: a connected component is born at its highest vertex.
* elder rule: when two components meet at level t, the one with the lower
  birth value dies; its persistence (log-prominence when f is a log-density)
  is birth - t.

None of this is new: it is the 0-dimensional persistence of the superlevel
filtration, the same object ToMATo computes.  The module exists so that the
vertex-level tree and the tree of a basin adjacency graph can be computed and
tested with one small, auditable routine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def edge_weights(f: np.ndarray, edges: np.ndarray, cap: np.ndarray | None = None) -> np.ndarray:
    """w(e) = min(f(u), f(v)), optionally capped from above by ``cap``."""
    w = np.minimum(f[edges[:, 0]], f[edges[:, 1]])
    if cap is not None:
        w = np.minimum(w, cap)
    return w


def unique_undirected_edges(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Symmetrise and deduplicate a directed edge list; drops self loops."""
    a = np.minimum(src, dst).astype(np.int64)
    b = np.maximum(src, dst).astype(np.int64)
    keep = a != b
    a, b = a[keep], b[keep]
    if a.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    n = int(b.max()) + 1
    key = np.unique(a * n + b)
    return np.stack([key // n, key % n], axis=1)


class _UF:
    __slots__ = ("p",)

    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r


@dataclass
class MergeTree:
    """Result of :func:`merge_tree`.

    Every vertex starts as its own component (born at f(v)).  ``dying[i]`` is
    the peak (vertex id) of the component that dies in the i-th merge,
    ``surviving[i]`` the peak of the component that absorbs it, ``level[i]``
    the merge level and ``edge[i]`` the index (into the input edge list) of
    the edge realising the merge.  Merges are listed in decreasing level.
    ``roots`` are the peaks that never die (one per connected component).
    """

    n: int
    f: np.ndarray
    dying: np.ndarray
    surviving: np.ndarray
    level: np.ndarray
    edge: np.ndarray
    edge_uv: np.ndarray
    roots: np.ndarray

    @property
    def persistence(self) -> np.ndarray:
        return self.f[self.dying] - self.level

    def peaks(self, tau: float) -> np.ndarray:
        """Peaks with persistence >= tau, plus the roots (infinite persistence)."""
        keep = self.dying[self.persistence >= tau]
        return np.concatenate([self.roots, keep])

    def death_level(self) -> np.ndarray:
        """Level at which each vertex's own component dies (-inf for roots)."""
        d = np.full(self.n, -np.inf)
        d[self.dying] = self.level
        return d

    def labels(self, tau: float) -> np.ndarray:
        """Label every vertex by the retained peak it belongs to (ToMATo's rule).

        The merges are replayed in decreasing level.  Two components are
        united only if the lower of their two peaks has prominence < tau at
        that level; otherwise both are kept and continue to collect lower
        vertices separately.  Replaying only the spanning merges is
        sufficient: an edge outside the spanning forest joins two components
        at a level no higher than a refused merge on the tree path between
        them, so it would be refused as well.
        """
        uf = _UF(self.n)
        peak = list(range(self.n))
        fl = self.f.tolist()
        for (u, v), lv in zip(self.edge_uv.tolist(), self.level.tolist()):
            ra, rb = uf.find(u), uf.find(v)
            if ra == rb:
                continue
            pa, pb = peak[ra], peak[rb]
            if (fl[pa], -pa) < (fl[pb], -pb):
                ra, rb, pa, pb = rb, ra, pb, pa
            if fl[pb] - lv < tau:
                uf.p[rb] = ra
        return np.array([peak[uf.find(i)] for i in range(self.n)], dtype=np.int64)

    def labels_keep(self, keep: np.ndarray) -> np.ndarray:
        """Labels for an arbitrary set of retained peaks (boolean per vertex).

        Same replay as :meth:`labels`, but a merge is refused when the lower
        of the two peaks is in ``keep`` instead of when its prominence
        reaches a threshold.  Roots are always retained.  With
        ``keep[v] = persistence(v) >= tau`` this equals ``labels(tau)``.
        """
        uf = _UF(self.n)
        peak = list(range(self.n))
        fl = self.f.tolist()
        kp = np.asarray(keep, dtype=bool).tolist()
        for (u, v) in self.edge_uv.tolist():
            ra, rb = uf.find(u), uf.find(v)
            if ra == rb:
                continue
            pa, pb = peak[ra], peak[rb]
            if (fl[pa], -pa) < (fl[pb], -pb):
                ra, rb, pa, pb = rb, ra, pb, pa
            if not kp[pb]:
                uf.p[rb] = ra
        return np.array([peak[uf.find(i)] for i in range(self.n)], dtype=np.int64)

    def merge_heights(self, q: np.ndarray) -> np.ndarray:
        """Matrix M[i, j] = m(q[i], q[j]) for a small set of query vertices.

        -inf where the two vertices are in different connected components.
        """
        q = np.asarray(q, dtype=np.int64)
        nq = len(q)
        M = np.full((nq, nq), -np.inf)
        M[np.arange(nq), np.arange(nq)] = self.f[q]
        # every vertex is the peak of its own initial component, so merging
        # by peak id is the same as merging vertex sets
        members: dict[int, list[int]] = {}
        for i, v in enumerate(q):
            members.setdefault(int(v), []).append(i)
        for d, s, lv in zip(self.dying.tolist(), self.surviving.tolist(), self.level.tolist()):
            md = members.pop(d, None)
            if md is None:
                continue
            ms = members.get(s)
            if ms is None:
                members[s] = md
                continue
            for i in md:
                for j in ms:
                    M[i, j] = M[j, i] = lv
            ms.extend(md)
        for i in range(nq):
            for j in range(i + 1, nq):
                if q[i] == q[j]:
                    M[i, j] = M[j, i] = self.f[q[i]]
        return M


def merge_tree(f: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None,
               use_mst: bool = True) -> MergeTree:
    """Merge tree of a field by Kruskal's algorithm with the elder rule.

    Ties: among equal peaks the smaller vertex id is the elder.  With
    ``use_mst`` the edge list is first reduced (vectorised) to the ascent
    forest plus the best crossing edge between adjacent ascent basins, which
    has the same merge heights; the Python union-find loop runs only over
    those edges.
    """
    f = np.asarray(f, dtype=float)
    n = f.shape[0]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if w is None:
        w = edge_weights(f, edges)
    w = np.asarray(w, dtype=float)
    if np.any(w > np.minimum(f[edges[:, 0]], f[edges[:, 1]]) + 1e-12):
        raise ValueError("edge weight exceeds an endpoint value")
    eidx = np.arange(len(edges))
    if use_mst and len(edges) > n:
        # Reduce to a subgraph with identical merge heights: the ascent edge
        # of every non-peak vertex plus the best crossing edge of every pair
        # of adjacent basins (see tests: basin graph tree == vertex tree).
        parent, pedge = _ascent_parent(f, edges, w)
        root = _jump(parent)
        cu, cv = root[edges[:, 0]], root[edges[:, 1]]
        crossing = np.flatnonzero(cu != cv)
        key = np.minimum(cu[crossing], cv[crossing]) * n + np.maximum(cu[crossing], cv[crossing])
        o = np.lexsort((-w[crossing], key))
        ks = key[o]
        first = np.ones(len(o), dtype=bool)
        first[1:] = ks[1:] != ks[:-1]
        eidx = np.concatenate([pedge[pedge >= 0], crossing[o[first]]])
    # decreasing weight; ties: the edge whose higher endpoint is higher first,
    # so a vertex joins the component of its highest neighbour (ToMATo).
    hi = np.maximum(f[edges[eidx, 0]], f[edges[eidx, 1]])
    order = eidx[np.lexsort((-hi, -w[eidx]))]

    uf = _UF(n)
    peak = list(range(n))  # peak vertex of the component rooted at i
    fl = f.tolist()
    dying, surviving, level, edge = [], [], [], []
    eu = edges[:, 0].tolist()
    ev = edges[:, 1].tolist()
    wl = w.tolist()
    for e in order.tolist():
        ra, rb = uf.find(eu[e]), uf.find(ev[e])
        if ra == rb:
            continue
        pa, pb = peak[ra], peak[rb]
        # elder = higher peak, ties -> smaller id
        if (fl[pa], -pa) < (fl[pb], -pb):
            ra, rb, pa, pb = rb, ra, pb, pa
        uf.p[rb] = ra
        dying.append(pb)
        surviving.append(pa)
        level.append(wl[e])
        edge.append(e)
    roots = np.array(sorted({peak[uf.find(i)] for i in range(n)}), dtype=np.int64) if n else np.empty(0, np.int64)
    return MergeTree(n=n, f=f, dying=np.array(dying, dtype=np.int64),
                     surviving=np.array(surviving, dtype=np.int64),
                     level=np.array(level, dtype=float), edge=np.array(edge, dtype=np.int64),
                     edge_uv=edges[np.array(edge, dtype=np.int64)].reshape(-1, 2),
                     roots=roots)


def _jump(parent: np.ndarray) -> np.ndarray:
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return parent
        parent = nxt


def _ascent_parent(f, edges, w):
    """Highest strictly-higher neighbour of every vertex (itself for peaks)
    and the index of the edge leading to it (-1 for peaks)."""
    n = len(f)
    ne = len(edges)
    u = np.concatenate([edges[:, 0], edges[:, 1]])
    v = np.concatenate([edges[:, 1], edges[:, 0]])
    ei = np.concatenate([np.arange(ne), np.arange(ne)])
    ww = np.concatenate([w, w])
    with np.errstate(invalid="ignore"):
        ok = (ww >= f[u] - 1e-12) | (ww == f[u])  # edge not capped below u
        # strictly higher in the (f, -id) order
        ok &= (f[v] > f[u]) | ((f[v] == f[u]) & (v < u))
    u, v, ei = u[ok], v[ok], ei[ok]
    parent = np.arange(n)
    pedge = -np.ones(n, dtype=np.int64)
    if u.size:
        order = np.lexsort((v, -f[v], u))  # per u: max f, then min id
        u_s = u[order]
        first = np.ones(len(u_s), dtype=bool)
        first[1:] = u_s[1:] != u_s[:-1]
        sel = order[first]
        parent[u[sel]] = v[sel]
        pedge[u[sel]] = ei[sel]
    return parent, pedge


def ascent_basins(f: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Steepest-ascent basins: each vertex points to its highest neighbour.

    Only edges that are not capped below the lower endpoint are followed, so
    that the edges of an ascent path never have a weight below the starting
    vertex.  Ties are broken by vertex id (smaller id is higher), which makes
    a plateau a single basin.  Returns the peak id of every vertex.
    """
    f = np.asarray(f, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if w is None:
        w = edge_weights(f, edges)
    return _jump(_ascent_parent(f, edges, w)[0])


@dataclass
class BasinGraph:
    """Region adjacency graph of the ascent basins.

    ``peak[a]`` is the peak vertex of basin a, ``basin[v]`` the basin index of
    vertex v, ``adj`` the (A, 2) array of adjacent basin pairs (a < b) and
    ``cross[i]`` the indices (into the vertex-level edge list) of the edges
    crossing adjacency i, sorted by decreasing weight.
    """

    peak: np.ndarray
    basin: np.ndarray
    adj: np.ndarray
    cross: list
    saddle: np.ndarray  # max crossing-edge weight per adjacency


def basin_graph(f: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None) -> BasinGraph:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if w is None:
        w = edge_weights(f, edges)
    root = ascent_basins(f, edges, w)
    peak, basin = np.unique(root, return_inverse=True)
    a = basin[edges[:, 0]]
    b = basin[edges[:, 1]]
    crossing = np.flatnonzero(a != b)
    lo = np.minimum(a[crossing], b[crossing])
    hi = np.maximum(a[crossing], b[crossing])
    B = len(peak)
    key = lo * B + hi
    order = np.lexsort((-w[crossing], key))
    key_s = key[order]
    cr_s = crossing[order]
    if key_s.size:
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        ends = np.r_[starts[1:], len(key_s)]
        adj = np.stack([key_s[starts] // B, key_s[starts] % B], axis=1)
        cross = [cr_s[s:e] for s, e in zip(starts, ends)]
        saddle = w[cr_s[starts]]
    else:
        adj = np.empty((0, 2), dtype=np.int64)
        cross = []
        saddle = np.empty(0)
    return BasinGraph(peak=peak, basin=basin, adj=adj, cross=cross, saddle=saddle)


# ---------------------------------------------------------------------------
# brute-force references used by the tests (exponential / cubic; tiny inputs)
# ---------------------------------------------------------------------------

def brute_merge_heights(f: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """All-pairs maximin path values by a Floyd-Warshall style recursion."""
    f = np.asarray(f, dtype=float)
    n = len(f)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if w is None:
        w = edge_weights(f, edges)
    M = np.full((n, n), -np.inf)
    for (u, v), ww in zip(edges, w):
        M[u, v] = M[v, u] = max(M[u, v], ww)
    np.fill_diagonal(M, f)
    for k in range(n):
        M = np.maximum(M, np.minimum(M[:, [k]], M[[k], :]))
    np.fill_diagonal(M, f)
    return M


def brute_persistence(f: np.ndarray, edges: np.ndarray, w: np.ndarray | None = None):
    """Persistence pairs straight from the definition, by sweeping levels.

    For every level t in decreasing order of the distinct values, the
    components of the superlevel graph are recomputed from scratch with a
    graph search; a component present just above t that is no longer the
    owner of its merged component dies at t.  Returns a dict
    peak -> death level (-inf for components that never die).
    """
    f = np.asarray(f, dtype=float)
    n = len(f)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if w is None:
        w = edge_weights(f, edges)

    def components(t):
        alive = f >= t
        lab = -np.ones(n, dtype=int)
        adjl = [[] for _ in range(n)]
        for (u, v), ww in zip(edges, w):
            if ww >= t:
                adjl[u].append(v)
                adjl[v].append(u)
        c = 0
        for s in range(n):
            if alive[s] and lab[s] < 0:
                stack = [s]
                lab[s] = c
                while stack:
                    x = stack.pop()
                    for y in adjl[x]:
                        if lab[y] < 0:
                            lab[y] = c
                            stack.append(y)
                c += 1
        return lab

    def elder(vs):
        return max(vs, key=lambda v: (f[v], -v))

    death = {}
    levels = np.unique(np.concatenate([f, w]))[::-1]
    prev_peaks: set[int] = set()
    for t in levels:
        lab = components(t)
        peaks_now = set()
        for c in range(lab.max() + 1):
            vs = np.flatnonzero(lab == c).tolist()
            peaks_now.add(elder(vs))
        # a vertex that appears at this level and is not the elder of its
        # component is born and dies at t
        for v in np.flatnonzero(f == t).tolist():
            if v not in peaks_now:
                death.setdefault(v, t)
        for p in prev_peaks - peaks_now:
            death.setdefault(p, t)
        prev_peaks = peaks_now
    for p in prev_peaks:
        death.setdefault(p, -np.inf)
    return death
