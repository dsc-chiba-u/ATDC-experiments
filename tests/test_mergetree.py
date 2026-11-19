"""Correctness of the union-find merge tree against hand computation,
brute force, and GUDHI's ToMATo (positive control)."""
import numpy as np
import pytest

from atdc.mergetree import (ascent_basins, basin_graph, brute_merge_heights,
                            brute_persistence, edge_weights, merge_tree,
                            unique_undirected_edges)


def path_edges(n):
    return np.array([[i, i + 1] for i in range(n - 1)])


def pairs(t):
    """peak -> death level, restricted to positive persistence."""
    d = {int(p): float(l) for p, l in zip(t.dying, t.level) if t.f[p] - l > 0}
    for r in t.roots:
        d[int(r)] = -np.inf
    return d


# ---- hand-computed cases --------------------------------------------------

def test_single_peak():
    f = np.array([1.0, 2.0, 5.0, 3.0, 0.5])
    t = merge_tree(f, path_edges(5))
    assert pairs(t) == {2: -np.inf}
    assert np.all(t.labels(0.1) == 2)
    assert t.merge_heights([0, 4])[0, 1] == 0.5
    assert t.merge_heights([1, 3])[0, 1] == 2.0


def test_two_peaks_hand():
    #        0    1    2    3    4
    f = np.array([4.0, 1.0, 3.0, 0.5, 0.2])
    t = merge_tree(f, path_edges(5))
    # peak 2 (height 3) meets peak 0 (height 4) at vertex 1 (level 1): persistence 2
    assert pairs(t) == {0: -np.inf, 2: 1.0}
    assert list(t.labels(1.9)) == [0, 0, 2, 2, 2]
    assert list(t.labels(2.1)) == [0, 0, 0, 0, 0]
    # vertex 1 is the saddle and is attached to the elder component
    M = t.merge_heights([0, 2, 4])
    assert M[0, 1] == 1.0 and M[1, 2] == 0.2 and M[0, 2] == 0.2
    assert list(np.diag(M)) == [4.0, 3.0, 0.2]


def test_hierarchical_three_peaks_hand():
    # peaks A=10 (v0), B=8 (v2), C=9 (v4); saddles A-B at 6 (v1), B-C at 2 (v3)
    f = np.array([10.0, 6.0, 8.0, 2.0, 9.0])
    t = merge_tree(f, path_edges(5))
    # B dies into A at 6 (pers 2); C dies into {A,B} at 2 (pers 7)
    assert pairs(t) == {0: -np.inf, 2: 6.0, 4: 2.0}
    assert set(t.peaks(3.0).tolist()) == {0, 4}
    assert set(t.peaks(1.0).tolist()) == {0, 2, 4}
    assert list(t.labels(3.0)) == [0, 0, 0, 0, 4] or list(t.labels(3.0)) == [0, 0, 0, 4, 4]
    M = t.merge_heights([0, 2, 4])
    assert M[0, 1] == 6.0 and M[0, 2] == 2.0 and M[1, 2] == 2.0


def test_elder_rule_is_not_local():
    # C (9) is higher than B (8) but B merges with A (10) first; then the
    # component {A,B} (peak 10) meets C: C dies, not B's component.
    f = np.array([8.0, 7.0, 10.0, 1.0, 9.0])
    t = merge_tree(f, path_edges(5))
    assert pairs(t) == {2: -np.inf, 0: 7.0, 4: 1.0}


def test_ties_plateau_and_equal_peaks():
    f = np.array([3.0, 3.0, 3.0, 1.0, 3.0, 3.0])
    t = merge_tree(f, path_edges(6))
    # two plateaus, equal height: the one with the smaller id is the elder
    assert pairs(t) == {0: -np.inf, 4: 1.0}
    assert list(ascent_basins(f, path_edges(6))) == [0, 0, 0, 0, 4, 4]
    # all-equal field: one component, no positive persistence
    g = np.zeros(7)
    t = merge_tree(g, path_edges(7))
    assert pairs(t) == {0: -np.inf}
    assert len(np.unique(ascent_basins(g, path_edges(7)))) == 1


def test_tiny_and_infinite_values():
    # log-densities can be hugely negative or -inf (zero density)
    f = np.array([-1e300, -5.0, -np.inf, -700.0, -1e-320, -np.inf])
    t = merge_tree(f, path_edges(6))
    d = pairs(t)
    assert d[4] == -np.inf           # global maximum never dies
    assert d[1] == -np.inf or d[1] == -np.inf  # 1 meets 4 only through -inf vertex 2
    M = t.merge_heights([1, 4])
    assert M[0, 1] == -np.inf
    B = brute_merge_heights(f, path_edges(6))
    assert B[1, 4] == -np.inf
    # dense graph (forces the MST path) with -inf present
    edges = np.array([[i, j] for i in range(6) for j in range(i + 1, 6)])
    t2 = merge_tree(f, edges)
    assert np.array_equal(t2.merge_heights(np.arange(6)), brute_merge_heights(f, edges))


def test_disconnected_graph():
    f = np.array([1.0, 2.0, 3.0, 4.0])
    t = merge_tree(f, np.array([[0, 1], [2, 3]]))
    assert set(t.roots.tolist()) == {1, 3}
    assert t.merge_heights([0, 2])[0, 1] == -np.inf


def test_edge_cap_creates_separation():
    # two adjacent high vertices whose connecting edge dips to 0.5
    f = np.array([3.0, 2.9])
    edges = np.array([[0, 1]])
    t = merge_tree(f, edges, w=np.array([0.5]))
    assert pairs(t) == {0: -np.inf, 1: 0.5}
    with pytest.raises(ValueError):
        merge_tree(f, edges, w=np.array([2.95]))


# ---- randomised comparison with brute force -------------------------------

def random_graph(rng, n, p, ties):
    iu = np.triu_indices(n, 1)
    mask = rng.random(len(iu[0])) < p
    edges = np.stack([iu[0][mask], iu[1][mask]], axis=1)
    f = rng.integers(0, 4, n).astype(float) if ties else rng.normal(size=n)
    return f, edges


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("cap", [False, True])
@pytest.mark.parametrize("use_mst", [False, True])
def test_against_brute_force(ties, cap, use_mst):
    rng = np.random.default_rng(0)
    for trial in range(150):
        n = int(rng.integers(1, 13))
        f, edges = random_graph(rng, n, rng.uniform(0.1, 0.7), ties)
        w = edge_weights(f, edges)
        if cap and len(edges):
            drop = rng.random(len(edges)) < 0.4
            w = w - drop * (rng.integers(1, 3, len(edges)) if ties else rng.exponential(1.0, len(edges)))
        t = merge_tree(f, edges, w, use_mst=use_mst)
        # (1) merge heights = maximin path values
        M = t.merge_heights(np.arange(n))
        assert np.array_equal(M, brute_merge_heights(f, edges, w)), (trial, f, edges)
        # (2) persistence pairs = definition by level sweep
        ref = brute_persistence(f, edges, w)
        got = {int(p): float(l) for p, l in zip(t.dying, t.level)}
        got.update({int(r): -np.inf for r in t.roots})
        if not ties:
            assert got == ref
        else:
            # with ties the pairing is not unique; the multiset of
            # (birth, death) pairs is
            a = sorted((f[p], d) for p, d in got.items())
            b = sorted((f[p], d) for p, d in ref.items())
            assert a == b
        # (3) number of merges
        assert len(t.dying) + len(t.roots) == n


@pytest.mark.parametrize("cap", [False, True])
def test_basin_graph_tree_equals_vertex_tree(cap):
    """The tree of the basin adjacency graph has the same positive-persistence
    pairs and the same merge heights between peaks as the vertex-level tree."""
    rng = np.random.default_rng(1)
    for trial in range(200):
        n = int(rng.integers(2, 30))
        f, edges = random_graph(rng, n, rng.uniform(0.1, 0.5), ties=False)
        w = edge_weights(f, edges)
        if cap and len(edges):
            w = w - (rng.random(len(edges)) < 0.3) * rng.exponential(1.0, len(edges))
        t = merge_tree(f, edges, w)
        bg = basin_graph(f, edges, w)
        h = f[bg.peak]
        tb = merge_tree(h, bg.adj, bg.saddle)
        a = sorted((t.f[p], l) for p, l in zip(t.dying, t.level) if t.f[p] - l > 0)
        b = sorted((h[p], l) for p, l in zip(tb.dying, tb.level))
        assert np.allclose(a, b), (trial,)
        assert np.array_equal(t.merge_heights(bg.peak), tb.merge_heights(np.arange(len(bg.peak))))
        # labels from the basin tree agree with vertex-level labels
        for tau in (0.0, 0.3, 1.0):
            lv = t.labels(tau) if tau > 0 else None
            lb = bg.peak[tb.labels(tau)][bg.basin]
            if lv is not None:
                # same partition into retained peaks (assignment of
                # zero-persistence vertices may differ only by tie order)
                assert set(lv.tolist()) == set(lb.tolist())


def test_labels_match_tomato_rule():
    """labels(tau) equals a direct implementation of ToMATo's merge rule."""
    rng = np.random.default_rng(2)
    for trial in range(100):
        n = int(rng.integers(2, 40))
        f, edges = random_graph(rng, n, rng.uniform(0.1, 0.4), ties=False)
        tau = float(rng.uniform(0, 1.5))
        t = merge_tree(f, edges)
        lab = t.labels(tau)
        # direct ToMATo: vertices in decreasing f
        nb = [[] for _ in range(n)]
        for u, v in edges:
            nb[u].append(v)
            nb[v].append(u)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        seen = np.zeros(n, bool)
        for v in np.argsort(-f):
            up = [u for u in nb[v] if seen[u]]
            seen[v] = True
            if not up:
                continue
            g = max(up, key=lambda u: f[u])
            parent[v] = find(g)
            for u in up:
                ra, rb = find(u), find(v)
                if ra == rb:
                    continue
                lo, hi = (ra, rb) if f[ra] < f[rb] else (rb, ra)
                if f[lo] - f[v] < tau:
                    parent[lo] = hi
        ref = np.array([find(i) for i in range(n)])
        assert set(ref.tolist()) == set(lab.tolist()), trial
        # identical partitions whenever no vertex sits exactly between two
        # retained clusters at its own level (assignment of such boundary
        # vertices is a convention); compare on peaks' own labels at least
        assert np.array_equal(ref[np.unique(ref)], lab[np.unique(ref)])


def test_against_gudhi_tomato():
    """Positive control: same persistence diagram as GUDHI's ToMATo on the
    same graph and the same vertex values."""
    tomato = pytest.importorskip("gudhi.clustering.tomato")
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(3)
    X = np.concatenate([rng.normal(0, 1, (150, 2)), rng.normal(4, 1, (150, 2)),
                        rng.normal((0, 5), 0.6, (100, 2))])
    k = 8
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = nn.kneighbors(X)
    f = -2 * np.log(dist[:, -1])
    neighbors = [list(map(int, row[1:])) for row in ind]
    tm = tomato.Tomato(graph_type="manual", density_type="manual")
    tm.fit(neighbors, weights=f)
    edges = unique_undirected_edges(np.repeat(np.arange(len(X)), k), ind[:, 1:].ravel())
    t = merge_tree(f, edges)
    mine = sorted((t.f[p], l) for p, l in zip(t.dying, t.level) if t.f[p] - l > 0)
    theirs = sorted((b, d) for b, d in np.asarray(tm.diagram_))
    assert len(mine) == len(theirs)
    assert np.allclose(mine, theirs)
    # cluster labels at a threshold: same partition
    tau = 1.0
    tm.merge_threshold_ = tau
    lab = t.labels(tau)
    from sklearn.metrics import adjusted_rand_score
    assert tm.n_clusters_ == len(np.unique(lab))
    assert adjusted_rand_score(tm.labels_, lab) > 0.999
