"""v2: graph refinement checked against the true density (positive control),
and the held-out test checked for calibration."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from atdc import datasets as D
from atdc.datasets import GMM
from atdc.mergetree import merge_tree
from atdc.v2 import ATDC2, KNNOracle, TrueOracle


def near(X, c, r):
    return np.flatnonzero(np.linalg.norm(X - np.asarray(c), axis=1) < r)


def analytic_saddle(g, lo, hi):
    return minimize_scalar(lambda x: g.logpdf(np.array([[x, 0.0]]))[0], bounds=(lo, hi),
                           method="bounded").fun


def test_labels_keep_equals_threshold_rule():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(2, 40))
        iu = np.triu_indices(n, 1)
        m = rng.random(len(iu[0])) < 0.2
        edges = np.stack([iu[0][m], iu[1][m]], axis=1)
        t = merge_tree(rng.normal(size=n), edges)
        tau = float(rng.uniform(0, 1.5))
        keep = np.zeros(n, bool)
        keep[t.dying[t.persistence >= tau]] = True
        keep[t.roots] = True
        assert np.array_equal(t.labels_keep(keep), t.labels(tau))


def test_two_peaks_one_saddle_true_density():
    g = GMM([[0, 0], [4.0, 0]], 1.0, [0.5, 0.5])
    X = g.sample(4000, np.random.default_rng(0))
    sad = analytic_saddle(g, 1, 3)
    # the sparse graph has no vertex at the pass: merge level too low
    e0 = ATDC2(X, m0=150, seed=1, oracle=TrueOracle(X, g, K=30)).refine(0.5, saddles=False) \
        .estimate(0.5, with_labels=False)
    assert sad - e0.M[0, 1] > 0.05
    a = ATDC2(X, m0=150, seed=1, oracle=TrueOracle(X, g, K=30)).refine(0.5, max_rounds=8, k_ins=6)
    e = a.estimate(0.5, with_labels=True)
    assert len(e.peak_idx) == 2
    assert abs(e.M[0, 1] - sad) < 0.06
    assert np.all(np.abs(e.peak_h - g.logpdf(np.array([[0.0, 0.0]]))[0]) < 0.05)
    # labels split at the saddle abscissa
    side = X[:, 0] > 2.0
    lab = e.labels == e.labels[np.argmax(X[:, 0])]
    assert np.mean(lab == side) > 0.97


def test_missed_peak_is_recovered_by_ascent():
    """No initial vertex near the second mode (only on its slope)."""
    g = GMM([[0, 0], [6.0, 0]], 1.0, [0.5, 0.5])
    X = g.sample(6000, np.random.default_rng(1))
    rng = np.random.default_rng(2)
    d2 = np.linalg.norm(X - [6.0, 0], axis=1)
    slope = np.flatnonzero((d2 > 1.3) & (d2 < 2.5))
    init = np.r_[rng.choice(near(X, [0, 0], 2.5), 60, replace=False), rng.choice(slope, 25, replace=False)]
    top = g.logpdf(np.array([[6.0, 0.0]]))[0]

    def peak2(refine):
        a = ATDC2(X, seed=0, oracle=TrueOracle(X, g, K=40), init_idx=init)
        if refine:
            a.refine(0.5, edges=False, saddles=False, max_steps=25, asc_margin=0.5, n_cand=8)
        e = a.estimate(0.5, with_labels=False)
        right = e.peak_h[X[e.peak_idx, 0] > 3.0]
        return right.max()
    assert top - peak2(False) > 0.5          # the sample misses the top
    assert top - peak2(True) < 0.1           # the ascent reaches it


def test_long_edge_false_connection_is_removed():
    """Vertices only near the two centres: the graph joins the clusters by
    long edges that jump the valley, so the saddle is far too high."""
    g = GMM([[0, 0], [5.0, 0]], 1.0, [0.5, 0.5])
    X = g.sample(6000, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    init = np.r_[rng.choice(near(X, [0, 0], 0.8), 30, replace=False),
                 rng.choice(near(X, [5, 0], 0.8), 30, replace=False)]
    sad = analytic_saddle(g, 1.5, 3.5)
    top = g.logpdf(np.array([[0.0, 0.0]]))[0]
    tau = 1.0
    assert top - sad > tau + 1.0             # truly two clusters at this tau

    a = ATDC2(X, seed=0, k_graph=40, oracle=TrueOracle(X, g, K=40), init_idx=init)
    e0 = a.estimate(tau, with_labels=False)
    assert len(e0.peak_idx) == 1             # false connection: merged
    a = ATDC2(X, seed=0, k_graph=40, oracle=TrueOracle(X, g, K=40), init_idx=init)
    e1 = a.refine(tau, ascent=False, saddles=False, margin=tau, max_rounds=80,
                  hidden_rounds=60).estimate(tau, with_labels=False)
    assert len(e1.peak_idx) == 2
    # the verified merge height is the lowest density seen on the crossing
    # edges: never above the best crossing edge, never below the true saddle
    assert sad - 1e-9 <= e1.M[0, 1] <= sad + 0.3
    assert a.n_edge_checks > 0


def test_refinement_changes_the_merge_order_to_the_true_one():
    """A, B close; C far.  The initial graph has valley vertices between A and
    B but only a long edge between B and C, so B and C seem to merge first."""
    g = GMM([[0, 0], [3.6, 0], [9.6, 0]], 1.0, [1, 1, 1])
    X = g.sample(9000, np.random.default_rng(5))
    rng = np.random.default_rng(6)
    init = np.r_[rng.choice(near(X, [0, 0], 0.7), 25, replace=False),
                 rng.choice(near(X, [1.8, 0], 0.5), 6, replace=False),      # the A-B valley
                 rng.choice(near(X, [3.6, 0], 0.7), 25, replace=False),
                 rng.choice(near(X, [9.6, 0], 0.7), 25, replace=False)]
    tau = 0.3

    def order(refine):
        a = ATDC2(X, seed=0, k_graph=30, oracle=TrueOracle(X, g, K=40), init_idx=init)
        if refine:
            a.refine(tau, ascent=False, saddles=False, margin=tau, max_rounds=80, hidden_rounds=60)
        e = a.estimate(1e-6, with_labels=False)       # no pruning: read the raw tree
        x = X[e.peak_idx, 0]
        top = [int(np.flatnonzero(m)[np.argmax(e.peak_h[m])])
               for m in (x < 1.0, (x > 2.6) & (x < 5), x > 8)]
        M = e.M[np.ix_(top, top)]
        return M[0, 1], M[1, 2]
    ab, bc = order(False)
    assert bc > ab + 0.5                      # wrong: B-C first (long edge)
    ab, bc = order(True)
    assert ab > bc + 1.0                      # true order, with a clear margin
    assert abs(ab - analytic_saddle(g, 1, 2.6)) < 0.15


@pytest.mark.parametrize("name", ["unimodal", "uniform", "banana"])
def test_single_mode_true_density(name):
    dist = D.make(name)
    X = dist.sample(4000, np.random.default_rng(0))
    a = ATDC2(X, m0=300, seed=0, oracle=TrueOracle(X, dist, K=30)).refine(0.3)
    assert len(a.estimate(0.3, with_labels=False).peak_idx) == 1


def test_held_out_prominence_is_unbiased_on_a_flat_density():
    """On a uniform density every event is noise.  The in-sample prominence of
    the selected events is positive by construction; the held-out prominence
    of the same events is centred at zero with the predicted spread."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60000, 2))
    X = X[np.all((X > 0.02) & (X < 0.98), axis=1)]
    ins, held, se = [], [], None
    for seed in range(8):
        a = ATDC2(X, K=60, m0=1500, rho=0.5, seed=seed)
        # keep away from the boundary of the support, where the density is not flat
        e = a.estimate(0.3, margin=0.3, with_labels=False)
        tree = a._tree()
        ins += e.info["in_sample_r"]
        held += e.info["held_r"]
        se = np.sqrt(2) * a.o.se_held
    ins, held = np.array(ins), np.array(held)
    assert len(held) > 200
    assert ins.mean() > 0.05
    assert abs(held.mean()) < 4 * se / np.sqrt(len(held)) + 0.02
    assert 0.8 < held.std() / se < 1.3
    # false positives at tau = 0.5: the in-sample rule keeps some, the
    # held-out rule keeps about the nominal tail
    assert np.mean(held >= 0.5) < 0.03


def test_split_uses_disjoint_data_and_same_bandwidth():
    X = D.make("gauss3").sample(8000, np.random.default_rng(0))
    o = KNNOracle(X, K=100, rho=0.3, seed=0)
    assert o.K1 == 30 and o.K2 == 70
    assert len(o.X1) + len(o.X2) == len(X)
    assert len(np.intersect1d(o.pool, np.setdiff1d(np.arange(len(X)), o.pool))) == 0
    a = ATDC2(X, K=100, m0=200, rho=0.3, seed=0)
    assert np.all(np.isin(a.idx, a.o.pool))          # vertices come from H1 only
    # same expected ball radius in both parts
    idx = np.array(a.idx[:200])
    r1 = a.o.i1.query(X[idx], [31])[0][:, 0]
    r2 = a.o.i2.query(X[idx], [70])[0][:, 0]
    assert abs(np.median(r1 / r2) - 1) < 0.1


def test_v2_without_refinement_equals_uniform_refinement():
    """ATDC2 with no split and no refinement is the Phase-1 'global' control
    on the same vertices: same peaks, same heights."""
    from atdc.core import knn_graph_edges
    from atdc.density import knn_log_density
    from scipy.spatial import cKDTree
    X = D.make("hier").sample(5000, np.random.default_rng(0))
    a = ATDC2(X, K=50, m0=400, seed=3)
    e = a.estimate(0.5, with_labels=False)
    idx = np.array(a.idx)
    r = cKDTree(X).query(X[idx], [51])[0][:, 0]
    f = knn_log_density(r, 50, len(X), 2)
    _, ind = cKDTree(X[idx]).query(X[idx], 11)
    t = merge_tree(f, knn_graph_edges(ind, 10))
    ref = np.sort(f[t.peaks(0.5)])
    assert np.allclose(np.sort(e.peak_h), ref)


def test_deterministic_and_cost_counted():
    X = D.make("moons").sample(5000, np.random.default_rng(0))
    es = [ATDC2(X, K=60, m0=300, rho=0.5, seed=2).refine(0.5).estimate(0.5) for _ in range(2)]
    assert np.array_equal(es[0].labels, es[1].labels)
    assert es[0].cost_units == es[1].cost_units > 300 * 30
    assert es[0].labels.min() >= 0 and len(np.unique(es[0].labels)) == len(es[0].peak_idx)
