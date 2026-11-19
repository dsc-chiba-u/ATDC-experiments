"""Stage A / Stage B invariants on small data."""
import numpy as np
import pytest
from scipy.special import digamma, polygamma

from atdc import datasets as D
from atdc.core import ATDC, full_tree
from atdc.density import knn_log_density, knn_log_density_se
from atdc.mergetree import merge_tree
from atdc.metrics import evaluate
from atdc.truth import compute_truth


def _pairs(est):
    return sorted(zip(np.round(est.peak_h, 10), np.round(est.peak_pers, 10)))


def test_knn_log_density_unbiased_under_uniform_density():
    rng = np.random.default_rng(0)
    from scipy.spatial import cKDTree
    n, d = 40000, 2
    X = rng.uniform(0, 1, (n, d))
    inner = np.all((X > 0.2) & (X < 0.8), axis=1)      # avoid the boundary
    tree = cKDTree(X)
    for k in (5, 20, 80):
        r = tree.query(X[inner], [k + 1])[0][:, 0]
        f = knn_log_density(r, k, n, d)                 # true log-density is 0
        se = knn_log_density_se(k)
        assert abs(f.mean()) < 4 * se / np.sqrt(inner.sum() / (k + 1)) + 0.01
        assert abs(f.std() / se - 1) < 0.1
        assert np.isclose(se ** 2, polygamma(1, k))


@pytest.mark.parametrize("name", ["gauss3", "moons", "unimodal"])
def test_stage_a_basin_tree_equals_vertex_tree(name):
    X = D.make(name).sample(1500, np.random.default_rng(0))
    a = ATDC(X, K=40, q=0.25, seed=0).stage_a()
    v = merge_tree(a.f0, a.edges)
    b = a.basin_tree()
    pv = sorted(p for p in v.persistence if p > 0)
    pb = sorted(b.persistence)
    assert np.allclose(pv, pb)
    for tau in (0.3, 0.8):
        e1 = a.vertex_estimate(tau, 0, "coarse", with_labels=False)
        e2 = a.estimate(tau, with_labels=False)
        assert _pairs(e1) == _pairs(e2)
        assert set(e1.peak_idx) == set(e2.peak_idx)


@pytest.mark.parametrize("mode", ["selective", "random", "all"])
def test_refinement_is_a_noop_when_fidelities_coincide(mode):
    """With q = 1 the subsample is the data and k0 = K, so the 'fine'
    estimate equals the coarse one.  Whatever is refined, the tree must stay
    exactly the vertex-level tree."""
    X = D.make("hier").sample(800, np.random.default_rng(1))
    tau = 0.4
    a = ATDC(X, K=12, q=1.0, seed=0, t_top=3).stage_a()
    ref = a.vertex_estimate(tau, 0, "coarse", with_labels=False)
    a.refine(tau, z=1.0, mode=mode, budget=200 if mode == "random" else None)
    assert a.n_refined > 0
    done = ~np.isnan(a.f1)
    assert np.allclose(a.f1[done], a.f0[done])
    est = a.estimate(tau, with_labels=False)
    assert _pairs(est) == _pairs(ref)
    assert np.allclose(np.sort(est.M, axis=None), np.sort(ref.M, axis=None))


def test_selective_fixpoint_verifies_every_retained_event():
    X = D.make("hier").sample(6000, np.random.default_rng(2))
    a = ATDC(X, K=60, q=0.15, seed=3).stage_a().refine(0.5, z=0.0)
    est = a.estimate(0.5)
    assert a.stop_reason == "fixpoint"
    assert est.info["n_unverified_retained"] == 0
    assert est.labels.shape == (len(X),) and est.labels.min() >= 0
    assert len(np.unique(est.labels)) == len(est.peak_idx)
    assert np.all(est.peak_pers >= 0.5)
    # estimated merge heights never exceed the lower of the two peaks
    for i in range(len(est.peak_h)):
        for j in range(len(est.peak_h)):
            assert est.M[i, j] <= min(est.peak_h[i], est.peak_h[j]) + 1e-12


def test_budget_and_cost_accounting():
    X = D.make("gauss3").sample(5000, np.random.default_rng(0))
    a = ATDC(X, K=50, q=0.2, seed=0).stage_a()
    c0 = a.cost
    assert c0 == a.m * a.k0
    a.refine(0.5, z=1.0, budget=60)
    assert a.stop_reason == "budget"
    assert a.n_refined <= 60 + 3 * a.t_top           # one event of slack
    assert a.cost == c0 + a.n_refined * a.K
    b = ATDC(X, K=50, q=0.2, seed=0).stage_a().refine(0.5, mode="random", budget=60)
    assert b.n_refined <= 60 + 3 * 2 * b.t_top


def test_deterministic_given_seed():
    X = D.make("moons").sample(3000, np.random.default_rng(0))
    r = [ATDC(X, K=40, q=0.2, seed=7).stage_a().refine(0.5).estimate(0.5) for _ in range(2)]
    assert np.array_equal(r[0].labels, r[1].labels)
    assert np.array_equal(r[0].peak_idx, r[1].peak_idx)


def test_refinement_removes_spurious_peaks_on_unimodal_data():
    """Sanity (not a proof of usefulness): on a unimodal Gaussian the coarse
    tree has many peaks above tau, the refined one a single peak."""
    dist = D.make("unimodal")
    X = dist.sample(20000, np.random.default_rng(0))
    truth = compute_truth(dist, X, n_extra=5000)
    tau = 0.5
    a = ATDC(X, K=100, q=0.1, seed=0).stage_a()
    coarse = a.vertex_estimate(tau, 0, "coarse", with_labels=False)
    a.refine(tau, z=0.0)
    sel = a.estimate(tau, with_labels=False)
    assert len(coarse.peak_idx) > 5
    assert evaluate(sel, truth, tau)["exact"] == 1.0
    assert a.n_refined < 0.5 * a.m
    full = full_tree(X, 100, 10, tau)
    assert len(full.peak_idx) == 1
