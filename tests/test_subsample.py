"""The Phase-A reference estimator reproduces uniform refinement (ATDC2 without
refinement) and its exact neighbour back-ends agree with each other."""
import numpy as np
import pytest

from atdc import datasets
from atdc.subsample import SubsampleTree, _brute_knn, nearest_vertex
from atdc.v2 import ATDC2


@pytest.fixture(scope="module")
def data():
    return datasets.make("hier").sample(4000, np.random.default_rng([7, 1]))


def test_equals_atdc2_uniform(data):
    a = ATDC2(data, K=50, m0=400, seed=3).estimate(0.5)
    b = SubsampleTree(data, m=400, K=50, seed=3).estimate(0.5)
    assert np.array_equal(np.sort(a.peak_idx), np.sort(b.peak_idx))
    oa, ob = np.argsort(a.peak_idx), np.argsort(b.peak_idx)
    assert np.allclose(a.peak_h[oa], b.peak_h[ob])
    assert np.allclose(a.M[np.ix_(oa, oa)], b.M[np.ix_(ob, ob)])
    assert np.array_equal(oa.argsort()[a.labels], ob.argsort()[b.labels])


def test_brute_backend_equals_kdtree(data):
    a = SubsampleTree(data, m=400, K=50, seed=3, backend="kdtree")
    b = SubsampleTree(data, m=400, K=50, seed=3, backend="brute", label_backend="brute")
    assert np.allclose(a.f, b.f, atol=1e-9)
    assert np.array_equal(a.edges, b.edges)
    assert np.array_equal(a.estimate(0.5).labels, b.estimate(0.5).labels)


def test_brute_knn_high_dim():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 40))
    d, i = _brute_knn(X[:50], X, 7, exclude=np.arange(50))
    full = np.linalg.norm(X[:50, None, :] - X[None, :, :], axis=2)
    full[np.arange(50), np.arange(50)] = np.inf
    ref = np.argsort(full, axis=1)[:, :7]
    assert np.array_equal(i, ref)
    assert np.array_equal(nearest_vertex(X, X[:50], "brute"), nearest_vertex(X, X[:50], "kdtree"))


def test_approximate_backends_are_close(data):
    a = SubsampleTree(data, m=300, K=30, seed=1, backend="kdtree")
    for be in ("hnsw", "nndescent"):
        b = SubsampleTree(data, m=300, K=30, seed=1, backend=be)
        recall = np.mean([len(np.intersect1d(x, y)) / 30 for x, y in zip(a.nb_ind, b.nb_ind)])
        assert recall > 0.9
        assert np.all(b.f <= a.f + 1e-9)          # an approximate radius is never smaller


def test_prominence_path_counts(data):
    t = SubsampleTree(data, m=500, K=50, seed=0)
    p = t.prominence()
    for tau in (0.2, 0.5, 1.0, 2.0):
        assert t.n_clusters(tau) == int(np.sum(p >= tau)) == len(t.estimate(tau, with_labels=False).peak_idx)
