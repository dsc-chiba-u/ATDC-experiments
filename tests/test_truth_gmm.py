"""The skeleton truth for isotropic mixtures agrees with the sample-graph truth
in 2-D and with the analytic two-component saddle in 30-D."""
import numpy as np

from atdc import datasets
from atdc.datasets import GMM
from atdc.truth import compute_truth
from atdc.truth_gmm import TruthGMM


def test_agrees_with_graph_truth_in_2d():
    for name in ("gauss3_close", "hier"):
        dist = datasets.make(name)
        X = dist.sample(3000, np.random.default_rng([5, 2]))
        a, b = compute_truth(dist, X), TruthGMM(dist, X)
        for tau in (0.3, 0.5):
            pa, ha, ra = a.modes(tau)
            pb, hb, rb = b.modes(tau)
            assert len(pa) == len(pb)
            assert np.allclose(np.sort(ha), np.sort(hb), atol=0.02)
            fa, fb = np.sort(ra[np.isfinite(ra)]), np.sort(rb[np.isfinite(rb)])
            assert np.allclose(fa, fb, atol=0.05)
            la, lb = a.labels(tau), b.labels(tau)
            # same partition up to a relabelling, except near basin boundaries
            from sklearn.metrics import adjusted_rand_score
            assert adjusted_rand_score(la, lb) > 0.97


def test_two_gaussians_in_30d():
    d, delta = 30, 5.0
    mu = np.zeros((2, d)); mu[1, 0] = delta
    dist = GMM(mu, 1.0, [1, 1])
    X = dist.sample(500, np.random.default_rng([1, 9]))
    t = TruthGMM(dist, X)
    p, h, r = t.modes(0.5)
    assert len(p) == 2
    mid = np.zeros((1, d)); mid[0, 0] = delta / 2
    assert abs(np.min(h) - dist.logpdf(mid)[0] - np.min(r[np.isfinite(r)])) < 5e-3
    lab = t.labels(0.5)
    side = (X[:, 0] > delta / 2).astype(int)
    assert min(np.mean(lab == side), np.mean(lab != side)) < 1e-9
    assert t.unconverged == 0.0
