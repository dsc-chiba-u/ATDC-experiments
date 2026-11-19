"""The discretised ground-truth tree against analytic values."""
import numpy as np
from scipy.optimize import minimize_scalar

from atdc import datasets as D
from atdc.datasets import GMM
from atdc.truth import compute_truth


def test_two_gaussians_analytic_saddle_and_peaks():
    g = GMM([[0, 0], [4.0, 0]], np.array([1.0, 0.64]), [0.6, 0.4])
    X = g.sample(2000, np.random.default_rng(0))
    tr = compute_truth(g, X)
    p, h, pers = tr.modes(0.05)
    assert len(p) == 2
    # by symmetry the modes and the saddle lie on the x-axis
    line = lambda x: -g.logpdf(np.array([[x, 0.0]]))[0]
    m1 = -minimize_scalar(line, bounds=(-1, 1), method="bounded").fun
    m2 = -minimize_scalar(line, bounds=(3, 5), method="bounded").fun
    sad = minimize_scalar(lambda x: -line(x), bounds=(1, 3), method="bounded").fun
    assert np.allclose(sorted(h), sorted([m1, m2]), atol=5e-3)
    M = tr.merge_matrix(0.05)
    assert abs(M[0, 1] - sad) < 0.03
    assert abs(np.min(pers) - (min(m1, m2) - sad)) < 0.03
    # labels: split by the saddle abscissa up to boundary effects
    lab = tr.labels(0.05)
    assert len(np.unique(lab)) == 2


def test_unimodal_and_flat_distributions_have_one_mode():
    for name in ("unimodal", "uniform", "banana"):
        dist = D.make(name)
        tr = compute_truth(dist, dist.sample(1000, np.random.default_rng(0)), n_extra=20000)
        assert len(tr.modes(0.05)[0]) == 1, name


def test_noise_dimensions_shift_heights_only():
    base, hi = D.make("gauss3"), D.make("gauss3_noise3")
    X = hi.sample(1500, np.random.default_rng(0))
    t_hi = compute_truth(hi, X, n_extra=20000)
    t_lo = compute_truth(base, X[:, :2], n_extra=20000)
    assert np.allclose(t_hi.modes(0.1)[2], t_lo.modes(0.1)[2])
    assert np.allclose(t_hi.modes(0.1)[1] - t_lo.modes(0.1)[1], -1.5 * np.log(2 * np.pi))
    # the full-dimensional log-density at a mode equals the shifted value
    p = t_hi.modes(0.1)[0][0]
    x = np.r_[t_hi.ref[p], np.zeros(3)]
    assert np.isclose(hi.logpdf(x[None])[0], t_hi.f[p])
