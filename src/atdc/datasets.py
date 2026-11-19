"""Synthetic distributions with an exactly computable density.

Every distribution is either a Gaussian mixture with diagonal covariances or
a uniform box, so ``logpdf`` is exact and sampling is exact.  Curved shapes
(moons, rings) are mixtures of many small isotropic Gaussians placed densely
along the curve; the density that generated the data is that mixture itself,
not an idealised curve.

The generating component of a point is NOT used as ground truth anywhere:
ground-truth clusters are the basins of the modes of the true density
(see ``truth.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp


@dataclass
class GMM:
    means: np.ndarray      # (C, d)
    var: np.ndarray        # (C, d) diagonal variances
    weights: np.ndarray    # (C,)
    name: str = "gmm"
    # the first ``core_dim`` coordinates carry the structure; the remaining
    # ones are independent N(0, s^2) noise shared by all components
    core_dim: int | None = None

    def __post_init__(self):
        self.means = np.atleast_2d(np.asarray(self.means, float))
        C, d = self.means.shape
        v = np.asarray(self.var, float)
        if v.ndim == 0:
            v = np.full((C, d), float(v))
        elif v.ndim == 1:
            v = np.repeat(v[:, None], d, axis=1)
        self.var = v
        w = np.asarray(self.weights, float)
        self.weights = w / w.sum()
        if self.core_dim is None:
            self.core_dim = d

    @property
    def dim(self):
        return self.means.shape[1]

    def sample(self, n, rng):
        comp = rng.choice(len(self.weights), size=n, p=self.weights)
        return self.means[comp] + rng.normal(size=(n, self.dim)) * np.sqrt(self.var[comp])

    def logpdf(self, X, chunk=20000):
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        lognorm = np.log(self.weights) - 0.5 * np.sum(np.log(2 * np.pi * self.var), axis=1)
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk]
            q = np.zeros((len(x), len(self.weights)))
            for j in range(self.dim):
                q += (x[:, [j]] - self.means[None, :, j]) ** 2 / self.var[None, :, j]
            out[s:s + chunk] = logsumexp(lognorm[None, :] - 0.5 * q, axis=1)
        return out

    def core(self) -> "GMM":
        """Marginal on the structured coordinates."""
        c = self.core_dim
        return GMM(self.means[:, :c], self.var[:, :c], self.weights, name=self.name + "_core")

    def noise_log_const(self) -> float:
        """log-density of the noise coordinates at 0 (they are centred)."""
        c = self.core_dim
        if c == self.dim:
            return 0.0
        return float(-0.5 * np.sum(np.log(2 * np.pi * self.var[0, c:])))


@dataclass
class UniformBox:
    low: np.ndarray
    high: np.ndarray
    name: str = "uniform"
    core_dim: int | None = None

    def __post_init__(self):
        self.low = np.asarray(self.low, float)
        self.high = np.asarray(self.high, float)
        self.core_dim = len(self.low)

    @property
    def dim(self):
        return len(self.low)

    def sample(self, n, rng):
        return rng.uniform(self.low, self.high, size=(n, self.dim))

    def logpdf(self, X):
        X = np.atleast_2d(X)
        inside = np.all((X >= self.low) & (X <= self.high), axis=1)
        return np.where(inside, -np.sum(np.log(self.high - self.low)), -np.inf)

    def core(self):
        return self

    def noise_log_const(self):
        return 0.0


def _arc(center, radius, a0, a1, m):
    t = np.linspace(a0, a1, m)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], axis=1)


def add_noise_dims(g: GMM, extra: int, sd: float) -> GMM:
    C = len(g.weights)
    means = np.concatenate([g.means, np.zeros((C, extra))], axis=1)
    var = np.concatenate([g.var, np.full((C, extra), sd ** 2)], axis=1)
    return GMM(means, var, g.weights, name=f"{g.name}_noise{extra}", core_dim=g.dim)


def make(name: str):
    """Return a distribution by name."""
    if name == "gauss3":          # spherical, well separated, equal
        return GMM([[0, 0], [5, 0], [2.5, 4.5]], 1.0, [1, 1, 1], name=name)
    if name == "gauss3_close":    # spherical, overlapping: true log-prominence ~0.9
        s = 3.6
        return GMM([[0, 0], [s, 0], [s / 2, s * 0.866]], 1.0, [1, 1, 1], name=name)
    if name == "gauss3_border":   # deliberately ambiguous: true log-prominence ~0.4-0.45
        return GMM([[0, 0], [3.0, 0], [1.5, 2.7]], 1.0, [1, 1, 1], name=name)
    if name == "hetero":          # different variance, density and size
        return GMM([[0, 0], [4.5, 0.5], [1.5, 5.5], [7.5, 5.0]],
                   np.array([0.25, 1.0, 2.25, 0.09]), [0.25, 0.4, 0.3, 0.05], name=name)
    if name == "moons":
        m = 120
        up = _arc((0.0, 0.0), 1.0, 0.0, np.pi, m)
        dn = _arc((1.0, 0.5), 1.0, np.pi, 2 * np.pi, m)
        return GMM(np.concatenate([up, dn]), 0.12 ** 2, np.ones(2 * m), name=name)
    if name == "circles":
        m_out, m_in = 240, 96
        out = _arc((0, 0), 1.0, 0, 2 * np.pi, m_out + 1)[:-1]
        inn = _arc((0, 0), 0.4, 0, 2 * np.pi, m_in + 1)[:-1]
        w = np.r_[np.full(m_out, 0.6 / m_out), np.full(m_in, 0.4 / m_in)]
        return GMM(np.concatenate([out, inn]), 0.07 ** 2, w, name=name)
    if name == "hier":
        # two super-clusters far apart; the left one has three sub-peaks with
        # shallow saddles, the right one two sub-peaks of unequal height
        # (true log-prominences ~0.9-1.1) and three merge levels
        means = np.array([[0, 0], [2.6, 0.3], [1.2, 2.5], [9.0, 0.5], [11.3, 1.6]])
        means[:3] *= 1.25
        means[4] = means[3] + (means[4] - means[3]) * 1.25
        var = np.array([0.8, 0.8, 0.7, 0.9, 0.6])
        return GMM(means, var, [0.22, 0.2, 0.18, 0.25, 0.15], name=name)
    if name == "unimodal":
        return GMM([[0, 0]], 1.0, [1], name=name)
    if name == "uniform":         # no natural partition
        return UniformBox([0, 0], [1, 1], name=name)
    if name == "banana":          # curved, unimodal ridge with a single top
        m = 150
        t = np.linspace(-1.2, 1.2, m)
        pts = np.stack([2 * np.sin(t), 2 * np.cos(t)], axis=1)
        w = np.exp(-0.5 * (t / 0.7) ** 2)
        return GMM(pts, 0.25 ** 2, w, name=name)
    if name.startswith("gauss3_noise"):   # e.g. gauss3_noise8 -> 2 + 8 dims
        return add_noise_dims(make("gauss3"), int(name[len("gauss3_noise"):]), 1.0)
    if name.startswith("hier_noise"):
        return add_noise_dims(make("hier"), int(name[len("hier_noise"):]), 1.0)
    raise KeyError(name)


ALL_2D = ["gauss3", "gauss3_close", "gauss3_border", "hetero", "moons", "circles", "hier",
          "unimodal", "uniform", "banana"]
HIGH_DIM = ["gauss3_noise3", "gauss3_noise8", "hier_noise3"]
