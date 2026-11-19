"""Phase A: can a null result be told apart from a lack of resolution?

On small tables the density tree often reports one cluster where k-means with
the right k matches the classes (17_tabular.py).  "The tree does not confirm
the k-means clusters" then has two possible reasons: the clusters are not
separated by density valleys, or the sample is too small for the tree to see
valleys that exist.  A parametric check separates the two:

  1. take the partition under review (here: k-means with k = number of
     classes, on standardised columns);
  2. build the Gaussian mixture it implies (cluster proportions, means and
     covariances; covariances shrunk towards their diagonal by 0.1);
  3. simulate R data sets of the same size from that mixture and run the tree
     with the same settings;
  4. detectability = fraction of simulations in which the tree finds at least
     as many clusters as the partition has and its partition has ARI >= 0.8
     with the simulated components.

Reading: detectability near 0 -> the tree could not have confirmed these
clusters even if they were ideal Gaussian clusters: its silence is
uninformative.  Detectability near 1 while the tree finds fewer clusters in
the real data -> the real clusters are less separated in density than the
Gaussian picture of them.  This is a parametric bootstrap, not a new
statistic; it is evaluated here as a candidate feature of the comparison
module.

``--synthetic`` runs the same check where the answer is known (written to
21_detectability_synth.csv, data seeds 811-815):
  four Gaussian clusters, clear or close, at several n   k-means (k = 4) is right
  one Gaussian cut into four by k-means                  not density clusters
  a uniform square cut into four by k-means              not density clusters
  two Gaussian clusters cut into four by k-means         two of the four are real

usage: 21_detectability.py [--tag main] [--R 20] [--synthetic]
       21_detectability.py --tag pca10 --pca 10      # tables reduced first
"""
import argparse
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

from common import RAW, dump_json, env_info, save
from atdc.diagnostics import default_K, noise_floor, vertex_labels
from atdc.subsample import SubsampleTree

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--R", type=int, default=20)
ap.add_argument("--seed", type=int, default=810)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--pca", type=int, default=0,
                help="reduce tables with more columns to this many principal components first (k-means and tree alike)")
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()



def load():
    """Same data sets and preprocessing as 17_tabular.py."""
    from sklearn import datasets as S
    out = []
    for name, fn in (("iris", S.load_iris), ("wine", S.load_wine), ("breast_cancer", S.load_breast_cancer),
                     ("digits", S.load_digits)):
        b = fn(); out.append((name, b.data, b.target))
    for name in ("penguins", "segment", "satimage", "pendigits", "MagicTelescope", "letter", "optdigits", "vehicle",
                 "banknote-authentication", "wall-robot-navigation", "phoneme", "mfeat-fourier"):
        try:
            b = S.fetch_openml(name=name, version=1, as_frame=True, parser="auto")
            Xf = b.data.select_dtypes("number")
            keep = Xf.notna().all(axis=1) & b.target.notna()
            out.append((name, Xf[keep].to_numpy(float), b.target[keep].astype("category").cat.codes.to_numpy()))
        except Exception:
            pass
    return out


def tree_labels(X, seed):
    n, d = X.shape
    K = default_K(n)
    t = SubsampleTree(X, m=min(n, 4000), K=K, seed=seed, dim="auto", backend="kdtree" if d <= 15 else "brute")
    return vertex_labels(t.tree, noise_floor(K))[t.nearest()]


def check(X, km, kc, seed):
    """Tree on X, parametric simulations from the mixture implied by ``km``, verdict."""
    n, d = X.shape
    real = tree_labels(X, seed)
    comp = []
    for c in range(kc):
        Z = X[km == c]
        S = np.cov(Z, rowvar=False) if len(Z) > d else np.diag(Z.var(axis=0) + 1e-6)
        S = 0.9 * S + 0.1 * np.diag(np.diag(S)) + 1e-6 * np.eye(d)
        comp.append((len(Z) / n, Z.mean(axis=0), np.linalg.cholesky(S)))
    rng = np.random.default_rng([*np.atleast_1d(seed).tolist(), 1])
    ks, aris = [], []
    for r in range(args.R):
        z = rng.choice(kc, size=n, p=[c[0] for c in comp])
        Xs = np.stack([comp[j][1] + comp[j][2] @ rng.normal(size=d) for j in z])
        ls = tree_labels(Xs, [*np.atleast_1d(seed).tolist(), r])
        ks.append(int(ls.max() + 1)); aris.append(ARI(z, ls))
    ks, aris = np.array(ks), np.array(aris)
    det = float(np.mean((ks >= kc) & (aris >= 0.8)))
    k_real = int(real.max() + 1)
    ari_km = ARI(km, real)
    verdict = ("consistent" if (k_real >= kc and ari_km >= 0.6) else
               "inconclusive: resolution too low" if det < 0.5 else "not supported by density")
    return real, dict(k_reference=kc, k_tree=k_real, ari_tree_vs_kmeans=ari_km, detectability=det,
                      sim_k_median=float(np.median(ks)), sim_ari_mean=float(aris.mean()), verdict=verdict)


if args.synthetic:
    def gauss(n, d, sep, k, rng):                 # k unit Gaussians on a regular simplex of side sep
        c = np.zeros((k, d)); c[np.arange(k), np.arange(k) % d] = sep / np.sqrt(2.0)
        if k > d:
            raise ValueError
        return c[rng.integers(k, size=n)] + rng.normal(size=(n, d))

    cases = [("4 Gaussians, distance 6", lambda n, rng: gauss(n, 4, 6.0, 4, rng), (200, 800, 3200), "clusters are real"),
             ("4 Gaussians, distance 4", lambda n, rng: gauss(n, 4, 4.0, 4, rng), (400, 3200, 12800), "clusters are real"),
             ("1 Gaussian cut in 4", lambda n, rng: rng.normal(size=(n, 4)), (3200,), "not density clusters"),
             ("uniform square cut in 4", lambda n, rng: rng.uniform(size=(n, 2)), (3200,), "not density clusters"),
             ("2 Gaussians, distance 6, cut in 4", lambda n, rng: gauss(n, 4, 6.0, 2, rng), (3200,), "2 of 4 are real")]
    rows = []
    for name, make, ns, truth in cases:
        for n in ns:
            for rep in range(args.reps):
                seed = 811 + rep
                X = make(n, np.random.default_rng(1000 + seed))
                km = KMeans(4, n_init=10, random_state=0).fit_predict(X)
                _, out = check(X, km, 4, [seed, n])
                rows.append(dict(case=name, n=n, d=X.shape[1], seed=seed, truth=truth, **out))
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            save(rows, "21_detectability_synth")
    dump_json({"args": vars(args), "env": env_info()}, RAW / "21_detectability_synth.json")
    raise SystemExit

rows = []
for name, X, y in load():
    sd = X.std(axis=0)
    X = (X[:, sd > 0] - X[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    n, d0 = X.shape
    if args.pca and d0 > args.pca:
        from sklearn.decomposition import PCA
        X = PCA(args.pca, random_state=0).fit_transform(X)
    d = X.shape[1]
    kc = len(np.unique(y))
    if n < 2 * default_K(n) + 2:
        continue
    km = KMeans(kc, n_init=10, random_state=0).fit_predict(X)
    real, out = check(X, km, kc, args.seed)
    out = {"k_reference": out.pop("k_reference"), "kmeans_ari_vs_classes": ARI(y, km), "k_tree": out.pop("k_tree"),
           "ari_tree_vs_kmeans": out.pop("ari_tree_vs_kmeans"), "ari_tree_vs_classes": ARI(y, real), **out}
    rows.append(dict(dataset=name, n=n, d=d, d_original=d0, **out))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    save(rows, f"21_detectability_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"21_detectability_{args.tag}.json")
