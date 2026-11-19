"""Phase A: how many observations does the density tree need to confirm a
clustering that k-means finds easily?

Data: four equal unit-variance Gaussians at pairwise distance s (a regular
simplex in three coordinates), in d = 3, 5 or 10 dimensions (the remaining
coordinates are N(0,1) noise).  True log-prominence of every non-global mode is
about s^2 / 8 (2.0, 3.1, 4.5 for s = 4, 5, 6): four clear modes.  k-means with
k = 4 separates them at any n.

For each n: the prototype with the defaults of the specification draft
(K by the default rule, every row a vertex up to 4 000, dim = "auto",
threshold = noise floor), its number of clusters and ARI against the true
basins; the same for the recommended threshold (stability path, B = 10);
k-means (k = 4, given the right k) and HDBSCAN with package defaults.

Seeds 500-519 (report; no setting was tuned on this script).

usage: 19_small_n.py [--tag main] [--jobs 12]
"""
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

from common import D, RAW, dump_json, env_info, save
from atdc.diagnostics import default_K, noise_floor, stability_path, vertex_labels
from atdc.truth_gmm import TruthGMM

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--seed0", type=int, default=500)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--ns", type=int, nargs="*", default=[100, 200, 400, 800, 1600, 3200, 6400, 12800])
ap.add_argument("--seps", type=float, nargs="*", default=[4.0, 5.0, 6.0])
ap.add_argument("--dims", type=int, nargs="*", default=[3, 5, 10])
ap.add_argument("--B", type=int, default=10)
ap.add_argument("--jobs", type=int, default=12)
args = ap.parse_args()


def task(job):
    sep, d, n, seed = job
    dist = D.make(f"simplex4_s{sep:g}_d{d}")
    X = dist.sample(n, np.random.default_rng(1000 + seed))
    truth = TruthGMM(dist, X)
    L = truth.labels(0.5)
    K = default_K(n)
    r = dict(sep=sep, d=d, n=n, seed=seed, K=K, tau0=noise_floor(K), n_true=int(L.max() + 1))
    if n < 2 * K + 2:
        return dict(r, error="n < 2K + 2")
    fit, tab = stability_path(X, K=K, m=4000, B=args.B, seed=seed)
    nn = fit.nearest()
    l0 = vertex_labels(fit.tree, noise_floor(K))[nn]
    r.update(d_hat=fit.d_hat, k_floor=int(l0.max() + 1), ari_floor=ARI(L, l0), exact_floor=float(l0.max() + 1 == L.max() + 1),
             stab_floor=min(tab, key=lambda q: abs(q["tau"] - noise_floor(K)))["stability"])
    for thr in (0.8, 0.9):
        ok = [q for q in tab if q["tau"] >= noise_floor(K) and q["count_mode"] >= 2 and q["stability"] >= thr
              and q["count_agreement"] >= 0.8]
        if ok:
            q = min(ok, key=lambda q: q["tau"])
            lr = vertex_labels(fit.tree, q["tau"])[nn]
            r.update({f"rec{int(thr * 100)}_k": int(lr.max() + 1), f"rec{int(thr * 100)}_ari": ARI(L, lr)})
        else:
            r.update({f"rec{int(thr * 100)}_k": 1, f"rec{int(thr * 100)}_ari": 0.0})
    km = KMeans(4, n_init=10, random_state=0).fit_predict(X)
    r.update(kmeans_ari=ARI(L, km))
    import hdbscan
    h = hdbscan.HDBSCAN(core_dist_n_jobs=1).fit(X).labels_
    r.update(hdbscan_k=int(h.max() + 1), hdbscan_ari=ARI(L, h), hdbscan_noise=float(np.mean(h < 0)))
    return r


if __name__ == "__main__":
    jobs = [(s, d, n, seed) for s in args.seps for d in args.dims for n in args.ns
            for seed in range(args.seed0, args.seed0 + args.seeds)]
    with Pool(args.jobs) as pool:
        rows = pool.map(task, jobs, chunksize=4)
    save(rows, f"19_small_n_{args.tag}")
    dump_json({"args": vars(args), "env": env_info()}, RAW / f"19_small_n_{args.tag}.json")
