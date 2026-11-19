"""Phase A: two real data sets of the kinds named in the plan (see data_prep.py).

  pbmc68k   68 579 cells, 50 principal components, 11 annotated cell types
  ng20      18 221 documents, 384-d sentence embeddings, 20 newsgroups

Representations: the data as given ("raw"), their first 10 principal
components, and a 5-d UMAP embedding (n_neighbors 15, min_dist 0, seed 0) - the
usual preprocessing for density clustering of embeddings.  UMAP does not
preserve density; a tree of a UMAP embedding describes the embedding.

Methods on each representation: k-means with the number of annotated classes,
Leiden on a 15-NN graph (resolution 1), HDBSCAN (min_cluster_size n/400 and
n/100, min_samples 10), and the prototype with the defaults of the
specification draft (K = 100, m = 4 000, dim = "auto"): partition at the noise
floor for R vertex draws, and the stability path (B = 10) with the recommended
threshold.

The annotation is an external label, not the density structure.

usage: 20_real_data.py [--tag main]
"""
import argparse
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from sklearn.neighbors import NearestNeighbors

from common import RAW, ROOT, dump_json, env_info, save
from atdc.diagnostics import noise_floor, recommend, stability_path, vertex_labels
from atdc.subsample import SubsampleTree

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--R", type=int, default=5)
ap.add_argument("--B", type=int, default=10)
ap.add_argument("--no_raw_hdbscan", action="store_true")
args = ap.parse_args()
K = 100
rows = []


def add(data, rep, method, y, lab, t, **kw):
    lab = np.asarray(lab); keep = lab >= 0
    rows.append(dict(data=data, representation=rep, method=method, k=int(len(np.unique(lab[keep]))),
                     ari=ARI(y, lab), nmi=NMI(y, lab), frac_noise=float(np.mean(~keep)),
                     ari_nonnoise=ARI(y[keep], lab[keep]) if keep.sum() > 1 else np.nan, wall=t, **kw))
    print(data, rep, method, {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()
                              if k in ("k", "ari", "frac_noise", "wall", "stability", "param")}, flush=True)
    save(rows, f"20_real_data_{args.tag}")


def leiden(X, k=15, res=1.0):
    import igraph as ig
    import leidenalg
    ind = NearestNeighbors(n_neighbors=k + 1, algorithm="brute" if X.shape[1] > 15 else "kd_tree") \
        .fit(X).kneighbors(X, return_distance=False)[:, 1:]
    e = np.unique(np.sort(np.stack([np.repeat(np.arange(len(X)), k), ind.ravel()], 1), 1), axis=0)
    p = leidenalg.find_partition(ig.Graph(n=len(X), edges=e.tolist()), leidenalg.RBConfigurationVertexPartition,
                                 resolution_parameter=res, seed=0)
    return np.asarray(p.membership)


for data, fx, fy in (("pbmc68k", "pbmc68k_pca50.npy", "pbmc68k_labels.npy"),
                     ("ng20", "ng20_minilm384.npy", "ng20_labels.npy")):
    X0 = np.load(ROOT / "data" / fx).astype(float)
    y = np.load(ROOT / "data" / fy)
    n, kc = len(X0), len(np.unique(y))
    import umap
    t0 = time.perf_counter()
    U = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, random_state=0, n_jobs=1).fit_transform(X0)
    t_umap = time.perf_counter() - t0
    reps = {"raw": X0, "pca10": PCA(10, random_state=0).fit_transform(X0), "umap5": np.asarray(U, float)}
    for rep, X in reps.items():
        extra = dict(n=n, d=X.shape[1], t_umap=t_umap if rep == "umap5" else 0.0)
        t0 = time.perf_counter(); l = KMeans(kc, n_init=5, random_state=0).fit_predict(X)
        add(data, rep, "kmeans", y, l, time.perf_counter() - t0, param=f"k={kc}", **extra)
        t0 = time.perf_counter(); l = leiden(X)
        add(data, rep, "leiden", y, l, time.perf_counter() - t0, param="k=15,res=1", **extra)
        if not (rep == "raw" and (args.no_raw_hdbscan or n > 30000)):
            import hdbscan
            for mcs in (n // 400, n // 100):
                t0 = time.perf_counter()
                l = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10, core_dist_n_jobs=1).fit(X).labels_
                add(data, rep, "hdbscan", y, l, time.perf_counter() - t0, param=f"mcs={mcs}", **extra)
        else:
            rows.append(dict(data=data, representation=rep, method="hdbscan", error="not run: n > 30000 in 50 dimensions "
                             "(the package took 416 s on the 18 221 x 384 set)", **extra))
        be = "kdtree" if X.shape[1] <= 15 else "brute"
        for r in range(args.R):
            t0 = time.perf_counter()
            t = SubsampleTree(X, m=4000, K=K, seed=[900, r], dim="auto", backend=be)
            l = t.estimate(noise_floor(K)).labels
            add(data, rep, "atdc_floor", y, l, time.perf_counter() - t0, param=f"draw={r}", d_hat=t.d_hat, **extra)
        t0 = time.perf_counter()
        fit, tab = stability_path(X, K=K, m=4000, B=args.B, seed=900, backend=be)
        ts = time.perf_counter() - t0
        st0 = min(tab, key=lambda q: abs(q["tau"] - noise_floor(K)))["stability"]
        rec = (recommend(tab, K) or [None])[0]
        if rec is None:
            add(data, rep, "atdc_recommended", y, np.zeros(n, int), ts, param="no stable partition", stability=np.nan,
                stability_at_floor=st0, **extra)
        else:
            l = vertex_labels(fit.tree, rec["tau"])[fit.nearest()]
            add(data, rep, "atdc_recommended", y, l, ts, param=f"tau={rec['tau']:.2f}", stability=rec["stability"],
                stability_at_floor=st0, **extra)
dump_json({"args": vars(args), "env": env_info(), "K": K}, RAW / f"20_real_data_{args.tag}.json")
