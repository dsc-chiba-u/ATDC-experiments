"""Phase A: the intended use case on ordinary multivariate tables.

Public data sets with a class label (scikit-learn and OpenML), columns
standardised to unit variance.  For each: the prototype with the default
settings of the specification draft (K by the default rule, m = min(n, 4000),
dim = "auto"), its partition at the noise floor and at the recommended
threshold (stability path, B = 20), next to k-means with the number of classes
and HDBSCAN with package defaults.

The recommendation rule (stability >= 0.8, at least two clusters) was fixed
after the development run of this script (tag dev: iris ... letter, rule at
0.9); the six data sets after "letter" in the list were first run with the
final rule.

The class label is an external annotation, not the density structure: a low
ARI means "the classes are not separated by density valleys at this sample
size", which is itself the answer the tool is supposed to give.  The table
therefore also reports the stability and whether any threshold qualifies.

usage: 17_tabular.py [--tag main]
"""
import argparse
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI

from common import RAW, dump_json, env_info, save
from atdc.diagnostics import default_K, noise_floor, recommend, stability_path, vertex_labels

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--B", type=int, default=20)
ap.add_argument("--seed", type=int, default=800)
args = ap.parse_args()


def load():
    from sklearn import datasets as S
    out = []
    for name, fn in (("iris", S.load_iris), ("wine", S.load_wine), ("breast_cancer", S.load_breast_cancer),
                     ("digits", S.load_digits)):
        b = fn()
        out.append((name, "scikit-learn", b.data, b.target))
    for name, ver in (("penguins", 1), ("segment", 1), ("satimage", 1), ("pendigits", 1), ("MagicTelescope", 1),
                      ("letter", 1), ("optdigits", 1), ("vehicle", 1), ("banknote-authentication", 1),
                      ("wall-robot-navigation", 1), ("phoneme", 1), ("mfeat-fourier", 1)):
        try:
            b = S.fetch_openml(name=name, version=ver, as_frame=True, parser="auto")
            Xf = b.data.select_dtypes("number")
            keep = Xf.notna().all(axis=1) & b.target.notna()
            out.append((name, f"OpenML {name} v{ver}", Xf[keep].to_numpy(float), b.target[keep].astype("category").cat.codes.to_numpy()))
        except Exception as ex:
            out.append((name, f"OpenML {name} v{ver}", None, repr(ex)[:200]))
    return out


rows = []
for name, src, X, y in load():
    if X is None:
        rows.append(dict(dataset=name, source=src, error=y)); print(name, "not loaded:", y, flush=True); continue
    sd = X.std(axis=0)
    X = (X[:, sd > 0] - X[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    n, d = X.shape
    kc = len(np.unique(y))
    K = default_K(n)
    base = dict(dataset=name, source=src, n=n, d=d, n_classes=kc, K=K, tau0=noise_floor(K))
    if n < 2 * K + 2:
        rows.append(dict(base, error="too few rows for the default K")); continue
    t0 = time.perf_counter()
    km = KMeans(kc, n_init=10, random_state=0).fit_predict(X)
    t_km = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit, tab = stability_path(X, K=K, m=4000, B=args.B, seed=args.seed)
    t_stab = time.perf_counter() - t0
    nn = fit.nearest()

    def lab(tau):
        return vertex_labels(fit.tree, tau)[nn]
    l0 = lab(noise_floor(K))
    rec = (recommend(tab, K) or [None])[0]
    best = max(tab, key=lambda r: ARI(y, lab(r["tau"])))
    r = dict(base, d_hat=fit.d_hat, t_fit=sum(fit.times.values()), t_stability=t_stab, t_kmeans=t_km,
             kmeans_ari=ARI(y, km), kmeans_nmi=NMI(y, km),
             k_floor=int(l0.max() + 1), ari_floor=ARI(y, l0), nmi_floor=NMI(y, l0), ari_floor_vs_kmeans=ARI(km, l0),
             stab_at_floor=min(tab, key=lambda q: abs(q["tau"] - noise_floor(K)))["stability"],
             max_stability_multi=max([q["stability"] for q in tab if q["count_mode"] > 1], default=np.nan),
             oracle_tau=best["tau"], oracle_k=best["n_clusters"], oracle_ari=ARI(y, lab(best["tau"])),
             top_prom=" ".join(f"{v:.2f}" for v in fit.prominence()[:8]))
    if rec is not None:
        lr = lab(rec["tau"])
        r.update(rec_tau=rec["tau"], rec_k=int(lr.max() + 1), rec_stability=rec["stability"], rec_ari=ARI(y, lr),
                 rec_nmi=NMI(y, lr), rec_ari_vs_kmeans=ARI(km, lr))
    else:
        r.update(rec_tau=np.inf, rec_k=1, rec_stability=np.nan, rec_ari=0.0, rec_ari_vs_kmeans=0.0)
    try:
        import hdbscan
        t0 = time.perf_counter()
        h = hdbscan.HDBSCAN(core_dist_n_jobs=1).fit(X).labels_          # package defaults
        keep = h >= 0
        r.update(hdbscan_k=int(h.max() + 1), hdbscan_ari=ARI(y, h), hdbscan_noise=float(np.mean(~keep)),
                 hdbscan_ari_nonnoise=ARI(y[keep], h[keep]) if keep.sum() > 1 else np.nan,
                 t_hdbscan=time.perf_counter() - t0)
        h2 = hdbscan.HDBSCAN(min_cluster_size=max(5, n // 50), core_dist_n_jobs=1).fit(X).labels_
        keep = h2 >= 0
        r.update(hdbscan2_k=int(h2.max() + 1), hdbscan2_ari=ARI(y, h2), hdbscan2_noise=float(np.mean(~keep)))
    except Exception as ex:
        r["hdbscan_error"] = repr(ex)[:200]
    rows.append(r)
    print(name, n, d, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()
                       if k in ("d_hat", "kmeans_ari", "k_floor", "ari_floor", "rec_k", "rec_ari", "rec_stability",
                                "oracle_k", "oracle_ari", "hdbscan_k", "hdbscan_ari", "hdbscan_noise")}, flush=True)
    save(rows, f"17_tabular_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"17_tabular_{args.tag}.json")
