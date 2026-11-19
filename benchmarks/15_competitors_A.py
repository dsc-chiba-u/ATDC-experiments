"""Phase A (3.3): the subsample tree next to existing tools.

What is compared, and for what purpose
  same target     sub(m): subsample tree (K = 100, tau = 0.5, m vertices)
                  full: the same estimator on all points
                  tomato_w: GUDHI ToMATo given the same log-kNN density
                  -> what subsampling loses (resolution m instead of n)
  near target     tomato_logdtm: GUDHI ToMATo with its default logDTM density
  other targets   hdbscan (hdbscan package; mutual-reachability hierarchy,
                  excess-of-mass selection, points may be noise),
                  densitycut (R package; random-walk density, valley/peak
                  ratio), adp (DADApy; Z-test of peak against saddle),
                  leiden (modularity on a kNN graph; not density based),
                  automato (parameter-free ToMATo; only with --automato, it
                  needs minutes per run)
All competitors are called through their own packages.  The reference
partition is the basin partition of the TRUE density pruned at tau = 0.5; it
is the target of the first group by construction, so for the other groups the
table says how far a different definition lands from it, not that they are
wrong.  Competitor settings: package defaults, plus a small grid of their main
parameter; the grid member closest to the reference is also reported
("oracle"), which favours the competitors.  The subsample tree is run at fixed
settings only.

Label-based columns (all methods): n_est, k_correct, ari, nmi, ari_nonnoise,
frac_noise.  Tree columns (first group): metrics.evaluate.
Wall times are taken under parallel load here; cost is measured in 16_cost.py.

usage: 15_competitors_A.py [--seed0 500] [--seeds 10] [--tag main] [--jobs 12] [--automato]
"""
import argparse
import json
import subprocess
import tempfile
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from common import D, DADAPY_PY, RAW, ROOT, compute_truth, dump_json, env_info, save
from atdc.core import full_tree
from atdc.metrics import evaluate
from atdc.subsample import SubsampleTree
from atdc.truth_gmm import TruthGMM

warnings.filterwarnings("ignore")
RSCRIPT = ROOT / ".conda-env-r" / "bin" / "Rscript"
RENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PATH": f"{RSCRIPT.parent}:/usr/bin:/bin"}
DATASETS = (D.ALL_2D + D.HIGH_DIM + D.MANY + ["rare1", "rare5", "many16", "blobs5_d10"])

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed0", type=int, default=500)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--ms", type=int, nargs="*", default=[1000, 4000, 8000])
ap.add_argument("--datasets", nargs="*", default=DATASETS)
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--tag", default="main")
ap.add_argument("--automato", action="store_true")
ap.add_argument("--no_r", action="store_true")
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau


def label_metrics(labels, L):
    labels = np.asarray(labels)
    keep = labels >= 0
    return dict(n_est=int(len(np.unique(labels[keep]))), n_true=int(L.max() + 1),
                k_correct=float(len(np.unique(labels[keep])) == L.max() + 1),
                ari=float(adjusted_rand_score(L, labels)), nmi=float(normalized_mutual_info_score(L, labels)),
                frac_noise=float(np.mean(~keep)),
                ari_nonnoise=float(adjusted_rand_score(L[keep], labels[keep])) if keep.sum() > 1 else np.nan)


def run_r(script, X, seed):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.ascontiguousarray(X).tofile(tmp / "X.bin")
        (tmp / "meta.json").write_text(json.dumps(dict(n=len(X), d=X.shape[1], seed=int(seed))))
        subprocess.run([str(RSCRIPT), str(ROOT / "benchmarks" / "r" / script), str(tmp)], check=True, env=RENV,
                       capture_output=True, timeout=3600)
        out = json.loads((tmp / "out.json").read_text())
        return np.fromfile(tmp / "labels.bin", dtype=np.int32), out


def run_adp(X, Z=1.6):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.save(tmp / "X.npy", X)
        p = subprocess.run([str(DADAPY_PY), str(ROOT / "benchmarks" / "dadapy_worker.py"), str(tmp / "X.npy"),
                            str(tmp / "out.npz"), str(K + 1), str(Z), "PAk"], capture_output=True, text=True,
                           timeout=3600)
        if p.returncode:
            raise RuntimeError(p.stderr[-300:])
        o = np.load(tmp / "out.npz")
        return o["labels"], float(o["time"])


def leiden(X, k=15, resolution=1.0, seed=0):
    import igraph as ig
    import leidenalg
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    ind = cKDTree(X).query(X, k + 1)[1][:, 1:]
    src = np.repeat(np.arange(len(X)), k)
    e = np.unique(np.sort(np.stack([src, ind.ravel()], axis=1), axis=1), axis=0)
    g = ig.Graph(n=len(X), edges=e.tolist())
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution,
                                    seed=int(seed))
    return np.asarray(part.membership), time.perf_counter() - t0


def task(job):
    name, seed = job
    dist = D.make(name)
    X = dist.sample(n, np.random.default_rng(1000 + seed))
    truth = TruthGMM(dist, X) if name.startswith("blobs") else compute_truth(dist, X)
    L = truth.labels(tau)
    base = dict(dataset=name, d=X.shape[1], seed=seed, n=n)
    rows = []

    def add(method, group, labels, wall, est=None, **kw):
        r = dict(base, method=method, group=group, wall=wall, **label_metrics(labels, L), **kw)
        if est is not None:
            ev = evaluate(est, truth, tau)
            r.update({k: ev[k] for k in ("exact", "fp", "fn", "f1", "peak_h_mae", "saddle_mae",
                                        "triplet_recall", "mode_merge_distortion")})
        rows.append(r)

    def guarded(method, group, fn, **kw):
        try:
            fn()
        except Exception as ex:                           # recorded, never dropped silently
            rows.append(dict(base, method=method, group=group, error=repr(ex)[:300], **kw))

    for m in args.ms:
        t0 = time.perf_counter()
        e = SubsampleTree(X, m=m, K=K, seed=seed).estimate(tau)
        add(f"sub{m}", "same target", e.labels, time.perf_counter() - t0, e, param=f"m={m}")
    t0 = time.perf_counter()
    ef = full_tree(X, K, 10, tau)
    add("full", "same target", ef.labels, time.perf_counter() - t0, ef)

    def gudhi(density):
        from gudhi.clustering.tomato import Tomato
        t0 = time.perf_counter()
        if density == "w":
            tm = Tomato(graph_type="knn", density_type="manual", k=11, merge_threshold=tau, n_jobs=1)
            tm.fit(X, weights=ef.info["f"])
        else:
            tm = Tomato(graph_type="knn", density_type="logDTM", k=11, k_DTM=K + 1, merge_threshold=tau, n_jobs=1)
            tm.fit(X)
        add("tomato_w" if density == "w" else "tomato_logdtm", "same target" if density == "w" else "near target",
            np.asarray(tm.labels_), time.perf_counter() - t0,
            note="density given; time excludes the density" if density == "w" else "")
    guarded("tomato_w", "same target", lambda: gudhi("w"))
    guarded("tomato_logdtm", "near target", lambda: gudhi("logDTM"))

    def hdb(mcs, single):
        import hdbscan
        t0 = time.perf_counter()
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10, allow_single_cluster=single,
                             core_dist_n_jobs=1).fit(X)
        add("hdbscan", "other target", cl.labels_, time.perf_counter() - t0,
            param=f"mcs={mcs},single={int(single)}", default=float(mcs == max(n // 100, 10) and not single))
    for mcs in (max(n // 400, 5), max(n // 100, 10), max(n // 25, 20)):
        for single in (False, True):
            guarded("hdbscan", "other target", lambda: hdb(mcs, single), param=f"mcs={mcs},single={int(single)}")

    def lei(res):
        lab, w = leiden(X, 15, res, seed)
        add("leiden", "other target", lab, w, param=f"k=15,res={res}", default=float(res == 1.0))
    for res in (0.1, 0.3, 1.0):
        guarded("leiden", "other target", lambda: lei(res), param=f"res={res}")

    if not args.no_r:
        def dcut():
            lab, out = run_r("densitycut_run.R", X, seed)
            add("densitycut", "other target", lab, out["wall"], param="defaults", default=1.0)
        guarded("densitycut", "other target", dcut)
    if DADAPY_PY.exists():
        def adp(Z):
            lab, w = run_adp(X, Z)
            add("adp", "other target", lab, w, param=f"Z={Z}", default=float(Z == 1.65))
        for Z in (1.65, 3.0):
            guarded("adp", "other target", lambda: adp(Z), param=f"Z={Z}")
    if args.automato:
        def auto():
            from automato import Automato
            t0 = time.perf_counter()
            a = Automato(random_state=int(seed), parallelize=1).fit(X)
            add("automato", "near target", np.asarray(a.labels_), time.perf_counter() - t0, param="defaults", default=1.0)
        guarded("automato", "near target", auto)
    return rows


if __name__ == "__main__":
    jobs = [(d, s) for d in args.datasets for s in range(args.seed0, args.seed0 + args.seeds)]
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(i + 1, "/", len(jobs), flush=True)
                save(rows, f"15_competitors_A_{args.tag}")
    rows.sort(key=lambda r: (args.datasets.index(r["dataset"]), r["seed"], r["method"], str(r.get("param"))))
    save(rows, f"15_competitors_A_{args.tag}")
    dump_json({"args": vars(args), "env": env_info()}, RAW / f"15_competitors_A_{args.tag}.json")
