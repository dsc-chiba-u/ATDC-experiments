"""Phase A (3.1, 3.2): wall time by stage and peak memory, one process per run.

What is varied
  data      synthetic (hier d=2, hier_noise8 d=10, hier_noise48 d=50) and real
            (pbmc68k PCA-50, ng20 MiniLM-384, ng20 mpnet-768; see data_prep.py)
  n         2e4, 2e5, 1e6 for the synthetic sets
  m         250 ... 8000 vertices
  backend   kdtree (exact), brute (exact, no index), hnsw, nndescent
            (approximate; index on all data)
Reference methods on the same data: the full-data tree of the same estimator
(``full``; exact neighbours), GUDHI ToMATo and HDBSCAN (both packages).

For the approximate back-ends the worker also reports, outside the timed part,
the recall of the K neighbours of the vertices, the largest change of a vertex
log-density and whether the retained peaks are the same as with exact search.

Stages: index_full (index on all data), query (K neighbours of the m vertices),
graph (kNN graph of the vertices), tree, label (nearest vertex of all points).

usage: 16_cost.py [--tag main] [--seeds 3]         # driver, run alone
       16_cost.py --worker method data n seed
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np

from common import D, RAW, ROOT, dump_json, env_info, save

K, TAU = 100, 0.5
REAL = {"pbmc68k": "pbmc68k_pca50.npy", "ng20_minilm": "ng20_minilm384.npy", "ng20_mpnet": "ng20_mpnet768.npy"}


def load(data, n, seed):
    if data in REAL:
        X = np.load(ROOT / "data" / REAL[data]).astype(float)
        return np.ascontiguousarray(X if n <= 0 or n >= len(X) else X[:n])
    return np.ascontiguousarray(D.make(data).sample(n, np.random.default_rng(1000 + seed)))


def worker(method, data, n, seed):
    from atdc.subsample import SubsampleTree
    X = load(data, n, seed)
    n, d = X.shape
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"method": method, "data": data, "n": n, "d": d, "seed": seed}
    t0 = time.perf_counter()
    if method.startswith("sub:"):
        _, backend, m = method.split(":")
        t = SubsampleTree(X, m=int(m), K=K, seed=seed, backend=backend)
        est = t.estimate(TAU)
        wall = time.perf_counter() - t0
        out.update(m=t.m, backend=backend, n_est=len(est.peak_idx), **{"t_" + k: v for k, v in est.times.items()})
    elif method == "full":
        from scipy.spatial import cKDTree
        from atdc.core import estimate_from_vertex_tree, knn_graph_edges
        from atdc.density import knn_log_density
        from atdc.mergetree import merge_tree
        from atdc.subsample import _brute_knn, _exact_radius
        if d <= 15:
            dist, ind = cKDTree(X).query(X, K + 1)
            r = dist[:, K]
        else:
            _, i = _brute_knn(X, X, K, exclude=np.arange(n))
            r = _exact_radius(X, X, i[:, -1:])[:, 0]
            ind = np.concatenate([np.arange(n)[:, None], i], axis=1)
        t1 = time.perf_counter()
        f = knn_log_density(r, K, n, d)
        est = estimate_from_vertex_tree("full", merge_tree(f, knn_graph_edges(ind, 10)), TAU, np.arange(n))
        wall = time.perf_counter() - t0
        out.update(m=n, backend="kdtree" if d <= 15 else "brute", n_est=len(est.peak_idx), t_query=t1 - t0,
                   t_tree=wall - (t1 - t0))
    elif method == "tomato_gudhi":
        from gudhi.clustering.tomato import Tomato
        tm = Tomato(graph_type="knn", density_type="logDTM", k=11, k_DTM=K + 1, merge_threshold=TAU, n_jobs=1)
        tm.fit(X)
        wall = time.perf_counter() - t0
        out.update(m=n, n_est=int(tm.n_clusters_))
    elif method.startswith("hdbscan"):
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=max(n // 100, 10), min_samples=10, core_dist_n_jobs=1).fit(X)
        wall = time.perf_counter() - t0
        out.update(m=n, n_est=int(cl.labels_.max() + 1), frac_noise=float(np.mean(cl.labels_ < 0)))
    else:
        raise KeyError(method)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.update(wall=wall, rss_added_mb=(rss1 - rss0) / 1024, rss_peak_mb=rss1 / 1024)
    if method.startswith("sub:") and backend in ("hnsw", "nndescent"):
        ex = SubsampleTree(X, m=int(m), K=K, seed=seed, backend="kdtree" if d <= 15 else "brute")
        out["recall"] = float(np.mean([len(np.intersect1d(a, b)) / K for a, b in zip(ex.nb_ind, t.nb_ind)]))
        out["max_abs_df"] = float(np.max(np.abs(ex.f - t.f)))
        out["mean_df"] = float(np.mean(t.f - ex.f))
        e2 = ex.estimate(TAU)
        out["same_peaks"] = bool(np.array_equal(np.sort(e2.peak_idx), np.sort(est.peak_idx)))
        out["n_est_exact"] = len(e2.peak_idx)
        from sklearn.metrics import adjusted_rand_score
        out["ari_vs_exact"] = float(adjusted_rand_score(e2.labels, est.labels))
    print("RESULT " + json.dumps(out, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    if args.worker:
        m, ds, n, seed = args.worker
        worker(m, ds, int(n), int(seed))
        sys.exit(0)
    grid = []
    for n in (20_000, 200_000, 1_000_000):
        for m in (250, 1000, 4000, 8000):
            grid.append((f"sub:kdtree:{m}", "hier", n))
        grid += [("full", "hier", n), ("tomato_gudhi", "hier", n)]
        if n <= 200_000:
            grid.append(("hdbscan", "hier", n))
    for data, ns in (("hier_noise8", (20_000, 200_000, 1_000_000)), ("hier_noise48", (20_000, 200_000, 1_000_000)),
                     ("pbmc68k", (0,)), ("ng20_minilm", (0,)), ("ng20_mpnet", (0,))):
        for n in ns:
            for m in (1000, 4000):
                for be in ("kdtree", "brute", "hnsw", "nndescent"):
                    if be == "kdtree" and data not in ("hier_noise8",):
                        continue                      # a k-d tree is not a sensible choice beyond d ~ 15
                    if be == "nndescent" and n == 1_000_000:
                        continue
                    grid.append((f"sub:{be}:{m}", data, n))
            if n <= 200_000:
                grid += [("full", data, n), ("hdbscan", data, n)]
            if n <= 20_000 or data == "pbmc68k":
                grid.append(("tomato_gudhi", data, n))
    rows = []
    for method, data, n in grid:
        for seed in range(700, 700 + args.seeds):
            try:
                p = subprocess.run([sys.executable, __file__, "--worker", method, data, str(n), str(seed)],
                                   capture_output=True, text=True, timeout=args.timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                r = json.loads(line[0][7:]) if line else {"method": method, "data": data, "n": n, "seed": seed,
                                                          "error": p.stderr[-300:]}
            except subprocess.TimeoutExpired:
                r = {"method": method, "data": data, "n": n, "seed": seed, "error": f"timeout {args.timeout}s"}
            rows.append(r)
            print(method, data, n, seed, round(r.get("wall", float("nan")), 3), r.get("error", ""), flush=True)
            if "error" in r:
                break                                  # do not repeat a failure or a time-out
        save(rows, f"16_cost_{args.tag}")
    dump_json({"K": K, "tau": TAU, "env": env_info()}, RAW / f"16_cost_{args.tag}.json")
