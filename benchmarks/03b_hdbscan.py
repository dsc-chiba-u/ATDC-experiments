"""Supplement to 03_competitors.py for HDBSCAN only (added afterwards).

03 scores every method by ARI over all points, which counts HDBSCAN's noise
points as errors, and runs it with allow_single_cluster=False, which cannot
return one cluster.  This script reports, on the same data and seeds,
  * ARI restricted to the points HDBSCAN labels (ari_nonnoise),
  * the variants with allow_single_cluster=True,
  * the best setting of a grid by each of the two ARIs (ORACLE: uses the truth).

usage: 03b_hdbscan.py [--n 20000] [--seeds 10]
"""
import argparse
import warnings

import numpy as np

from common import D, compute_truth, dump_json, env_info, row, save, timed, RAW
from atdc import baselines as B

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM)
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau
rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)
        grid = []
        for single in (False, True):
            for ms in (10, 30, K):
                for mcs in (max(n // 400, 5), max(n // 100, 10), max(n // 25, 10)):
                    est, wall = timed(lambda: B.hdbscan_tree(X, ms, mcs, "hdbscan", allow_single_cluster=single))
                    grid.append(row(est, truth, tau, wall, dataset=name, seed=seed, n=n, min_samples=ms,
                                    min_cluster_size=mcs, allow_single=single))
        def pick(f, label):
            r = dict(next(g for g in grid if f(g))); r["method"] = label; rows.append(r)
        pick(lambda g: g["min_samples"] == 10 and g["min_cluster_size"] == max(n // 100, 10) and not g["allow_single"], "hdbscan_mcs")
        pick(lambda g: g["min_samples"] == 10 and g["min_cluster_size"] == max(n // 100, 10) and g["allow_single"], "hdbscan_mcs_single")
        pick(lambda g: g["min_samples"] == K and g["min_cluster_size"] == max(n // 100, 10) and g["allow_single"], "hdbscan_K_single")
        for key in ("ari", "ari_nonnoise"):
            best = dict(max(grid, key=lambda g: -1 if np.isnan(g[key]) else g[key]))
            best["method"] = f"hdbscan_oracle_{key}"
            rows.append(best)
    print(name, "done", flush=True)
    save(rows, "03b_hdbscan_main")
dump_json({"args": vars(args), "env": env_info()}, RAW / "03b_hdbscan_main.json")
