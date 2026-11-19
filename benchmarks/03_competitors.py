"""Comparison with existing methods through their own implementations.

  full            ToMATo-type reference: log-kNN density (K), kNN graph, tau
  tomato_gudhi    GUDHI Tomato, its default log-DTM density with k_DTM = K, same tau
  hdbscan_*       hdbscan package:
                    default  min_cluster_size=5 (package default), min_samples=None
                    mcs      min_cluster_size = n/100, min_samples = 10
                    K        min_samples = K (same bandwidth), min_cluster_size = K
                  plus the best ARI over a grid, reported as an ORACLE upper
                  bound (it uses the ground truth; not a usable setting)
  dadapy_*        DADApy Advanced Density Peaks, PAk density, Z = 1.65 (package
                  default) and Z = 3; maxk = K
  automato        AuToMATo with its defaults (only when --automato; B = 1000
                  bootstrap ToMATo runs)
  atdc / global / global_matched / coarse   as in 01_ablation.py

tau is a parameter of the prominence-based methods only.  HDBSCAN, ADP and
AuToMATo choose their clusters by their own criteria and are evaluated against
the same ground truth (true density tree pruned at tau).

usage: 03_competitors.py [--n 20000] [--seeds 10] [--automato]
"""
import argparse
import warnings

import numpy as np

from common import (D, compute_truth, dadapy_estimate, dump_json, env_info, matched_q, row,
                    run_atdc, run_coarse, run_dadapy, run_full, run_global, save, timed, RAW)
from atdc import baselines as B

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--q", type=float, default=0.1)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--z", type=float, default=0.0)
ap.add_argument("--z_saddle", type=float, default=0.0)
ap.add_argument("--t_top", type=int, default=5)
ap.add_argument("--automato", action="store_true")
ap.add_argument("--no_dadapy", action="store_true")
ap.add_argument("--tag", default="main")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM)
args = ap.parse_args()
n, K, q, tau, kg = args.n, args.K, args.q, args.tau, 10

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)
        meta = dict(dataset=name, seed=seed, n=n, d=X.shape[1], K=K, q=q)

        def add(res, heights=True, **kw):
            est, wall = res
            rows.append(row(est, truth, tau, wall, heights=heights, **meta, **kw))
            return est

        add(run_full(X, K, kg, tau))
        add(run_coarse(X, K, q, tau, seed, k_graph=kg))
        add(run_global(X, K, q, tau, seed, k_graph=kg))
        sel = add(run_atdc(X, K, q, tau, seed, z=args.z, z_saddle=args.z_saddle, t_top=args.t_top,
                           method="atdc", k_graph=kg))
        qg = matched_q(sel.cost_units, n, K, kg, 1)
        add(run_global(X, K, qg, tau, seed, k_graph=kg, method="global_matched"), q_matched=qg)
        add(timed(lambda: B.tomato_gudhi(X, K, kg, tau)), heights=False)
        add(timed(lambda: B.hdbscan_tree(X, None, 5, "hdbscan_default")))
        add(timed(lambda: B.hdbscan_tree(X, 10, max(n // 100, 10), "hdbscan_mcs")))
        add(timed(lambda: B.hdbscan_tree(X, K, K, "hdbscan_K")))
        best = None
        for ms in (10, 30, K):
            for mcs in (max(n // 400, 5), max(n // 100, 10), max(n // 25, 10)):
                est, wall = timed(lambda: B.hdbscan_tree(X, ms, mcs, "hdbscan_oracle"))
                r = row(est, truth, tau, wall, **meta, min_samples=ms, min_cluster_size=mcs)
                if best is None or r["ari"] > best["ari"]:
                    best = r
        rows.append(best)
        if not args.no_dadapy:
            for Z in (1.65, 3.0):
                out = run_dadapy(X, K, Z)
                if out is None or "error" in out:
                    print("dadapy failed:", None if out is None else out["error"][-300:], flush=True)
                    continue
                est = dadapy_estimate(out, n, f"dadapy_Z{Z:g}")
                rows.append(row(est, truth, tau, float(out["time"]), **meta))
        if args.automato:
            add(timed(lambda: B.automato_fit(X, seed=seed)), heights=False)
    print(name, "done", flush=True)
    save(rows, f"03_competitors_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"03_competitors_{args.tag}.json")
