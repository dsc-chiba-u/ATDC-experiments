"""Phase 2, DEVELOPMENT seeds (200+): which v2 ingredients help, at what cost.

Never used for the reported comparison (08_v2_final.py, seeds 300+).
Seeds 0-9 and 100-104 were used in Phase 1 and are not reused.

Families, each over a grid of initial vertex counts m0:
  U       uniform refinement (full-data density at m0 random vertices)
  U+T     + sample splitting and held-out test of the events (rho)
  U+G     + adaptive graph refinement (hidden peaks, ascent, edges, saddles)
  U+G+T   both
plus single-ingredient ablations at fixed m0, v1 and full data.

usage: 07_v2_dev.py [--n 20000] [--seeds 5] [--K 100] [--tag dev] [--datasets ...]
"""
import argparse

import numpy as np

from common2 import (D, RAW, compute_truth, dump_json, env_info, row2, run_atdc, run_full, run_v2,
                     save)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--seed0", type=int, default=200)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--tag", default="dev")
ap.add_argument("--light", action="store_true", help="U and U+T only")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM + D.MANY)
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau
M_GRID = (100, 200, 400, 1000, 2000)
OFF = dict(hidden=False, ascent=False, edges=False, saddles=False)

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)

        def add(res, family, m0=np.nan, **kw):
            est, wall = res
            rows.append(row2(est, truth, tau, wall, dist, X, dataset=name, seed=seed, n=n, K=K,
                             family=family, m0_req=m0, **kw))

        for m0 in M_GRID:
            add(run_v2(X, K, m0, tau, seed), "U", m0)
            add(run_v2(X, K, m0, tau, seed, rho=0.5), "U+T", m0, rho=0.5)
            add(run_v2(X, K, m0, tau, seed, rho=0.3), "U+T3", m0, rho=0.3)
            if args.light or m0 > 1000:
                continue
            add(run_v2(X, K, m0, tau, seed, refine={}), "U+G", m0)
            add(run_v2(X, K, m0, tau, seed, rho=0.5, refine={}), "U+G+T", m0, rho=0.5)
        if not args.light:
            for m0 in (200, 400):
                for key in ("hidden", "ascent", "edges", "saddles"):
                    add(run_v2(X, K, m0, tau, seed, refine={**OFF, key: True}), f"U+{key}", m0)
                add(run_v2(X, K, m0, tau, seed, refine=dict(saddles=False)), "U+G-saddles", m0)
                add(run_v2(X, K, m0, tau, seed, refine=dict(hidden=False)), "U+G-hidden", m0)
                add(run_v2(X, K, m0, tau, seed, rho=0.5, refine=dict(hidden=False, edges=False)),
                    "U+AS+T", m0, rho=0.5)
            est, wall = run_atdc(X, K, 0.1, tau, seed, z_saddle=2.0, method="v1", k_graph=10)
            add((est, wall), "v1")
            add(run_full(X, K, 10, tau), "full")
    print(name, "done", flush=True)
    save(rows, f"07_v2_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"07_v2_{args.tag}.json")
