"""Accuracy against computational cost, at a fixed bandwidth K and threshold tau.

Each family is swept along its own cost knob:

  coarse(q)        Stage A only, subsample rate q
  global(q)        uniform refinement: all q*n subsample vertices get the
                   full-data density (the strongest uniform alternative)
  atdc(q)          selective refinement run to its fixpoint
  atdc_budget(b)   selective refinement at q = 0.1 stopped after b fine evaluations
  random(b)        random verification at q = 0.1 with the same budgets
  full             no subsampling (single point)

usage: 05_tradeoff.py [--n 20000] [--seeds 10] [--z_saddle 0] [--t_top 5]
"""
import argparse

import numpy as np

from common import (D, compute_truth, dump_json, env_info, row, run_atdc, run_coarse, run_full,
                    run_global, save, RAW)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--z", type=float, default=0.0)
ap.add_argument("--z_saddle", type=float, default=0.0)
ap.add_argument("--t_top", type=int, default=5)
ap.add_argument("--tag", default="main")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM)
args = ap.parse_args()
n, K, tau, kg = args.n, args.K, args.tau, 10
QS_UNI = (0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.2, 0.4)
QS_SEL = (0.04, 0.07, 0.1, 0.2, 0.4)
BUDGET_FRAC = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)     # of the subsample size at q = 0.1

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)

        def add(res, family, knob):
            est, wall = res
            rows.append(row(est, truth, tau, wall, dataset=name, seed=seed, n=n, K=K,
                            family=family, knob=knob))

        add(run_full(X, K, kg, tau), "full", 1.0)
        for q in QS_UNI:
            add(run_coarse(X, K, q, tau, seed, k_graph=kg), "coarse", q)
            add(run_global(X, K, q, tau, seed, k_graph=kg), "global", q)
        for q in QS_SEL:
            add(run_atdc(X, K, q, tau, seed, z=args.z, z_saddle=args.z_saddle, t_top=args.t_top,
                         method="atdc", k_graph=kg), "atdc", q)
        m = int(round(0.1 * n))
        for b in BUDGET_FRAC:
            add(run_atdc(X, K, 0.1, tau, seed, z=args.z, z_saddle=args.z_saddle, t_top=args.t_top,
                         budget=int(b * m), method="atdc_budget", k_graph=kg), "atdc_budget", b)
            add(run_atdc(X, K, 0.1, tau, seed, mode="random", budget=int(b * m), t_top=args.t_top,
                         method="random", k_graph=kg), "random", b)
    print(name, "done", flush=True)
    save(rows, f"05_tradeoff_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"05_tradeoff_{args.tag}.json")
