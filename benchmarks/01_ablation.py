"""Ablation of the selective refinement (ATDC family only).

For every data set and seed, at the same bandwidth K and threshold tau:

  coarse          Stage A only (subsample rate q)
  global          every subsample vertex refined with the full-data density
  rag_all         every basin peak / adjacency verified (ATDC without selection)
  atdc            selective refinement, run to its fixpoint
  random          random peaks / adjacencies verified, same number of fine
                  evaluations as atdc
  global_matched  uniform refinement on a smaller subsample, same total cost
  coarse_matched  Stage A only on a larger subsample, same total cost
  full            no subsampling (ToMATo with a log-kNN density)

The defaults (z = 0, z_saddle = 2, t_top = 5) were fixed on the development
seeds of 02_sensitivity.py.  The tag "v1" holds the first run, made with
z_saddle = 0 before that choice.

usage: 01_ablation.py [--n 20000] [--seeds 10] [--K 100] [--q 0.1] [--tau 0.5] [--tag main]
"""
import argparse

import numpy as np

from common import (D, compute_truth, dump_json, env_info, matched_q, row, run_atdc, run_coarse,
                    run_full, run_global, save, RAW)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--q", type=float, default=0.1)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--z", type=float, default=0.0)
ap.add_argument("--z_saddle", type=float, default=2.0)
ap.add_argument("--t_top", type=int, default=5)
ap.add_argument("--seed0", type=int, default=0)
ap.add_argument("--kg", type=int, default=10)
ap.add_argument("--tag", default="main")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM)
args = ap.parse_args()
n, K, q, tau, kg = args.n, args.K, args.q, args.tau, args.kg

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)
        meta = dict(dataset=name, seed=seed, n=n, d=X.shape[1], K=K, q=q)

        def add(res, **kw):
            est, wall = res
            rows.append(row(est, truth, tau, wall, **meta, **kw))
            return est

        add(run_full(X, K, kg, tau))
        add(run_coarse(X, K, q, tau, seed, k_graph=kg))
        add(run_global(X, K, q, tau, seed, k_graph=kg))
        add(run_atdc(X, K, q, tau, seed, mode="all", method="rag_all", k_graph=kg, t_top=args.t_top))
        sel = add(run_atdc(X, K, q, tau, seed, z=args.z, z_saddle=args.z_saddle, t_top=args.t_top,
                           method="atdc", k_graph=kg))
        add(run_atdc(X, K, q, tau, seed, mode="random", budget=sel.info["n_refined"],
                     method="random", k_graph=kg, t_top=args.t_top))
        qg = matched_q(sel.cost_units, n, K, kg, 1)
        add(run_global(X, K, qg, tau, seed, k_graph=kg, method="global_matched"), q_matched=qg)
        qc = matched_q(sel.cost_units, n, K, kg, 0)
        add(run_coarse(X, K, qc, tau, seed, k_graph=kg, method="coarse_matched"), q_matched=qc)
    print(name, "done", flush=True)
    save(rows, f"01_ablation_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"01_ablation_{args.tag}.json")
