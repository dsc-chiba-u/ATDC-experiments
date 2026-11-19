"""Phase 2, FINAL comparison on seeds never used before (300+).

Protocol fixed from the development rounds (07, 07b) before this script was
first run:

  primary candidate   v2 = AS: uniform refinement on m0 vertices + hill-climbing
                      ascent of near-threshold peaks + new vertices at the merge
                      location of near-threshold events (no sample splitting)
  primary comparison  v2(m0=200) against U(m=400) and U(m=1000): on the
                      development seeds U(400) took the same wall time and
                      U(1000) about twice as long
  primary metric      exact recovery of the set of true modes; secondary: ARI,
                      strict peak F1 (f1_top), saddle MAE, merge-order recall
  reported apart      gauss3_border (undecidable by construction) and many64

Also run: the uniform family over m (the frontier), v2 at other m0, v2 with the
held-out test (AS+T) and the test alone (T), v1, full data, and HDBSCAN.
Wall time of the U / v2 families is the median of ``--repeat`` runs; run this
script alone on the machine.

usage: 08_v2_final.py [--n 20000] [--seeds 10] [--K 100] [--tag main] [--datasets ...]
"""
import argparse
import warnings

import numpy as np

from common2 import (D, RAW, compute_truth, dump_json, env_info, row2, run_atdc, run_full, run_v2,
                     save)
from common import timed
from atdc import baselines as B

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--seed0", type=int, default=300)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--tag", default="main")
ap.add_argument("--no_full", action="store_true")
ap.add_argument("--no_hdbscan", action="store_true")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM + D.MANY)
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau
AS = dict(hidden=False, edges=False)

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)

        def add(fn, family, m0=np.nan, repeat=1, **kw):
            runs = [fn() for _ in range(repeat)]
            est = runs[0][0]
            wall = float(np.median([w for _, w in runs]))
            r = row2(est, truth, tau, wall, dist, X, dataset=name, seed=seed, n=n, K=K,
                     family=family, m0_req=m0, **kw)
            lab = [e.times.get("label", 0.0) for e, _ in runs]
            r["wall_tree"] = float(np.median([w - l for (_, w), l in zip(runs, lab)]))
            rows.append(r)

        R = args.repeat
        for m0 in (200, 400, 1000, 2000, 4000):
            add(lambda: run_v2(X, K, m0, tau, seed), "U", m0, R)
        for m0 in (200, 400, 1000):
            add(lambda: run_v2(X, K, m0, tau, seed, refine=AS), "AS", m0, R)
        for m0 in (400, 1000):
            add(lambda: run_v2(X, K, m0, tau, seed, rho=0.5, refine=AS), "AS+T", m0, R, rho=0.5)
        add(lambda: run_v2(X, K, 1000, tau, seed, rho=0.5), "T", 1000, R, rho=0.5)
        add(lambda: run_atdc(X, K, 0.1, tau, seed, z_saddle=2.0, method="v1", k_graph=10), "v1",
            repeat=R)
        if not args.no_full:
            add(lambda: run_full(X, K, 10, tau), "full")
        if not args.no_hdbscan:
            add(lambda: timed(lambda: B.hdbscan_tree(X, 10, max(n // 100, 10), "hdbscan",
                                                     allow_single_cluster=True)), "hdbscan_single")
            add(lambda: timed(lambda: B.hdbscan_tree(X, 10, max(n // 100, 10), "hdbscan")), "hdbscan")
    print(name, "done", flush=True)
    save(rows, f"08_v2_final_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"08_v2_final_{args.tag}.json")
