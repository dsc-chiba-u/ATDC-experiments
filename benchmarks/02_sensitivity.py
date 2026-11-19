"""Parameter sensitivity of the selective refinement, on DEVELOPMENT seeds.

These seeds (100+) are disjoint from the seeds of the reported comparisons
(0-9); design choices (z, z_saddle, t_top) were made from this script only.

Varied one at a time around the base configuration:
  z         confidence multiplier of the candidate rule
  z_saddle  optimism for unverified saddles
  t_top     number of top vertices / crossing edges re-evaluated per item
  midpoint  additionally evaluate the fine density at edge midpoints
  q, K, tau subsample rate, bandwidth, prominence threshold
For q, K and tau the uniform alternatives are rerun as well, because they
change the problem and not only ATDC.

usage: 02_sensitivity.py [--n 20000] [--seeds 5]
"""
import argparse

import numpy as np

from common import (D, compute_truth, dump_json, env_info, matched_q, row, run_atdc, run_coarse,
                    run_full, run_global, save, RAW)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--seed0", type=int, default=100)
ap.add_argument("--tag", default="dev")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + ["gauss3_noise3"])
args = ap.parse_args()
n, kg = args.n, 10
BASE = dict(K=100, q=0.1, tau=0.5, z=0.0, z_saddle=0.0, t_top=5, midpoint=False)

atdc_only = ([dict(z=v) for v in (0.5, 1.0, 2.0)] +
             [dict(z_saddle=v) for v in (0.5, 1.0, 2.0, 3.0)] +
             [dict(t_top=v) for v in (1, 2, 10, 20)] +
             [dict(midpoint=True)] +
             [dict(z_saddle=2.0, t_top=10), dict(z_saddle=2.0, z=0.5), dict(z_saddle=1.0, t_top=10)])
problem = ([dict(q=v) for v in (0.05, 0.2, 0.4)] + [dict(K=v) for v in (30, 50, 200)] +
           [dict(tau=v) for v in (0.3, 0.75, 1.0)])

rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)

        def add(res, cfg, varied, **kw):
            est, wall = res
            rows.append(row(est, truth, cfg["tau"], wall, dataset=name, seed=seed, n=n,
                            varied=varied, **{k: cfg[k] for k in BASE if k != "tau"}, **kw))
            return est

        def atdc(cfg, varied):
            return add(run_atdc(X, cfg["K"], cfg["q"], cfg["tau"], seed, z=cfg["z"],
                                z_saddle=cfg["z_saddle"], t_top=cfg["t_top"],
                                midpoint=cfg["midpoint"], method="atdc", k_graph=kg), cfg, varied)

        def uniform(cfg, varied, sel):
            K, q, tau = cfg["K"], cfg["q"], cfg["tau"]
            add(run_full(X, K, kg, tau), cfg, varied)
            add(run_coarse(X, K, q, tau, seed, k_graph=kg), cfg, varied)
            add(run_global(X, K, q, tau, seed, k_graph=kg), cfg, varied)
            qg = matched_q(sel.cost_units, n, K, kg, 1)
            add(run_global(X, K, qg, tau, seed, k_graph=kg, method="global_matched"), cfg, varied,
                q_matched=qg)

        sel = atdc(BASE, "base")
        uniform(BASE, "base", sel)
        for ov in atdc_only:
            atdc({**BASE, **ov}, ",".join(f"{k}={v}" for k, v in ov.items()))
        for ov in problem:
            cfg = {**BASE, **ov}
            v = ",".join(f"{k}={v}" for k, v in ov.items())
            uniform(cfg, v, atdc(cfg, v))
    print(name, "done", flush=True)
    save(rows, f"02_sensitivity_{args.tag}")
dump_json({"args": vars(args), "base": BASE, "env": env_info()}, RAW / f"02_sensitivity_{args.tag}.json")
