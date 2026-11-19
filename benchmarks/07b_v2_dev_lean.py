"""Phase 2, second DEVELOPMENT round (seeds 205-214): the lean v2 variants.

Round one (07_v2_dev.py) showed that hidden-peak checks and edge verification
cost several hundred density evaluations and do not pay in low dimension,
whereas ascent (A), saddle densification (S) and the held-out test (T) are
cheap.  This round chooses among the cheap combinations and their parameters.

usage: 07b_v2_dev_lean.py [--n 20000] [--seeds 10] [--tag dev2] [--datasets ...]
"""
import argparse

import numpy as np

from common2 import D, RAW, compute_truth, dump_json, env_info, row2, run_v2, save

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--seed0", type=int, default=205)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--tag", default="dev2")
ap.add_argument("--datasets", nargs="*", default=D.ALL_2D + D.HIGH_DIM + D.MANY)
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau
AS = dict(hidden=False, edges=False)
VARIANTS = {
    "AS+T": (0.5, AS), "AS+T3": (0.3, AS), "AS": (None, AS),
    "A+T": (0.5, dict(hidden=False, edges=False, saddles=False)),
    "S+T": (0.5, dict(hidden=False, edges=False, ascent=False)),
    "AS+T_m50": (0.5, dict(hidden=False, edges=False, margin=0.5)),
    "ASE+T": (0.5, dict(hidden=False)),
    "T": (0.5, None),
}
rows = []
for name in args.datasets:
    dist = D.make(name)
    for seed in range(args.seed0, args.seed0 + args.seeds):
        X = dist.sample(n, np.random.default_rng(1000 + seed))
        truth = compute_truth(dist, X)

        def add(res, family, m0, **kw):
            est, wall = res
            rows.append(row2(est, truth, tau, wall, dist, X, dataset=name, seed=seed, n=n, K=K,
                             family=family, m0_req=m0, **kw))

        for m0 in (200, 400, 1000, 2000, 4000):
            add(run_v2(X, K, m0, tau, seed), "U", m0)
            if m0 > 2000:
                continue
            for fam, (rho, ref) in VARIANTS.items():
                if fam not in ("AS+T", "AS", "T") and m0 not in (400, 1000):
                    continue
                add(run_v2(X, K, m0, tau, seed, rho=rho, refine=ref), fam, m0, rho=rho)
    print(name, "done", flush=True)
    save(rows, f"07b_v2_{args.tag}")
dump_json({"args": vars(args), "env": env_info()}, RAW / f"07b_v2_{args.tag}.json")
