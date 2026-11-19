"""Phase 2: wall time by stage and peak memory at large n, one process per run.

Methods: U(m) for m in {400, 1000, 4000}, v2 = A+S (m0 = 200, 400), v2 with the
held-out test (m0 = 400), v1, GUDHI ToMATo.  No ground truth here.

usage: 10_v2_scaling.py            # driver (run alone on the machine)
       10_v2_scaling.py --worker method dataset n seed
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np

from common import D, RAW, dump_json, env_info, save

K, TAU = 100, 0.5
AS = dict(hidden=False, edges=False)


def worker(method, dataset, n, seed):
    from atdc import baselines as B
    from atdc.core import ATDC
    from atdc.v2 import ATDC2
    X = D.make(dataset).sample(n, np.random.default_rng(1000 + seed))
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if method.startswith("U"):
        est = ATDC2(X, K=K, m0=int(method[1:]), seed=seed).estimate(TAU, method=method)
    elif method.startswith("AS+T"):
        est = ATDC2(X, K=K, m0=int(method[4:]), rho=0.5, seed=seed).refine(TAU, **AS).estimate(TAU, method=method)
    elif method.startswith("AS"):
        est = ATDC2(X, K=K, m0=int(method[2:]), seed=seed).refine(TAU, **AS).estimate(TAU, method=method)
    elif method == "v1":
        a = ATDC(X, K=K, q=0.1, seed=seed, k_graph=10).stage_a()
        est = a.refine(TAU, z_saddle=2.0).estimate(TAU, method)
    elif method == "tomato_gudhi":
        est = B.tomato_gudhi(X, K, 10, TAU)
    else:
        raise KeyError(method)
    wall = time.perf_counter() - t0
    if "total" in est.times:
        wall = est.times["total"]
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"method": method, "dataset": dataset, "n": n, "d": X.shape[1], "seed": seed, "wall": wall,
           "wall_tree": wall - est.times.get("label", 0.0), "rss_added_mb": (rss1 - rss0) / 1024,
           "rss_peak_mb": rss1 / 1024, "n_est": int(len(est.peak_idx)), "cost": est.cost_units,
           "n_vertices": est.info.get("n_vertices")}
    out.update({"t_" + k: v for k, v in est.times.items()})
    print("RESULT " + json.dumps(out, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    if args.worker:
        m, ds, n, seed = args.worker
        worker(m, ds, int(n), int(seed))
        sys.exit(0)
    methods = ["U400", "U1000", "U4000", "AS200", "AS400", "AS+T400", "v1", "tomato_gudhi"]
    grid = [("hier", n) for n in (20_000, 100_000, 1_000_000)]
    grid += [("gauss3_noise8", 100_000), ("hier_noise3", 100_000)]
    rows = []
    for ds, n in grid:
        for m in methods:
            for seed in range(300, 300 + args.seeds):
                p = subprocess.run([sys.executable, __file__, "--worker", m, ds, str(n), str(seed)],
                                   capture_output=True, text=True, timeout=1800)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                r = json.loads(line[0][7:]) if line else {"method": m, "dataset": ds, "n": n,
                                                          "seed": seed, "error": p.stderr[-300:]}
                rows.append(r)
                print(m, ds, n, seed, round(r.get("wall", float("nan")), 3), flush=True)
            save(rows, "10_v2_scaling")
    dump_json({"K": K, "tau": TAU, "env": env_info()}, RAW / "10_v2_scaling.json")
