"""Scaling with sample size and dimension: wall time by stage and peak RSS.

Every (method, n, d, seed) runs in a fresh process so that the peak resident
set size (ru_maxrss) belongs to that method alone; the data set is generated
inside the process and its size is reported as the baseline RSS before the
method starts.  No ground truth is computed here (accuracy is covered by the
other scripts); the number of clusters found is recorded as a sanity check.

usage: 04_scaling.py            # driver
       04_scaling.py --worker method dataset n seed
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np

from common import D, ROOT, dump_json, env_info, matched_q, save, RAW

K, Q, TAU, KG = 100, 0.1, 0.5, 10


def worker(method, dataset, n, seed, z_saddle, t_top):
    from atdc import baselines as B
    from atdc.core import ATDC, full_tree
    X = D.make(dataset).sample(n, np.random.default_rng(1000 + seed))
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    if method == "full":
        est = full_tree(X, K, KG, TAU)
    elif method == "coarse":
        est = ATDC(X, K=K, q=Q, seed=seed, k_graph=KG).stage_a().vertex_estimate(TAU, 0, method)
    elif method == "global":
        est = ATDC(X, K=K, q=Q, seed=seed, k_graph=KG).stage_a().vertex_estimate(TAU, 1, method)
    elif method == "atdc":
        a = ATDC(X, K=K, q=Q, seed=seed, k_graph=KG, t_top=t_top).stage_a()
        est = a.refine(TAU, z_saddle=z_saddle).estimate(TAU, method)
    elif method == "tomato_gudhi":
        est = B.tomato_gudhi(X, K, KG, TAU)
    elif method == "hdbscan_mcs":
        est = B.hdbscan_tree(X, 10, max(n // 100, 10), method)
    elif method == "hdbscan_K":
        est = B.hdbscan_tree(X, K, K, method)
    else:
        raise KeyError(method)
    wall = time.perf_counter() - t0
    if "total" in est.times:        # wrappers time the method itself, without
        wall = est.times["total"]   # the conversions done only for evaluation
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"method": method, "dataset": dataset, "n": n, "d": X.shape[1], "seed": seed,
           "wall": wall, "rss_before_mb": rss0 / 1024, "rss_peak_mb": rss1 / 1024,
           "rss_added_mb": (rss1 - rss0) / 1024, "n_est": int(len(est.peak_idx)),
           "cost": est.cost_units, "frac_refined": est.info.get("frac_refined"),
           "iterations": est.info.get("iterations")}
    out.update({"t_" + k: v for k, v in est.times.items()})
    print("RESULT " + json.dumps(out, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4)
    ap.add_argument("--z_saddle", type=float, default=2.0)
    ap.add_argument("--t_top", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=1_000_000)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    if args.worker:
        m, ds, n, seed = args.worker
        worker(m, ds, int(n), int(seed), args.z_saddle, args.t_top)
        sys.exit(0)
    methods = ["coarse", "atdc", "global", "full", "tomato_gudhi", "hdbscan_mcs", "hdbscan_K"]
    grid = [("hier", n) for n in (10_000, 30_000, 100_000, 300_000, 1_000_000) if n <= args.nmax]
    grid += [(ds, 100_000) for ds in ("hier_noise3", "gauss3_noise3", "gauss3_noise8")]
    rows, dead = [], set()
    for ds, n in grid:
        for m in methods:
            for seed in range(args.seeds):
                if (m, ds) in dead:
                    rows.append({"method": m, "dataset": ds, "n": n, "seed": seed, "status": "skipped_after_timeout"})
                    continue
                cmd = [sys.executable, __file__, "--worker", m, ds, str(n), str(seed),
                       "--z_saddle", str(args.z_saddle), "--t_top", str(args.t_top)]
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                    if line:
                        r = json.loads(line[0][7:]); r["status"] = "ok"
                    else:
                        r = {"method": m, "dataset": ds, "n": n, "seed": seed, "status": "error",
                             "stderr": p.stderr[-300:]}
                except subprocess.TimeoutExpired:
                    r = {"method": m, "dataset": ds, "n": n, "seed": seed, "status": "timeout"}
                    dead.add((m, ds))
                rows.append(r)
                print(r.get("status"), m, ds, n, seed, round(r.get("wall", float("nan")), 3), flush=True)
            save(rows, "04_scaling")
    dump_json({"K": K, "q": Q, "tau": TAU, "k_graph": KG, "args": vars(args), "env": env_info()},
              RAW / "04_scaling.json")
