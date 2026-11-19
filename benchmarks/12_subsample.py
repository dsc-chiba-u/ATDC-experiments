"""Phase A (3.1): how many vertices does the uniform subsample tree need?

For every data set and data seed, the subsample tree is fitted with m vertices,
B independent vertex draws per m.  Recorded per fit: recovery of the true tree
(metrics.evaluate_plus), whether the SMALLEST true cluster was found and how
many vertices fell into it, and two stability numbers that need no ground
truth:
  ari_same   ARI between the labelling of this draw and of the next draw at
             the same m (stability to the subsample);
  ari_next   ARI between this labelling and the one with 2m vertices.
Stability is not accuracy: a small cluster that no draw can see is missed
"stably".  The script records both so that the report can show where they part.

Variants (development only, --variants): k_graph and the midpoint edge check.
Wall times here are taken under parallel load and are not reported; cost is
measured by 12b_subsample_cost.py.

Seeds: development 400-404, report 500-509 (never used before Phase A).

usage: 12_subsample.py [--seed0 400] [--seeds 5] [--tag dev] [--jobs 16] [--variants base kg5 kg20 mid]
"""
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import D, RAW, compute_truth, dump_json, env_info, save
from atdc.core import full_tree
from atdc.metrics import evaluate_plus
from atdc.subsample import SubsampleTree

warnings.filterwarnings("ignore")
VAR = {"base": dict(), "kg5": dict(k_graph=5), "kg20": dict(k_graph=20), "mid": dict(edge_check=True),
       "kg5mid": dict(k_graph=5, edge_check=True)}
DATASETS = D.RARE + ["many16", "many64", "hetero", "hier", "gauss3_close", "moons", "circles", "unimodal"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed0", type=int, default=400)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--B", type=int, default=8)
ap.add_argument("--ms", type=int, nargs="*", default=[250, 500, 1000, 2000, 4000, 8000])
ap.add_argument("--variants", nargs="*", default=["base"])
ap.add_argument("--datasets", nargs="*", default=DATASETS)
ap.add_argument("--jobs", type=int, default=16)
ap.add_argument("--tag", default="dev")
args = ap.parse_args()
n, K, tau = args.n, args.K, args.tau


def small_cluster(truth, est):
    """Index of the smallest true cluster, its share, whether it was found."""
    L = truth.labels(tau)
    cnt = np.bincount(L, minlength=len(truth.modes(tau)[0]))
    a = int(np.argmin(cnt))
    return a, cnt[a] / len(L), L


def task(job):
    name, seed = job
    dist = D.make(name)
    X = dist.sample(n, np.random.default_rng(1000 + seed))
    truth = compute_truth(dist, X)
    a, share, L = small_cluster(truth, None)
    n_true = len(truth.modes(tau)[0])
    rows = []
    base = dict(dataset=name, seed=seed, n=n, K=K, n_true=n_true, small_share=share)
    e = full_tree(X, K, 10, tau)
    r = evaluate_plus(e, truth, tau, dist, X)
    rows.append({**base, **r, "variant": "full", "m": n, "b": 0,
                 "small_found": float(np.any(L[e.peak_idx] == a)) if n_true > 1 else np.nan})
    for vname in args.variants:
        lab = {}
        for m in args.ms:
            for b in range(args.B):
                t = SubsampleTree(X, m=m, K=K, seed=[seed, b], **VAR[vname])
                e = t.estimate(tau)
                r = evaluate_plus(e, truth, tau, dist, X)
                lab[m, b] = e.labels
                rows.append({**base, **r, "variant": vname, "m": m, "b": b,
                             "small_found": float(np.any(L[e.peak_idx] == a)) if n_true > 1 else np.nan,
                             "small_vertices": int(np.sum(L[t.idx] == a)),
                             "n_local_max": int(len(t.tree.dying) + len(t.tree.roots)),
                             "n_components": int(len(t.tree.roots)),
                             "cost": e.cost_units})
        k = len(rows) - len(args.ms) * args.B
        for i, m in enumerate(args.ms):
            for b in range(args.B):
                row = rows[k + i * args.B + b]
                row["ari_same"] = adjusted_rand_score(lab[m, b], lab[m, (b + 1) % args.B])
                if 2 * m in args.ms:
                    row["ari_next"] = adjusted_rand_score(lab[m, b], lab[2 * m, b])
                    row["n_est_next"] = int(lab[2 * m, b].max() + 1)
    return rows


if __name__ == "__main__":
    jobs = [(d, s) for d in args.datasets for s in range(args.seed0, args.seed0 + args.seeds)]
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(i + 1, "/", len(jobs), flush=True)
    rows.sort(key=lambda r: (args.datasets.index(r["dataset"]), r["seed"], r["variant"], r["m"], r["b"]))
    save(rows, f"12_subsample_{args.tag}")
    dump_json({"args": vars(args), "env": env_info()}, RAW / f"12_subsample_{args.tag}.json")
