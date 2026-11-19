"""Phase A (3.3, 8): the same estimator in R and in Python on the same data and
the same vertices.

Python: src/atdc/subsample.py (SciPy cKDTree, NumPy, Python union-find loop on
the reduced edge list).  R: benchmarks/r/atdc_proto.R (RANN / FNN / dbscan for
the neighbours; union-find loop in R or in C++ through Rcpp).  The vertex
indices are drawn once in Python and passed to R, so that differences are
differences of the implementations, not of the random number generators.

Reported per configuration: maximum absolute difference of the vertex
densities, identity of the graph, of the retained peaks and of the labelling,
and wall time by stage (best of --repeat; run alone).

usage: 14_cross_language.py [--tag main]
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import D, RAW, ROOT, dump_json, env_info, save
from atdc.subsample import SubsampleTree

RSCRIPT = ROOT / ".conda-env-r" / "bin" / "Rscript"
ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--repeat", type=int, default=3)
ap.add_argument("--tag", default="main")
ap.add_argument("--configs", nargs="*", default=["hier:20000:1000", "hier:20000:8000", "many64:20000:4000",
                                                 "hier:200000:1000", "hier:200000:8000", "hier:1000000:1000",
                                                 "hier_noise8:20000:1000", "hier_noise8:200000:1000",
                                                 "blobs5_d20:20000:1000", "rare1:20000:4000"])
ap.add_argument("--nn", nargs="*", default=["RANN", "FNN", "dbscan"],
                help="R neighbour search: RANN, FNN, dbscan (k-d trees) or matrix (base R matrix products)")
ap.add_argument("--backend", default="kdtree", choices=["kdtree", "auto"],
                help="Python search for the vertex densities; auto = matrix products above 15 dimensions")
args = ap.parse_args()

rows = []
for cfg in args.configs:
    name, n, m = cfg.split(":"); n, m = int(n), int(m)
    dist = D.make(name)
    X = np.ascontiguousarray(dist.sample(n, np.random.default_rng(1000 + 600)))
    best = None
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        t = SubsampleTree(X, m=m, K=args.K, seed=600,
                          backend="brute" if args.backend == "auto" and X.shape[1] > 15 else "kdtree")
        e = t.estimate(args.tau)
        wall = time.perf_counter() - t0
        if best is None or wall < best[0]:
            best = (wall, t, e)
    wall, t, e = best
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        X.tofile(tmp / "X.bin")
        (tmp / "meta.json").write_text(json.dumps(dict(
            n=n, d=X.shape[1], K=args.K, k_graph=10, tau=args.tau, idx=t.idx.tolist(), cpp=True,
            nn=args.nn, repeat=args.repeat)))
        t0 = time.perf_counter()
        subprocess.run([str(RSCRIPT), str(ROOT / "benchmarks" / "r" / "cross_language.R"), str(tmp)],
                       check=True, env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PATH": f"{RSCRIPT.parent}:/usr/bin:/bin"})
        r_proc = time.perf_counter() - t0
        out = json.loads((tmp / "r_out.json").read_text())
        lab_r = np.fromfile(tmp / "labels_r.bin", dtype=np.int32)
    lab_py = e.peak_idx[e.labels]
    base = dict(dataset=name, n=n, d=X.shape[1], m=m, K=args.K, tau=args.tau)
    cmp = dict(max_abs_df=float(np.max(np.abs(np.array(out["f"]) - t.f))),
               same_edges=bool(np.array_equal(np.array(out["edges"]), t.edges)),
               same_peaks=bool(np.array_equal(np.sort(e.peak_idx), np.array(out["peaks"]))),
               n_peaks_py=len(e.peak_idx), n_peaks_r=len(out["peaks"]),
               labels_identical=bool(np.array_equal(lab_py, lab_r)),
               ari_py_r=float(adjusted_rand_score(lab_py, lab_r)),
               n_merges_py=len(t.tree.level), n_merges_r=out["n_merges"])
    py_t = {k: v for k, v in e.times.items()}
    rows.append({**base, **cmp, "impl": "python", "backend": t.backend, "total": wall,
                 **{"t_" + k: v for k, v in py_t.items()}})
    for key in [f"{a}_{b}" for a in args.nn for b in ("r", "cpp")]:
        if key in out:
            rows.append({**base, **cmp, "impl": "R:" + key, "total": out[key]["total"],
                         **{"t_" + k: v for k, v in out[key]["times"].items()}, "r_process": r_proc})
    print(cfg, cmp, flush=True)
    save(rows, f"14_cross_language_{args.tag}")
dump_json({"args": vars(args), "env": env_info(),
           "R": subprocess.run([str(RSCRIPT), "-e", "cat(R.version.string, as.character(packageVersion('RANN')), "
                                "as.character(packageVersion('FNN')), as.character(packageVersion('dbscan')), "
                                "as.character(packageVersion('Rcpp')))"], capture_output=True, text=True).stdout},
          RAW / f"14_cross_language_{args.tag}.json")
