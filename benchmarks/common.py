"""Shared helpers for the benchmark scripts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # single thread for every method

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atdc import datasets as D                      # noqa: E402
from atdc.core import ATDC, TreeEstimate, full_tree  # noqa: E402
from atdc.metrics import evaluate                    # noqa: E402
from atdc.truth import compute_truth                 # noqa: E402

RAW = ROOT / "results" / "raw"
SUMMARY = ROOT / "results" / "summary"
FIG = ROOT / "results" / "figures"
for _p in (RAW, SUMMARY, FIG):
    _p.mkdir(parents=True, exist_ok=True)

DADAPY_PY = ROOT / ".conda-env-dadapy" / "bin" / "python"


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def row(est: TreeEstimate, truth, tau, wall, heights=True, **extra) -> dict:
    m = evaluate(est, truth, tau)
    if not heights:
        for k in ("peak_h_mae", "peak_h_bias", "saddle_mae", "saddle_bias", "saddle_max",
                  "mode_merge_distortion", "triplet_acc", "triplet_recall"):
            m[k] = np.nan
    if "total" in est.times:        # wrappers time the method itself, without
        wall = est.times["total"]   # the conversions done only for evaluation
    r = {"method": est.method, "tau": tau, "wall": wall, "cost": est.cost_units, **m}
    for k in ("frac_refined", "iterations", "n_unverified_retained", "n_basins", "m", "k0", "stop"):
        if k in est.info:
            r[k] = est.info[k]
    for k, v in est.times.items():
        r["t_" + k] = v
    r.update(extra)
    return r


# ---------------------------------------------------------------- ATDC family

def run_coarse(X, K, q, tau, seed, k_graph=None, method="coarse"):
    def f():
        a = ATDC(X, K=K, q=q, seed=seed, k_graph=k_graph).stage_a()
        return a.vertex_estimate(tau, 0, method)
    return timed(f)


def run_global(X, K, q, tau, seed, k_graph=None, method="global"):
    """Every subsample vertex gets the full-data density (uniform refinement)."""
    def f():
        a = ATDC(X, K=K, q=q, seed=seed, k_graph=k_graph).stage_a()
        return a.vertex_estimate(tau, 1, method)
    return timed(f)


def run_atdc(X, K, q, tau, seed, z=0.0, t_top=5, budget=None, mode="selective", method=None,
             midpoint=False, k_graph=None, z_saddle=0.0):
    def f():
        a = ATDC(X, K=K, q=q, seed=seed, t_top=t_top, midpoint=midpoint, k_graph=k_graph).stage_a()
        a.refine(tau, z=z, budget=budget, mode=mode, z_saddle=z_saddle)
        return a.estimate(tau, method or f"atdc_{mode}")
    return timed(f)


def matched_q(cost, n, K, k_graph, fidelity):
    """Subsample rate of a uniform method whose total cost equals ``cost``.

    fidelity 1 (global): cost = m*(k_graph + K) + n ;  fidelity 0 (coarse):
    cost = m*max(q K, k_graph) + n  (solved for the q K >= k_graph branch).
    """
    if fidelity == 1:
        q = (cost - n) / (n * (K + k_graph))
    else:
        q = np.sqrt(max(cost - n, 0.0) / (n * K))
    return float(min(max(q, 20.0 / n), 1.0))


def run_full(X, K, k_graph, tau):
    return timed(lambda: full_tree(X, K, k_graph, tau))


# ------------------------------------------------------------------- DADApy

def run_dadapy(X, maxk, Z, density="PAk", timeout=1800):
    """ADP clustering through DADApy in its own environment.  Returns a dict
    or None when the environment is missing."""
    if not DADAPY_PY.exists():
        return None
    with tempfile.TemporaryDirectory() as td:
        np.save(f"{td}/X.npy", X)
        cmd = [str(DADAPY_PY), str(ROOT / "benchmarks" / "dadapy_worker.py"), f"{td}/X.npy",
               f"{td}/out.npz", str(maxk), str(Z), density]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return {"error": p.stderr[-2000:]}
        o = np.load(f"{td}/out.npz", allow_pickle=True)
        return {k: o[k] for k in o.files}


def dadapy_estimate(out, n, method) -> TreeEstimate:
    from atdc.mergetree import merge_tree
    labels = out["labels"].astype(np.int64)
    centers = out["centers"].astype(np.int64)
    h = out["log_den"][centers]                   # already a normalised log-density
    bord = np.asarray(out["log_den_bord"], dtype=float)
    nc = len(centers)
    iu = np.triu_indices(nc, 1)
    ok = np.isfinite(bord[iu]) & (out["bord_valid"][iu] > 0)
    adj = np.stack([iu[0][ok], iu[1][ok]], axis=1)
    w = np.minimum(bord[iu][ok], np.minimum(h[adj[:, 0]], h[adj[:, 1]])) if len(adj) else np.empty(0)
    mt = merge_tree(h, adj, w, use_mst=False)
    death = mt.death_level()
    return TreeEstimate(method=method, peak_idx=centers, peak_h=h, peak_pers=h - death,
                        M=mt.merge_heights(np.arange(nc)), labels=labels,
                        cost_units=float(n * int(out["maxk"])), times={"total": float(out["time"])},
                        info={"id": float(out["id"])})


def save(rows, name):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RAW / f"{name}.csv", index=False)
    return df


def env_info():
    import platform
    import scipy
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "machine": platform.platform(),
            "threads": os.environ.get("OMP_NUM_THREADS")}


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
