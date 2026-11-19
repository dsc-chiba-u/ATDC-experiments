"""Helpers for the Phase-2 benchmarks (v2).  Phase-1 helpers are untouched."""
from __future__ import annotations

import time

import numpy as np

from common import D, RAW, compute_truth, dump_json, env_info, run_atdc, run_full, save  # noqa: F401
from atdc.metrics import evaluate_plus
from atdc.v2 import ATDC2

LEAN = dict()                                   # refine() defaults


def run_v2(X, K, m0, tau, seed, rho=None, refine=None, method="U", k_graph=10, z=0.0):
    """One ATDC2 run.  refine=None is plain uniform refinement on m0 vertices
    (identical to the Phase-1 'global' control, same code path as v2)."""
    t0 = time.perf_counter()
    a = ATDC2(X, K=K, m0=m0, k_graph=k_graph, rho=rho, seed=seed)
    if refine is not None:
        a.refine(tau, **refine)
    est = a.estimate(tau, method=method, z=z)
    return est, time.perf_counter() - t0


def row2(est, truth, tau, wall, dist, X, **extra):
    m = evaluate_plus(est, truth, tau, dist, X)
    r = {"method": est.method, "tau": tau, "wall": wall, "cost": est.cost_units,
         "wall_tree": wall - est.times.get("label", 0.0), **m}
    for k in ("n_vertices", "m0", "rounds", "n_ascent", "n_edge_checks", "n_inserted", "n_saddle_ins",
              "n_hidden_checks", "n_tested", "n_in_sample", "K1", "K2", "frac_refined", "iterations"):
        if k in est.info:
            r[k] = est.info[k]
    for k, v in est.times.items():
        r["t_" + k] = v
    r.update(extra)
    return r
