"""Audit of the evaluation (Phase 2, before the new experiments).

(a) Discretisation of the true tree: the reference graph (data + n_extra
    points with the true density, k neighbours, midpoint caps) is recomputed
    at 4x the points and 2x the neighbours; reported are the largest changes of
    the true peak heights, prominences and merge heights, and whether the set
    of true modes at tau changes.
(b) Stability of the true labels of the data points under that change (ARI).

usage: 09_audit.py
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import D, RAW, compute_truth, dump_json, env_info, save

rows = []
for name in D.ALL_2D + D.HIGH_DIM + D.MANY:
    dist = D.make(name)
    X = dist.sample(20000, np.random.default_rng(1200))
    a = compute_truth(dist, X)                                  # setting used everywhere
    b = compute_truth(dist, X, n_extra=160000, k=32, seed=777)  # finer, different points
    for tau in (0.3, 0.5, 1.0):
        pa, ha, ra = a.modes(tau)
        pb, hb, rb = b.modes(tau)
        r = dict(dataset=name, tau=tau, n_modes=len(pa), n_modes_fine=len(pb))
        if len(pa) == len(pb):
            # match modes by location
            d = np.linalg.norm(a.ref[pa][:, None] - b.ref[pb][None], axis=2)
            m = d.argmin(1)
            ok = len(set(m.tolist())) == len(pa)
            r["matched"] = ok
            if ok:
                Ma, Mb = a.merge_matrix(tau), b.merge_matrix(tau)[np.ix_(m, m)]
                fin = np.isfinite(Ma) & np.isfinite(Mb)
                r["max_dh"] = float(np.max(np.abs(ha - hb[m])))
                r["max_dM"] = float(np.max(np.abs(Ma - Mb)[fin]))
                fr = np.isfinite(ra) & np.isfinite(rb[m])
                r["max_dpers"] = float(np.max(np.abs(ra - rb[m])[fr])) if fr.any() else 0.0
                r["max_dloc"] = float(d.min(1).max())
                r["min_pers_margin"] = float(np.min(np.abs(ra[fr] - tau))) if fr.any() else np.nan
        r["label_ari"] = float(adjusted_rand_score(a.labels(tau), b.labels(tau)))
        rows.append(r)
    print(name, flush=True)
df = save(rows, "09_audit_truth")
dump_json({"env": env_info(), "coarse": dict(n_extra=40000, k=16), "fine": dict(n_extra=160000, k=32)},
          RAW / "09_audit_truth.json")
print(df.round(4).to_string())
