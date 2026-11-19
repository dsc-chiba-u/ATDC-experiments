"""Phase A (3.2): recovery of the true tree as the dimension grows.

Three synthetic families with an exact density:
  gauss3_noiseN, hier_noiseN   2-D structure plus N independent N(0,1) noise
                               coordinates (d = 2 ... 50); truth = tree of the
                               2-D density (truth.compute_truth)
  blobs5_dD                    five isotropic Gaussians of unequal scale and
                               weight whose structure fills D dimensions;
                               truth by modal EM and the segment skeleton
                               (truth_gmm.TruthGMM)

For every fit (m vertices, B vertex draws) the tree is pruned on a grid of
thresholds.  Recorded per threshold: recovery of the truth pruned at the SAME
threshold, agreement with the reference partition (truth at tau = 0.5), and
the ARI with the next vertex draw (stability, no ground truth needed).  The
full-data tree (same density and graph on all n points) is the reference for
what subsampling loses; whatever it gets wrong is statistical, not
computational.

Wall times are taken under parallel load and are not reported (see 13b).
Seeds: development 400-404, report 500-509.

usage: 13_dimension.py [--seed0 400] [--seeds 5] [--tag dev] [--jobs 16]
"""
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import adjusted_rand_score

from common import D, RAW, compute_truth, dump_json, env_info, save
from atdc.core import estimate_from_vertex_tree, knn_graph_edges
from atdc.density import knn_log_density
from atdc.mergetree import merge_tree
from atdc.metrics import evaluate
from atdc.subsample import SubsampleTree, _brute_knn, _exact_radius, intrinsic_dim_mle
from atdc.truth_gmm import TruthGMM

warnings.filterwarnings("ignore")
NOISE = [0, 1, 3, 8, 18, 48]
DATASETS = (["gauss3"] + [f"gauss3_noise{k}" for k in NOISE[1:]] + ["hier"] + [f"hier_noise{k}" for k in NOISE[1:]]
            + [f"blobs5_d{d}" for d in (5, 10, 20, 50)])
TAUS = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--seed0", type=int, default=400)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--K", type=int, default=100)
ap.add_argument("--B", type=int, default=4)
ap.add_argument("--ms", type=int, nargs="*", default=[1000, 4000])
ap.add_argument("--datasets", nargs="*", default=DATASETS)
ap.add_argument("--jobs", type=int, default=12)
ap.add_argument("--tag", default="dev")
args = ap.parse_args()
n, K = args.n, args.K


class FullTree:
    """Same estimator with every data point as a vertex (exact neighbours)."""

    def __init__(self, X, dim=None):
        nn, d = X.shape
        if d <= 15:
            from scipy.spatial import cKDTree
            dist, ind = cKDTree(X).query(X, K + 1)
            r, ind = dist[:, K], ind
            self.d_hat = intrinsic_dim_mle(dist[:, 1:])
        else:
            dd, i = _brute_knn(X, X, K, exclude=np.arange(nn))
            r = _exact_radius(X, X, i[:, -1:])[:, 0]
            ind = np.concatenate([np.arange(nn)[:, None], i], axis=1)
            self.d_hat = intrinsic_dim_mle(dd)
        self.f = knn_log_density(r, K, nn, self.d_hat if dim == "auto" else d)
        self.tree = merge_tree(self.f, knn_graph_edges(ind, 10))
        self.n = nn

    def estimate(self, tau):
        return estimate_from_vertex_tree("full", self.tree, tau, np.arange(self.n))


def task(job):
    name, seed = job
    dist = D.make(name)
    X = dist.sample(n, np.random.default_rng(1000 + seed))
    truth = TruthGMM(dist, X) if name.startswith("blobs") else compute_truth(dist, X)
    Lref = truth.labels(0.5)
    n_ref = len(truth.modes(0.5)[0])
    base = dict(dataset=name, d=X.shape[1], seed=seed, n=n, K=K, n_true_ref=n_ref)
    rows = []

    def record(est, tau, **kw):
        r = evaluate(est, truth, tau)
        r.update(base, tau=tau, ari_ref=adjusted_rand_score(Lref, est.labels), **kw)
        rows.append(r)
        return r

    for dim in (None, "auto"):
        dn = "ambient" if dim is None else "auto"
        ft = FullTree(X, dim)
        pf = np.sort(ft.tree.persistence)[::-1][:30]
        for tau in TAUS:
            record(ft.estimate(tau), tau, method="full", m=n, b=0, dim=dn, d_hat=ft.d_hat,
                   top_prom=" ".join(f"{v:.3f}" for v in pf) if tau == TAUS[0] else "")
        for m in args.ms:
            fits = [SubsampleTree(X, m=m, K=K, seed=[seed, b], dim=dim) for b in range(args.B)]
            for tau in TAUS:
                est = [t.estimate(tau) for t in fits]
                for b, (t, e) in enumerate(zip(fits, est)):
                    p = np.sort(t.tree.persistence)[::-1][:30]
                    record(e, tau, method="subsample", m=m, b=b, dim=dn, d_hat=t.d_hat,
                           n_components=len(t.tree.roots),
                           ari_same=adjusted_rand_score(e.labels, est[(b + 1) % args.B].labels),
                           top_prom=" ".join(f"{v:.3f}" for v in p) if tau == TAUS[0] else "")
    return rows


if __name__ == "__main__":
    jobs = [(d, s) for d in args.datasets for s in range(args.seed0, args.seed0 + args.seeds)]
    jobs.sort(key=lambda j: -D.make(j[0]).dim)            # the slow ones first
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(task, jobs)):
            rows += r
            if (i + 1) % 10 == 0:
                print(i + 1, "/", len(jobs), flush=True)
    rows.sort(key=lambda r: (args.datasets.index(r["dataset"]), r["seed"], r["dim"], r["method"], r["m"], r["b"],
                             r["tau"]))
    save(rows, f"13_dimension_{args.tag}")
    dump_json({"args": vars(args), "env": env_info(), "taus": TAUS}, RAW / f"13_dimension_{args.tag}.json")
