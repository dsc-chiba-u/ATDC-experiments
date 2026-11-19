"""Runs DADApy's Advanced Density Peaks in the separate numpy<2 environment.

usage: dadapy_worker.py X.npy out.npz maxk Z {PAk|kNN}
"""
import sys
import time

import numpy as np
from dadapy import Data

x_path, out_path, maxk, Z, density = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
X = np.load(x_path)
t0 = time.perf_counter()
d = Data(X, verbose=False, n_jobs=1)
d.compute_distances(maxk=maxk)
d.compute_id_2NN()
if density == "PAk":
    d.compute_density_PAk()
else:
    d.compute_density_kNN(k=maxk - 1)
d.compute_clustering_ADP(Z=Z)
dt = time.perf_counter() - t0
bord = np.asarray(d.log_den_bord, dtype=float)
# borders that do not exist are stored as -1 with zero error in DADApy
bord_err = np.asarray(d.log_den_bord_err, dtype=float)
valid = (bord_err > 0).astype(int)
np.savez(out_path, labels=np.asarray(d.cluster_assignment), centers=np.asarray(d.cluster_centers),
         log_den=np.asarray(d.log_den), log_den_err=np.asarray(d.log_den_err), log_den_bord=bord,
         bord_valid=valid, time=dt, id=float(d.intrinsic_dim), maxk=maxk)
