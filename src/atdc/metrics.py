"""Evaluation of an estimated pruned merge tree against the true one.

Kept separate on purpose:
  peak detection  - are the true modes (log-prominence >= tau) found, are
                    there extra peaks?
  peak height     - error of the estimated log-density at matched peaks
  saddle / merge  - error of the merge heights between matched peaks
  merge order     - for triples of modes, which two merge first
  labelling       - ARI / NMI against the basins of the TRUE density pruned at
                    the same tau (not against the generating components)
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate(est, truth, tau, margin=0.05) -> dict:
    p_true, h_true, pers_true = truth.modes(tau)
    L = truth.labels(tau)
    M_true = truth.merge_matrix(tau)
    n_true = len(p_true)
    n_est = len(est.peak_idx)
    where = L[est.peak_idx]                       # true basin of each estimated peak
    match = {}
    for a in range(n_true):
        c = np.flatnonzero(where == a)
        if c.size:
            match[a] = int(c[np.argmax(est.peak_h[c])])
    tp = len(match)
    fp = n_est - tp
    fn = n_true - tp
    out = {
        "n_true": n_true, "n_est": n_est, "tp": tp, "fp": fp, "fn": fn,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0,
        "exact": float(fp == 0 and fn == 0),
    }
    det = sorted(match)
    out["peak_h_mae"] = float(np.mean([abs(est.peak_h[match[a]] - h_true[a]) for a in det])) if det else np.nan
    out["peak_h_bias"] = float(np.mean([est.peak_h[match[a]] - h_true[a] for a in det])) if det else np.nan
    errs = [est.M[match[a], match[b]] - M_true[a, b] for a, b in combinations(det, 2)
            if np.isfinite(M_true[a, b])]
    errs = [e for e in errs if np.isfinite(e)]
    out["saddle_mae"] = float(np.mean(np.abs(errs))) if errs else np.nan
    out["saddle_bias"] = float(np.mean(errs)) if errs else np.nan
    out["saddle_max"] = float(np.max(np.abs(errs))) if errs else np.nan
    out["n_saddle_pairs"] = len(errs)
    # merge distortion restricted to matched modes (peaks on the diagonal)
    dist = [abs(est.peak_h[match[a]] - h_true[a]) for a in det] + [abs(e) for e in errs]
    out["mode_merge_distortion"] = float(np.max(dist)) if dist else np.nan
    # merge order on triples
    good = tot = 0
    for a, b, c in combinations(det, 3):
        t = np.array([M_true[a, b], M_true[a, c], M_true[b, c]])
        o = np.sort(t)
        if o[2] - o[1] <= margin:
            continue                  # no unique first merge in the truth
        e = np.array([est.M[match[a], match[b]], est.M[match[a], match[c]], est.M[match[b], match[c]]])
        tot += 1
        good += int(np.argmax(e) == np.argmax(t) and np.sum(e == e.max()) == 1)
    out["triplets"] = tot
    out["triplet_acc"] = good / tot if tot else np.nan
    # all true triples, counting undetected modes as failures
    tot_all = 0
    for a, b, c in combinations(range(n_true), 3):
        t = np.sort([M_true[a, b], M_true[a, c], M_true[b, c]])
        tot_all += int(t[2] - t[1] > margin)
    out["triplets_true"] = tot_all
    out["triplet_recall"] = good / tot_all if tot_all else np.nan
    if est.labels is not None:
        out["ari"] = float(adjusted_rand_score(L, est.labels))
        out["nmi"] = float(normalized_mutual_info_score(L, est.labels))
        keep = np.asarray(est.labels) >= 0
        out["frac_noise"] = float(np.mean(~keep))
        # methods that leave points unlabelled: agreement on the labelled ones
        out["ari_nonnoise"] = float(adjusted_rand_score(L[keep], np.asarray(est.labels)[keep])) \
            if keep.sum() > 1 else np.nan
    return out


def evaluate_plus(est, truth, tau, dist, X, top_tol=0.25) -> dict:
    """:func:`evaluate` plus peak *location* accuracy (added in Phase 2).

    ``evaluate`` counts a true mode as detected when an estimated peak lies
    anywhere in its basin.  That says nothing about where in the basin.  Here,
    for every detected mode:
      top_deficit  true log-density at the mode minus true log-density at the
                   estimated peak (0 = exactly at the top; scale-free);
      loc_err      Euclidean distance between the two in the structured
                   coordinates;
    and ``tp_top`` counts the modes whose estimated peak is within ``top_tol``
    of the top in log-density, with ``f1_top`` the F1 score using that
    stricter notion of a true positive.
    """
    out = evaluate(est, truth, tau)
    p_true, h_true, _ = truth.modes(tau)
    L = truth.labels(tau)
    where = L[est.peak_idx]
    c = dist.core_dim
    f_at = dist.logpdf(np.asarray(X)[est.peak_idx])
    defi, loc = [], []
    for a in range(len(p_true)):
        cand = np.flatnonzero(where == a)
        if cand.size:
            j = cand[np.argmax(est.peak_h[cand])] if np.all(np.isfinite(est.peak_h[cand])) else cand[0]
            defi.append(h_true[a] - f_at[j])
            loc.append(np.linalg.norm(np.asarray(X)[est.peak_idx[j], :c] - truth.ref[p_true[a]]))
    defi = np.array(defi)
    tp_top = int(np.sum(defi <= top_tol))
    n_true, n_est = len(p_true), len(est.peak_idx)
    out.update(top_deficit=float(defi.mean()) if defi.size else np.nan,
               loc_err=float(np.mean(loc)) if loc else np.nan, tp_top=tp_top,
               f1_top=2 * tp_top / (n_true + n_est) if (n_true + n_est) else 1.0)
    return out
