"""Summary tables (results/summary/*.csv) and figures (results/figures/*.png)
from the raw CSVs written by scripts 01-05.  Nothing is typed in by hand;
a table or figure is skipped when its raw file does not exist.

usage: 06_report.py [--tag main] [--z_saddle 0] [--t_top 5]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import D, FIG, RAW, SUMMARY, compute_truth, run_atdc, run_coarse, run_full, run_global
from atdc import baselines as B

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--z_saddle", type=float, default=0.0)
ap.add_argument("--t_top", type=int, default=5)
args = ap.parse_args()
TAG = args.tag

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"full": "#52514e", "coarse": "#eda100", "global": "#2a78d6", "atdc": "#eb6834",
       "atdc_budget": "#eb6834", "random": "#e87ba4", "global_matched": "#1baf7a",
       "coarse_matched": "#eda100", "rag_all": "#6250d6", "tomato_gudhi": "#008300",
       "hdbscan_mcs": "#6250d6", "hdbscan_K": "#e34948"}
LAB = {"full": "full data (ToMATo-type)", "coarse": "coarse only (Stage A)",
       "global": "uniform refinement", "atdc": "ATDC selective", "atdc_budget": "ATDC, budget-limited",
       "random": "random refinement", "global_matched": "uniform, cost-matched",
       "tomato_gudhi": "ToMATo (GUDHI)", "hdbscan_mcs": "HDBSCAN (mcs=n/100)",
       "hdbscan_K": "HDBSCAN (min_samples=K)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})

METRICS = ["n_true", "n_est", "exact", "fp", "fn", "f1", "peak_h_mae", "saddle_mae",
           "mode_merge_distortion", "triplet_recall", "ari", "nmi", "cost", "wall", "frac_refined"]


def read(name):
    p = RAW / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def table(df, by, name):
    cols = [c for c in METRICS if c in df.columns]
    g = df.groupby(by, sort=False)[cols]
    out = g.mean().round(4)
    out["n_runs"] = g.size()
    out.to_csv(SUMMARY / f"{name}.csv")
    return out


# ------------------------------------------------------------------ tables
abl = read(f"01_ablation_{TAG}")
if abl is not None:
    table(abl, ["dataset", "method"], f"01_ablation_{TAG}_by_dataset")
    table(abl, ["method"], f"01_ablation_{TAG}_overall")
sen = read("02_sensitivity_dev")
if sen is not None:
    table(sen, ["varied", "method"], "02_sensitivity_dev_overall")
    table(sen, ["varied", "method", "dataset"], "02_sensitivity_dev_by_dataset")
cmp_ = read(f"03_competitors_{TAG}")
if cmp_ is not None:
    table(cmp_, ["dataset", "method"], f"03_competitors_{TAG}_by_dataset")
    table(cmp_, ["method"], f"03_competitors_{TAG}_overall")
trd = read(f"05_tradeoff_{TAG}")
if trd is not None:
    table(trd, ["family", "knob"], f"05_tradeoff_{TAG}_overall")
    table(trd, ["dataset", "family", "knob"], f"05_tradeoff_{TAG}_by_dataset")
scl = read("04_scaling")
if scl is not None and "status" in scl:
    ok = scl[scl.status == "ok"]
    tcols = [c for c in ok.columns if c.startswith("t_")]
    g = ok.groupby(["dataset", "n", "method"], sort=False)[["wall", "rss_added_mb", "rss_peak_mb",
                                                              "n_est", "frac_refined"] + tcols]
    g.mean().round(4).to_csv(SUMMARY / "04_scaling.csv")
    scl[scl.status != "ok"].groupby(["dataset", "n", "method"]).status.first().to_csv(
        SUMMARY / "04_scaling_failures.csv")

# ----------------------------------------------------------- fig: trade-off
if trd is not None:
    fams = ["coarse", "global", "atdc", "atdc_budget", "random"]
    for xcol, xlabel in (("cost", "neighbour evaluations (sum of k over kNN queries)"),
                         ("wall", "wall time [s]")):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (ycol, ylabel) in zip(axes, (("exact", "peak set exactly right (fraction of runs)"),
                                              ("ari", "ARI vs true density basins"),
                                              ("saddle_mae", "saddle log-density MAE"))):
            for f in fams:
                d = trd[trd.family == f].groupby("knob")[[xcol, ycol]].mean().sort_values(xcol)
                ax.plot(d[xcol], d[ycol], marker="o", ms=4, color=COL[f], label=LAB[f],
                        ls="--" if f in ("atdc_budget",) else "-")
            d = trd[trd.family == "full"][[xcol, ycol]].mean()
            ax.plot(d[xcol], d[ycol], marker="s", ms=6, color=COL["full"], ls="none", label=LAB["full"])
            ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_title(ylabel, loc="left")
        axes[0].legend(fontsize=7.5, loc="lower right")
        fig.suptitle(f"Accuracy vs cost, mean over data sets and seeds (n={int(trd.n.iloc[0])}, "
                     f"K={int(trd.K.iloc[0])}, tau={trd.tau.iloc[0]})", x=0.01, ha="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"05_tradeoff_{xcol}_{TAG}.png", dpi=150); plt.close(fig)
    # per data set, exact rate vs cost
    names = list(dict.fromkeys(trd.dataset))
    nc = 4; nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(12, 2.6 * nr), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), names):
        for f in ("global", "atdc", "atdc_budget", "random"):
            d = trd[(trd.family == f) & (trd.dataset == name)].groupby("knob")[["cost", "exact"]].mean().sort_values("cost")
            ax.plot(d.cost, d.exact, marker="o", ms=3, color=COL[f], label=LAB[f],
                    ls="--" if f == "atdc_budget" else "-")
        ax.set_xscale("log"); ax.set_title(name, loc="left"); ax.set_ylim(-0.05, 1.05)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("neighbour evaluations"); fig.supylabel("peak set exactly right")
    fig.tight_layout(); fig.savefig(FIG / f"05_tradeoff_by_dataset_{TAG}.png", dpi=150); plt.close(fig)

# ----------------------------------------------------------- fig: ablation
if abl is not None:
    order = ["coarse", "random", "coarse_matched", "global_matched", "atdc", "rag_all", "global", "full"]
    names = list(dict.fromkeys(abl.dataset))
    piv = abl.groupby(["dataset", "method"]).exact.mean().unstack()[order].loc[names]
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(names) + 1.6))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [o.replace("_", "\n") for o in order]); ax.set_yticks(range(len(names)), names)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 0.6 else INK, fontsize=8)
    ax.grid(False); ax.set_title("Fraction of runs with exactly the true set of peaks (tau-pruned)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(FIG / f"01_ablation_exact_{TAG}.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ fig: scaling
if scl is not None and "status" in scl:
    ok = scl[(scl.status == "ok") & (scl.dataset == "hier")]
    ms = ["coarse", "atdc", "global", "full", "tomato_gudhi", "hdbscan_mcs", "hdbscan_K"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, (col, lab) in zip(axes, (("wall", "wall time [s]"), ("rss_added_mb", "peak RSS above the data [MB]"))):
        for m in ms:
            d = ok[ok.method == m].groupby("n")[col].mean()
            if len(d):
                ax.plot(d.index, d.values.clip(1e-1 if col != "wall" else 1e-4), marker="o", ms=4,
                        color=COL[m], label=LAB[m], ls=":" if m.startswith("hdbscan") else "-")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_title(lab, loc="left")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Scaling with sample size (hier, d=2, K=100, q=0.1, single thread)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "04_scaling.png", dpi=150); plt.close(fig)
    # stage breakdown of ATDC and its uniform counterparts at the largest n
    nmax = ok.n.max()
    stages = ["t_index_sub", "t_knn_sub", "t_graph_tree", "t_index_full", "t_refine_query",
              "t_refine_tree", "t_label", "t_knn_full"]
    scol = ["#eda100", "#eb6834", "#e87ba4", "#86b6ef", "#2a78d6", "#6250d6", "#1baf7a", "#52514e"]
    d = ok[(ok.n == nmax) & ok.method.isin(["coarse", "atdc", "global", "full"])].groupby("method")[
        [s for s in stages if s in ok.columns]].mean().fillna(0).loc[["coarse", "atdc", "global", "full"]]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    left = np.zeros(len(d))
    for s, c in zip(stages, scol):
        if s in d:
            ax.barh(d.index, d[s], left=left, color=c, label=s[2:], edgecolor=SURF, linewidth=1.5, height=0.6)
            left += d[s].values
    ax.set_xscale("symlog", linthresh=1.0); ax.invert_yaxis(); ax.set_xlabel("seconds (linear below 1 s, log above)")
    ax.set_title(f"Time by stage, n={nmax:,}", loc="left"); ax.legend(fontsize=7, ncol=4, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "04_scaling_stages.png", dpi=150); plt.close(fig)

# ------------------------------------- fig: scatter of clusterings + trees
CL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
n, K, q, tau, kg, seed = 20000, 100, 0.1, 0.5, 10, 0


def colours(lab):
    lab = np.asarray(lab)
    c = np.array(["#b9b8b2"] * len(lab), dtype=object)
    for i, u in enumerate(np.unique(lab[lab >= 0])):
        c[lab == u] = CL[i % len(CL)] if i < len(CL) else "#52514e"
    return c


show = ["hier", "hetero", "moons", "circles", "unimodal"]
fig, axes = plt.subplots(len(show), 6, figsize=(15, 2.5 * len(show)))
trees = {}
for r, name in enumerate(show):
    dist = D.make(name)
    X = dist.sample(n, np.random.default_rng(1000 + seed))
    truth = compute_truth(dist, X)
    ests = [("truth (density basins)", None),
            ("coarse", run_coarse(X, K, q, tau, seed, k_graph=kg)[0]),
            ("ATDC selective", run_atdc(X, K, q, tau, seed, z_saddle=args.z_saddle, t_top=args.t_top, k_graph=kg)[0]),
            ("uniform refinement", run_global(X, K, q, tau, seed, k_graph=kg)[0]),
            ("full data", run_full(X, K, kg, tau)[0]),
            ("HDBSCAN (mcs=n/100)", B.hdbscan_tree(X, 10, n // 100))]
    sub = np.random.default_rng(0).choice(n, 4000, replace=False)
    for c, (title, est) in enumerate(ests):
        ax = axes[r, c]
        lab = truth.labels(tau) if est is None else est.labels
        ax.scatter(X[sub, 0], X[sub, 1], s=1.5, c=colours(lab)[sub], linewidths=0)
        if est is not None:
            ax.scatter(X[est.peak_idx, 0], X[est.peak_idx, 1], s=26, marker="^", c=INK, edgecolors=SURF, linewidths=0.8)
        k = len(np.unique(lab[lab >= 0]))
        ax.set_title(f"{title}: {k}" if r == 0 else f"{k} clusters", loc="left", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
        if c == 0:
            ax.set_ylabel(name, color=INK)
    if name == "hier":
        trees = {"truth": (truth.modes(tau)[1], truth.merge_matrix(tau)),
                 "ATDC selective": (ests[2][1].peak_h, ests[2][1].M),
                 "uniform refinement": (ests[3][1].peak_h, ests[3][1].M),
                 "full data": (ests[4][1].peak_h, ests[4][1].M),
                 "coarse": (ests[1][1].peak_h, ests[1][1].M)}
fig.suptitle(f"Cluster labels at tau={tau} (n={n}, K={K}, q={q}, seed {seed}); triangles = retained peaks; grey = noise",
             x=0.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / f"06_scatter_{TAG}.png", dpi=130); plt.close(fig)


def draw_tree(ax, h, M, title):
    """Merge tree: leaves at their peak height, joined at the merge heights."""
    k = len(h)
    lo = np.nanmin(M[np.isfinite(M)]) if np.isfinite(M).any() else h.min()
    # single-linkage order on similarity M
    groups = [[i] for i in np.argsort(-h)]
    xpos, x = {}, 0
    top = {}
    order = []
    import itertools
    S = M.copy(); np.fill_diagonal(S, -np.inf)
    comp = {i: [i] for i in range(k)}
    merges = []
    while len(comp) > 1:
        best = (-np.inf, None)
        for a, b in itertools.combinations(comp, 2):
            v = max(S[i, j] for i in comp[a] for j in comp[b])
            if v > best[0]:
                best = (v, (a, b))
        if best[1] is None or not np.isfinite(best[0]):
            break
        a, b = best[1]
        merges.append((a, b, best[0], list(comp[a]), list(comp[b])))
        comp[a] = comp[a] + comp.pop(b)
    leaves = [i for c in comp.values() for i in c]
    xpos = {leaf: i for i, leaf in enumerate(leaves)}
    cx = {i: (xpos[i], h[i]) for i in range(k)}
    node = {i: (xpos[i], h[i]) for i in range(k)}
    rep = {i: i for i in range(k)}
    for a, b, v, ca, cb in merges:
        xa, ya = node[a]; xb, yb = node[b]
        ax.plot([xa, xa], [ya, v], color=INK2, lw=1.5); ax.plot([xb, xb], [yb, v], color=INK2, lw=1.5)
        ax.plot([xa, xb], [v, v], color=INK2, lw=1.5)
        node[a] = ((xa + xb) / 2, v)
    for c in comp:
        xa, ya = node[c]
        ax.plot([xa, xa], [ya, lo - 0.6], color=INK2, lw=1.5)
    ax.scatter([xpos[i] for i in range(k)], h, s=22, color="#eb6834", zorder=3)
    ax.set_title(f"{title} ({k} leaves)", loc="left", fontsize=9); ax.set_xticks([]); ax.grid(axis="x", visible=False)


if trees:
    fig, axes = plt.subplots(1, len(trees), figsize=(3.0 * len(trees), 3.2), sharey=True)
    for ax, (t, (h, M)) in zip(axes, trees.items()):
        if len(h) <= 12:
            draw_tree(ax, np.asarray(h), np.asarray(M), t)
        else:
            ax.set_title(f"{t}: {len(h)} leaves (not drawn)", loc="left", fontsize=9)
    axes[0].set_ylabel("log-density")
    fig.suptitle(f"Merge tree pruned at tau={tau}, data set 'hier' (seed {seed})", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / f"06_trees_hier_{TAG}.png", dpi=150); plt.close(fig)
print("report written")
