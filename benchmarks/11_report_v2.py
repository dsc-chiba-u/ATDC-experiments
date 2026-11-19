"""Phase 2 tables and figures from results/raw/07*, 08*, 09*, 10*.

Writes results/summary/07*, 08*, 10* and results/figures/08*, 10*.  Phase-1
outputs are not touched.

Paired analysis: two configurations are compared on the same (data set, seed)
runs; the mean difference gets a 95 % bootstrap interval that resamples seeds
within each data set (2 000 resamples), and wins / ties / losses are counted.

usage: 11_report_v2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from common import FIG, RAW, SUMMARY

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
COL = {"U": "#2a78d6", "AS": "#eb6834", "AS+T": "#1baf7a", "T": "#eda100", "v1": "#e87ba4",
       "full": "#52514e"}
LAB = {"U": "uniform refinement (m vertices)", "AS": "v2: ascent + saddle vertices (m0)",
       "AS+T": "v2 + held-out test", "T": "held-out test only", "v1": "v1 (Phase 1)",
       "full": "full data (ToMATo-type)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})
COLS = ["exact", "fp", "fn", "f1", "f1_top", "top_deficit", "loc_err", "peak_h_mae", "saddle_mae",
        "triplet_recall", "ari", "nmi", "cost", "wall", "wall_tree", "n_vertices"]
APART = ["gauss3_border", "many64"]


def read(name):
    p = RAW / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def main_sets(df):
    return df[~df.dataset.isin(APART)]


def tab(df, by, name):
    c = [x for x in COLS if x in df.columns]
    g = df.groupby(by, sort=False, dropna=False)[c]
    out = g.mean().round(4)
    out["n_runs"] = g.size()
    out.to_csv(SUMMARY / f"{name}.csv")
    return out


def paired(df, a, b, metrics=("exact", "ari", "f1_top", "saddle_mae", "triplet_recall", "wall", "wall_tree"),
           n_boot=2000, seed=0):
    """a, b = (family, m0).  Returns one row per metric: mean(a) - mean(b)."""
    def pick(key):
        fam, m0 = key
        x = df[df.family == fam]
        if m0 is not None:
            x = x[x.m0_req == m0]
        return x.set_index(["dataset", "seed"])
    A, B = pick(a), pick(b)
    idx = A.index.intersection(B.index)
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        d = (A.loc[idx, m] - B.loc[idx, m]).dropna()
        if d.empty:
            continue
        groups = [g.values for _, g in d.groupby(level="dataset")]
        boot = [np.mean(np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(n_boot)]
        rows.append(dict(a=f"{a[0]}({a[1]})", b=f"{b[0]}({b[1]})", metric=m, mean_a=A.loc[d.index, m].mean(),
                         mean_b=B.loc[d.index, m].mean(), diff=d.mean(), ci_lo=np.percentile(boot, 2.5),
                         ci_hi=np.percentile(boot, 97.5), wins=int((d > 1e-12).sum()),
                         ties=int((d.abs() <= 1e-12).sum()), losses=int((d < -1e-12).sum()), n=len(d)))
    return rows


PAIRS = [(("AS", 200), ("U", 400)), (("AS", 200), ("U", 1000)), (("AS", 400), ("U", 1000)),
         (("AS", 1000), ("U", 2000)), (("AS", 1000), ("U", 4000)), (("AS+T", 400), ("U", 1000)),
         (("AS+T", 400), ("AS", 400)), (("T", 1000), ("U", 1000)), (("v1", None), ("U", 1000)),
         (("AS", 200), ("v1", None)), (("AS", 400), ("full", None))]

for tag in ("main", "n200k", "K50"):
    df = read(f"08_v2_final_{tag}")
    if df is None:
        continue
    tab(main_sets(df), ["family", "m0_req"], f"08_v2_final_{tag}_overall")
    tab(df, ["dataset", "family", "m0_req"], f"08_v2_final_{tag}_by_dataset")
    for ds in APART:
        if (df.dataset == ds).any():
            tab(df[df.dataset == ds], ["family", "m0_req"], f"08_v2_final_{tag}_{ds}")
    rows = []
    for a, b in PAIRS:
        rows += [dict(scope="12 data sets" if tag != "n200k" else "8 data sets", **r)
                 for r in paired(main_sets(df), a, b)]
        for ds in APART:
            if (df.dataset == ds).any():
                rows += [dict(scope=ds, **r) for r in paired(df[df.dataset == ds], a, b,
                                                             metrics=("exact", "f1", "ari", "wall"))]
    pd.DataFrame(rows).round(4).to_csv(SUMMARY / f"08_v2_final_{tag}_paired.csv", index=False)

    # ------------------------------------------------ frontier: accuracy vs wall time
    x = main_sets(df)
    for xcol, xlab in (("wall", "wall time, total [s]"), ("wall_tree", "wall time without labelling [s]")):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, (ycol, ylab) in zip(axes, (("exact", "true mode set exactly recovered"),
                                            ("f1_top", "peak F1, peak within 0.25 of the top"),
                                            ("ari", "ARI vs true density basins"),
                                            ("saddle_mae", "saddle log-density MAE"))):
            for fam in ("U", "AS", "AS+T", "T", "v1", "full"):
                g = x[x.family == fam].groupby("m0_req", dropna=False)[[xcol, ycol]].mean().sort_values(xcol)
                if g.empty:
                    continue
                ax.plot(g[xcol], g[ycol], marker="o" if len(g) > 1 else "s", ms=4 if len(g) > 1 else 6,
                        ls="-" if len(g) > 1 else "none", color=COL[fam], label=LAB[fam])
                if len(g) > 1 and ycol == "exact":
                    for m0, r in g.iterrows():
                        ax.annotate(f"{int(m0)}", (r[xcol], r[ycol]), textcoords="offset points",
                                    xytext=(3, -9), fontsize=6.5, color=INK2)
            ax.set_xscale("log"); ax.set_xlabel(xlab); ax.set_title(ylab, loc="left")
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[0].legend(fontsize=7.5, loc="lower right")
        fig.suptitle(f"Accuracy against measured wall time, final seeds, n={int(df.n.iloc[0])}, "
                     f"K={int(df.K.iloc[0])}, tau={df.tau.iloc[0]} (labels: number of initial vertices)",
                     x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(FIG / f"08_frontier_{xcol}_{tag}.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ per data set exact recovery
    cfg = [("U", 400), ("U", 1000), ("U", 4000), ("AS", 200), ("AS", 400), ("AS", 1000), ("AS+T", 400),
           ("T", 1000), ("v1", None), ("full", None)]
    names = list(dict.fromkeys(df.dataset))
    mat = np.full((len(names), len(cfg)), np.nan)
    for j, (fam, m0) in enumerate(cfg):
        s = df[df.family == fam]
        s = s if m0 is None else s[s.m0_req == m0]
        col = "f1" if False else "exact"
        v = s.groupby("dataset")[col].mean()
        for i, nm in enumerate(names):
            if nm in v.index:
                mat[i, j] = v[nm]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(names) + 1.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cfg)), [f"{f}\n{'' if m is None else m}" for f, m in cfg])
    ax.set_yticks(range(len(names)), names)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.6 else INK, fontsize=8)
    ax.grid(False); ax.set_title("Fraction of runs with exactly the true set of modes", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(FIG / f"08_exact_by_dataset_{tag}.png", dpi=150); plt.close(fig)

    # ------------------------------------------------ paired differences
    pr = pd.DataFrame(rows)
    pr = pr[(pr.scope.str.contains("data sets")) & pr.metric.isin(["exact", "ari", "f1_top"])]
    if not pr.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 0.33 * pr.a.nunique() * 2 + 1.5), sharey=True)
        for ax, m in zip(axes, ("exact", "ari", "f1_top")):
            s = pr[pr.metric == m].reset_index(drop=True)
            y = np.arange(len(s))[::-1]
            ax.errorbar(s["diff"], y, xerr=[s["diff"] - s.ci_lo, s.ci_hi - s["diff"]], fmt="o", ms=4,
                        color="#2a78d6", ecolor="#86b6ef", elinewidth=2, capsize=0)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(y, [f"{a} − {b}" for a, b in zip(s.a, s.b)])
            ax.set_title(f"difference in {m} (95% bootstrap CI)", loc="left")
        fig.tight_layout(); fig.savefig(FIG / f"08_paired_{tag}.png", dpi=150); plt.close(fig)

for name in ("07_v2_dev", "07b_v2_dev2"):
    df = read(name)
    if df is not None:
        tab(main_sets(df), ["family", "m0_req"], f"{name}_overall")
        tab(df, ["dataset", "family", "m0_req"], f"{name}_by_dataset")

sc = read("10_v2_scaling")
if sc is not None and "wall" in sc:
    tc = [c for c in sc.columns if c.startswith("t_")]
    g = sc.groupby(["dataset", "n", "method"], sort=False)[["wall", "wall_tree", "rss_added_mb", "n_est",
                                                             "n_vertices", "cost"] + tc]
    g.mean().round(4).to_csv(SUMMARY / "10_v2_scaling.csv")
    h = sc[sc.dataset == "hier"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cm = {"U400": "#86b6ef", "U1000": "#2a78d6", "U4000": "#104281", "AS200": "#eb6834", "AS400": "#eda100",
          "AS+T400": "#1baf7a", "v1": "#e87ba4", "tomato_gudhi": "#52514e"}
    for ax, (col, lab) in zip(axes, (("wall", "wall time [s]"), ("rss_added_mb", "peak RSS above the data [MB]"))):
        for m, c in cm.items():
            d = h[h.method == m].groupby("n")[col].mean()
            if len(d):
                ax.plot(d.index, d.values.clip(1e-1 if col != "wall" else 1e-4), marker="o", ms=4, color=c, label=m)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_title(lab, loc="left")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Scaling with n (hier, d=2, K=100, single thread)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "10_v2_scaling.png", dpi=150); plt.close(fig)
print("report v2 written")
