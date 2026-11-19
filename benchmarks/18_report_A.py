"""Phase A tables and figures from results/raw/12*-17*.

Writes results/summary/A_* and results/figures/A_*.  Earlier outputs are not
touched.  Each block is skipped when its raw file is missing.

usage: 18_report_A.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from common import FIG, RAW, SUMMARY

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, GREEN, YELLOW, PINK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
                     "grid.linewidth": 0.6, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})


def read(name):
    p = RAW / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def logx(ax, ticks=None):
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
    if ticks is not None:
        ax.set_xticks(ticks, [f"{t:g}" if t < 1000 else f"{t / 1000:g}k" for t in ticks]); ax.minorticks_off()


# ------------------------------------------------------------------ 12: subsample size
df = read("12_subsample_final")
if df is not None:
    cols = ["exact", "fn", "fp", "f1", "f1_top", "ari", "nmi", "peak_h_mae", "saddle_mae", "triplet_recall",
            "small_found", "ari_same", "ari_next"]
    g = df.groupby("m")[cols]
    out = g.mean().round(4); out["n_runs"] = g.size()
    out.to_csv(SUMMARY / "A_12_by_m.csv")
    df.groupby(["dataset", "m"], sort=False)[cols + ["n_true", "n_est", "small_share"]].mean().round(4) \
        .to_csv(SUMMARY / "A_12_by_dataset_m.csv")
    s = df[df.variant == "base"].copy()
    rare = s[s.dataset.str.startswith("rare")].copy()
    rare["expected_vertices"] = rare.m * rare.small_share
    bins = [0, 3, 6, 9, 12, 16, 22, 30, 45, 1e4]
    det = rare.groupby(pd.cut(rare.expected_vertices, bins), observed=True).small_found.agg(["mean", "size"])
    det.to_csv(SUMMARY / "A_12_small_cluster_detection.csv")
    actual = rare.groupby(pd.cut(rare.small_vertices, [-1, 2, 5, 8, 11, 15, 20, 30, 50, 1e4]),
                          observed=True).small_found.agg(["mean", "size"])
    actual.to_csv(SUMMARY / "A_12_small_cluster_by_vertices.csv")
    # doubling rule (start 1000, double while the count or the partition changes, cap 8000)
    si = s.set_index(["dataset", "seed", "b", "m"])
    rows = []
    for (ds, seed, b), g in si.groupby(level=[0, 1, 2]):
        g = g.droplevel([0, 1, 2]); m = 1000
        while m < 8000 and not (g.loc[m, "n_est"] == g.loc[m, "n_est_next"] and g.loc[m, "ari_next"] >= 0.95):
            m *= 2
        rows.append(dict(dataset=ds, m_selected=m, exact=g.loc[m, "exact"], small_found=g.loc[m, "small_found"],
                         exact_m1000=g.loc[1000, "exact"], exact_m4000=g.loc[4000, "exact"],
                         exact_m8000=g.loc[8000, "exact"]))
    d = pd.DataFrame(rows)
    pd.concat([d.groupby("dataset", sort=False).mean(), d.drop(columns="dataset").mean().to_frame("ALL").T]) \
        .round(3).to_csv(SUMMARY / "A_12_doubling_rule.csv")
    # stability is not accuracy: same count at 2m although the small cluster is missed
    r2 = rare[rare.n_est_next.notna()].copy()
    r2["same_count"] = r2.n_est == r2.n_est_next
    r2.groupby(["dataset", "m"]).apply(lambda x: pd.Series({
        "missed": 1 - x.small_found.mean(), "ari_same": x.ari_same.mean(), "ari_next": x.ari_next.mean(),
        "p_same_count": x.same_count.mean(),
        "missed_given_same_count": 1 - x.small_found[x.same_count].mean() if x.same_count.any() else np.nan})) \
        .round(3).to_csv(SUMMARY / "A_12_stable_but_missed.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    full = df[df.variant == "full"]
    groups = {"rare05 (0.5 %)": (["rare05"], PINK), "rare1 (1 %)": (["rare1"], ORANGE), "rare2 (2 %)": (["rare2"], YELLOW),
              "many16": (["many16"], GREEN),
              "other 7 sets": (["rare5", "hetero", "hier", "gauss3_close", "moons", "circles", "unimodal"], BLUE)}
    ax = axes[0]
    for lab, (names, c) in groups.items():
        v = s[s.dataset.isin(names)].groupby("m").exact.mean()
        ax.plot(v.index, v.values, marker="o", ms=4, color=c, label=lab)
        ax.plot([20000], [full[full.dataset.isin(names)].exact.mean()], marker="s", ms=6, color=c)
    logx(ax, [250, 500, 1000, 2000, 4000, 8000, 20000]); ax.set_xlabel("vertices m  (square: all 20 000 points)")
    ax.set_title("True mode set exactly recovered", loc="left")
    ax.legend(fontsize=7.5, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax = axes[1]
    x = np.arange(len(actual)); ax.bar(x, actual["mean"], color=BLUE, width=0.7)
    ax.set_xticks(x, ["≤2", "3–5", "6–8", "9–11", "12–15", "16–20", "21–30", "31–50", ">50"])
    ax.axvline(3.5, color=INK2, lw=0.8, ls="--")
    ax.text(3.4, 1.03, "more than k_graph = 10 →", fontsize=8, color=INK2, ha="left", va="bottom")
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("vertices that fell into the small cluster"); ax.set_title("Small cluster found (rare sets)", loc="left")
    ax = axes[2]
    for name, c in (("rare05", PINK), ("rare1", ORANGE), ("rare2", YELLOW)):
        v = rare[rare.dataset == name].groupby("m")[["small_found", "ari_same"]].mean()
        ax.plot(v.index, v.small_found, marker="o", ms=4, color=c, label=f"{name}: small cluster found")
        ax.plot(v.index, v.ari_same, marker="^", ms=4, color=c, ls=":", label=f"{name}: stability (ARI between draws)")
    logx(ax, [250, 500, 1000, 2000, 4000, 8000]); ax.set_xlabel("vertices m")
    ax.set_title("Stable is not the same as complete", loc="left")
    ax.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.08))
    fig.suptitle("Subsample size, n = 20 000, K = 100, tau = 0.5, seeds 500-509, 8 vertex draws each",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "A_12_subsample.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 13: dimension
df = read("13_dimension_final")
if df is not None:
    fam = lambda n: "blobs5" if n.startswith("blobs") else n.split("_")[0]
    df["family"] = df.dataset.map(fam)
    df["cfg"] = np.where(df.method == "full", "full", "sub" + df.m.astype(str))
    a = df[np.isclose(df.tau, 0.5)]
    a.groupby(["dataset", "d", "dim", "cfg"], sort=False)[["n_true", "n_est", "exact", "fn", "fp", "ari", "ari_ref",
                                                          "ari_same", "d_hat"]].mean().round(3) \
        .to_csv(SUMMARY / "A_13_at_floor.csv")
    # best threshold in hindsight (mean over seeds and draws), per configuration
    b = df.groupby(["dataset", "d", "dim", "cfg", "tau"], sort=False)[["ari_ref", "n_est", "ari_same"]].mean().reset_index()
    b.loc[b.groupby(["dataset", "dim", "cfg"]).ari_ref.idxmax()].round(3).to_csv(SUMMARY / "A_13_best_tau.csv", index=False)
    # stable thresholds (>= floor, >= 2 clusters, stability >= 0.8): finest one, per data seed
    rows = []
    for (ds, d, seed, dim, m), g in df[df.method == "subsample"].groupby(["dataset", "d", "seed", "dim", "m"], sort=False):
        tab = g.groupby("tau").agg(stab=("ari_same", "mean"), k=("n_est", lambda x: x.value_counts().index[0]),
                                   ari=("ari_ref", "mean"))
        ok = tab[(tab.index >= 0.5) & (tab.stab >= 0.8) & (tab.k >= 2)]
        rows.append(dict(dataset=ds, d=d, dim=dim, m=m, any_stable=float(len(ok) > 0),
                         tau=ok.index[0] if len(ok) else np.nan, k=ok.k.iloc[0] if len(ok) else 1,
                         ari_ref=ok.ari.iloc[0] if len(ok) else 0.0, n_true_ref=g.n_true_ref.iloc[0],
                         stab_at_floor=tab.loc[0.5, "stab"], ari_at_floor=tab.loc[0.5, "ari"]))
    pd.DataFrame(rows).groupby(["dataset", "d", "dim", "m"], sort=False).mean().round(3).to_csv(SUMMARY / "A_13_stable_threshold.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    x = a[a.dim == "auto"]
    for ax, (f, title) in zip(axes, (("gauss3", "3 clusters in 2-D + noise coordinates"),
                                     ("hier", "5 clusters in 2-D (hierarchy) + noise coordinates"),
                                     ("blobs5", "5 clusters filling all d coordinates"))):
        g = x[x.family == f]
        for cfg, c, lab in (("full", GREY, "all 20 000 points"), ("sub4000", BLUE, "4 000 vertices"),
                            ("sub1000", ORANGE, "1 000 vertices")):
            v = g[g.cfg == cfg].groupby("d").ari_ref.mean()
            ax.plot(v.index, v.values, marker="o", ms=4, color=c, label=f"ARI with the true partition, {lab}")
        v = g[g.cfg == "sub4000"].groupby("d").ari_same.mean()
        ax.plot(v.index, v.values, marker="^", ms=4, color=BLUE, ls=":", label="stability (ARI between draws), 4 000 vertices")
        logx(ax, sorted(g.d.unique())); ax.set_xlabel("dimension d"); ax.set_title(title, loc="left")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Dimension, n = 20 000, K = 100, threshold = noise floor (0.5), dim = auto, seeds 500-509",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "A_13_dimension.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 15: competitors
df = read("15_competitors_A_main")
if df is not None:
    if "error" in df:
        df[df.error.notna()].to_csv(SUMMARY / "A_15_errors.csv", index=False)
        df = df[df.error.isna()]
    df["cfg"] = df.method + "|" + df.param.fillna("")
    tuned = ["hdbscan", "leiden", "adp", "densitycut", "automato"]
    dflt = df[~df.method.isin(tuned) | (df.default == 1.0)]
    cols = ["k_correct", "ari", "nmi", "ari_nonnoise", "frac_noise", "n_est", "n_true"]
    g = dflt.groupby(["group", "method"])[cols]
    t = g.mean().round(3); t["runs"] = g.size(); t.sort_values("ari", ascending=False).to_csv(SUMMARY / "A_15_default.csv")
    best = df.groupby(["method", "dataset", "cfg"]).ari.mean().reset_index().sort_values("ari") \
        .groupby(["method", "dataset"]).tail(1)
    orc = df.merge(best[["method", "dataset", "cfg"]], on=["method", "dataset", "cfg"])
    orc.groupby(["group", "method"])[cols].mean().round(3).sort_values("ari", ascending=False).to_csv(SUMMARY / "A_15_oracle.csv")
    dflt.pivot_table(index="dataset", columns="method", values="ari", sort=False).round(3).to_csv(SUMMARY / "A_15_ari_by_dataset.csv")
    # gauss3_border was built in Phase 1 to be undecidable at tau = 0.5 (true
    # prominence 0.40-0.45): one data set of 18 that moves the mean by 0.05.
    # Both means are reported; the split was looked at after the results.
    pa = dflt.pivot_table(index="dataset", columns="method", values="ari", sort=False)
    pk = dflt.pivot_table(index="dataset", columns="method", values="k_correct", sort=False)
    nb = [x for x in pa.index if x != "gauss3_border"]
    pd.DataFrame({"ari_all": pa.mean(), "ari_without_border": pa.loc[nb].mean(), "ari_border": pa.loc["gauss3_border"],
                  "k_correct_all": pk.mean(), "k_correct_without_border": pk.loc[nb].mean()}).round(3) \
        .sort_values("ari_all", ascending=False).to_csv(SUMMARY / "A_15_with_without_border.csv")
    dflt.pivot_table(index="dataset", columns="method", values="k_correct", sort=False).round(3) \
        .to_csv(SUMMARY / "A_15_kcorrect_by_dataset.csv")
    order = [m for m in ("sub1000", "sub4000", "sub8000", "full", "tomato_w", "tomato_logdtm", "densitycut", "adp",
                         "hdbscan", "leiden") if m in set(dflt.method)]
    mat = dflt.pivot_table(index="dataset", columns="method", values="ari", sort=False)[order]
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(mat) + 1.8))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), order, rotation=30, ha="right"); ax.set_yticks(range(len(mat)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.6 else INK, fontsize=7.5)
    ax.grid(False); ax.set_title("ARI with the basins of the true density (default settings, 10 seeds)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout()
    fig.savefig(FIG / "A_15_competitors.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 16: cost
df = read("16_cost_main")
b2 = read("16_cost_brute2")          # exact matrix search after it was made to work in fixed-size blocks
if df is not None:
    df["version"] = "main"
    if b2 is not None:
        b2["version"] = "blocked search"
        b2.groupby(["data", "n", "d", "method"], sort=False)[["wall", "rss_peak_mb", "t_query", "t_label", "n_est"]] \
            .mean().round(3).join(df[df.method.str.startswith("sub:brute")]
                                  .groupby(["data", "n", "d", "method"])[["wall", "rss_peak_mb", "n_est"]].mean().round(3),
                                  rsuffix="_first_version").to_csv(SUMMARY / "A_16_brute_blocked.csv")
        df = pd.concat([df[~df.method.str.startswith("sub:brute")], b2], ignore_index=True)
    if "error" in df:
        df[df.error.notna()].to_csv(SUMMARY / "A_16_errors.csv", index=False)
        df = df[df.error.isna()]
    tc = [c for c in df.columns if c.startswith("t_")]
    extra = [c for c in ("recall", "max_abs_df", "mean_df", "same_peaks", "ari_vs_exact", "n_est", "n_est_exact") if c in df]
    for c in ("same_peaks",):
        if c in df:
            df[c] = df[c].astype(float)
    g = df.groupby(["data", "n", "d", "method", "version"], sort=False)[["wall", "rss_added_mb", "rss_peak_mb"] + tc + extra]
    t = g.mean().round(4); t["runs"] = g.size(); t.to_csv(SUMMARY / "A_16_cost.csv")
    h = df[df.data == "hier"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    cm = {"sub:kdtree:1000": ("subsample tree, 1 000 vertices", "#86b6ef"), "sub:kdtree:4000": ("4 000 vertices", BLUE),
          "sub:kdtree:8000": ("8 000 vertices", "#104281"), "full": ("same tree on all points", GREY),
          "tomato_gudhi": ("GUDHI ToMATo", ORANGE), "hdbscan": ("HDBSCAN", GREEN), "densitycut": ("densityCut (R)", PINK),
          "adp": ("DADApy ADP", YELLOW)}
    for ax, (col, lab) in zip(axes[:2], (("wall", "wall time [s]"), ("rss_peak_mb", "peak resident memory [MB]"))):
        for m, (name, c) in cm.items():
            v = h[h.method == m].groupby("n")[col].mean()
            if len(v):
                ax.plot(v.index, v.values, marker="o", ms=4, color=c, label=name)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_title(lab + ", 2-D data", loc="left")
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, fontsize=7.5, loc="lower left", ncol=4, bbox_to_anchor=(0.03, 0.0))
    ax = axes[2]
    st = [("t_index_full", "index on all data", GREY), ("t_query", "K neighbours of the vertices", BLUE),
          ("t_graph", "vertex graph", GREEN), ("t_tree", "merge tree", ORANGE), ("t_label", "label all rows", YELLOW)]
    sel = [("hier", 200000, "sub:kdtree:4000", "d = 2, n = 200 000 (k-d tree)"),
           ("hier_noise8", 200000, "sub:kdtree:4000", "d = 10, n = 200 000 (k-d tree)"),
           ("hier_noise48", 200000, "sub:brute:4000", "d = 50, n = 200 000 (matrix)"),
           ("pbmc68k", None, "sub:brute:4000", "cells, d = 50, n = 68 579 (matrix)"),
           ("ng20_minilm", None, "sub:brute:4000", "documents, d = 384, n = 18 221 (matrix)"),
           ("ng20_mpnet", None, "sub:brute:4000", "documents, d = 768, n = 18 221 (matrix)")]
    vals, labs = [], []
    for data, n, m, lab in sel:
        q = df[(df.data == data) & (df.method == m)]
        if n is not None:
            q = q[q.n == n]
        vals.append([q[c].mean() if c in q and len(q) else 0.0 for c, _, _ in st]); labs.append(lab)
    vals = np.nan_to_num(np.array(vals)); tot = vals.sum(axis=1); share = vals / np.maximum(tot, 1e-12)[:, None]
    y = np.arange(len(sel))[::-1]; left = np.zeros(len(sel))
    for j, (_, name, c) in enumerate(st):
        ax.barh(y, share[:, j], left=left, color=c, label=name, height=0.6, edgecolor=SURF, linewidth=1)
        left += share[:, j]
    for yi, t_ in zip(y, tot):
        ax.text(1.02, yi, f"{t_:.2f} s" if t_ < 10 else f"{t_:.0f} s", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labs, fontsize=8); ax.set_xlim(0, 1.17); ax.grid(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_title("time by stage, 4 000 vertices\n(share; total at right)", loc="left")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=3)
    fig.suptitle("Cost, single thread, K = 100", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97)); fig.savefig(FIG / "A_16_cost.png", dpi=150); plt.close(fig)

cl = read("14_cross_language_main")
if cl is not None:
    tc = [c for c in cl.columns if c.startswith("t_")]
    cl.groupby(["dataset", "n", "d", "m", "impl"], sort=False)[["total"] + tc + ["max_abs_df"]].mean().round(5) \
        .to_csv(SUMMARY / "A_14_cross_language_time.csv")
    cl.groupby(["dataset", "n", "m"], sort=False)[["max_abs_df", "same_edges", "same_peaks", "labels_identical",
                                                   "ari_py_r"]].first().to_csv(SUMMARY / "A_14_cross_language_agreement.csv")

# ------------------------------------------------------------------ 17: public tables
for tag in ("main2", "main"):
    df = read(f"17_tabular_{tag}")
    if df is not None:
        break
if df is not None:
    df = df[df.get("error", pd.Series(index=df.index, dtype=object)).isna()] if "error" in df else df
    keep = ["dataset", "n", "d", "n_classes", "K", "d_hat", "kmeans_ari", "k_floor", "ari_floor", "stab_at_floor",
            "rec_k", "rec_stability", "rec_ari", "ari_floor_vs_kmeans", "oracle_k", "oracle_ari", "hdbscan_k",
            "hdbscan_ari", "hdbscan_noise", "hdbscan2_ari", "t_fit", "t_stability", "t_hdbscan"]
    df[[c for c in keep if c in df]].round(3).to_csv(SUMMARY / "A_17_tabular.csv", index=False)
    m = ["kmeans_ari", "ari_floor", "rec_ari", "hdbscan_ari", "hdbscan2_ari"]
    pd.DataFrame({"all": df[m].fillna(0).mean(), "n>=2000": df[df.n >= 2000][m].fillna(0).mean(),
                  "n<2000": df[df.n < 2000][m].fillna(0).mean()}).round(3).to_csv(SUMMARY / "A_17_means.csv")
    o = df.sort_values("n").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(o) + 1.6))
    y = np.arange(len(o))[::-1]
    for col, c, lab, mk in (("kmeans_ari", GREY, "k-means, k = number of classes", "s"),
                            ("ari_floor", BLUE, "density tree, default threshold", "o"),
                            ("hdbscan_ari", GREEN, "HDBSCAN, package defaults", "^")):
        ax.scatter(o[col], y, color=c, marker=mk, s=38, label=lab, zorder=3, edgecolor=SURF, linewidth=1)
    ax.set_yticks(y, [f"{a}  (n={b}, d={c})" for a, b, c in zip(o.dataset, o.n, o.d)])
    ax.set_xlabel("ARI with the class label (an annotation, not the density structure)"); ax.set_xlim(-0.05, 1)
    ax.set_title("16 public tables, sorted by n", loc="left")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "A_17_tabular.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 19: sample size
df = read("19_small_n_main")
if df is not None:
    ok = df[df.get("error").isna()] if "error" in df else df
    cols = ["K", "tau0", "d_hat", "k_floor", "exact_floor", "ari_floor", "stab_floor", "rec80_k", "rec80_ari", "rec90_ari",
            "kmeans_ari", "hdbscan_k", "hdbscan_ari", "hdbscan_noise"]
    ok.groupby(["sep", "d", "n"])[[c for c in cols if c in ok]].mean().round(3).to_csv(SUMMARY / "A_19_small_n.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, d in zip(axes, sorted(ok.d.unique())):
        g = ok[ok.d == d]
        for sep, c in zip(sorted(g.sep.unique()), (ORANGE, BLUE, "#104281")):
            v = g[g.sep == sep].groupby("n")[["exact_floor", "hdbscan_ari", "kmeans_ari"]].mean()
            ax.plot(v.index, v.exact_floor, marker="o", ms=4, color=c, label=f"distance {sep:g}: density tree finds 4 clusters")
        v = g.groupby("n").kmeans_ari.mean()
        ax.plot(v.index, v.values, color=GREY, ls=":", marker="s", ms=3, label="k-means (k = 4 given): ARI, all distances")
        logx(ax, sorted(g.n.unique())); ax.set_xlabel("observations n"); ax.set_title(f"d = {d}", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Four equal Gaussian clusters: sample size needed by the density tree (default settings, 20 seeds)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "A_19_small_n.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 20: real data
df = read("20_real_data_main")
if df is not None:
    ok = df[df.error.isna()] if "error" in df else df
    g = ok.groupby(["data", "representation", "method", ok.param.where(~ok.method.eq("atdc_floor"), "5 draws")], sort=False)
    t = g[[c for c in ("k", "ari", "nmi", "frac_noise", "ari_nonnoise", "wall", "stability", "stability_at_floor", "d_hat") if c in ok]] \
        .mean().round(3)
    t["runs"] = g.size(); t.to_csv(SUMMARY / "A_20_real_data.csv")
# ------------------------------------------------------------------ 21: detectability
df = read("21_detectability_main")
if df is not None:
    df.sort_values("n").round(3).to_csv(SUMMARY / "A_21_detectability.csv", index=False)
    df.groupby("verdict").agg(tables=("dataset", "size"), kmeans_ari_vs_classes=("kmeans_ari_vs_classes", "mean"),
                              ari_tree_vs_classes=("ari_tree_vs_classes", "mean"),
                              detectability=("detectability", "mean")).round(3).to_csv(SUMMARY / "A_21_by_verdict.csv")

reps = {"as given": read("21_detectability_main"), "10 principal components": read("21_detectability_pca10"),
        "5 principal components": read("21_detectability_pca5")}
if all(v is not None for v in reps.values()):
    out = []
    for name, v in reps.items():
        c = v.verdict.value_counts()
        out.append({"representation": name, "tables": len(v), "consistent": int(c.get("consistent", 0)),
                    "inconclusive": int(c.get("inconclusive: resolution too low", 0)),
                    "not supported by density": int(c.get("not supported by density", 0)),
                    "kmeans_ari_vs_classes": v.kmeans_ari_vs_classes.mean(), "ari_tree_vs_classes": v.ari_tree_vs_classes.mean(),
                    "consistent_tables": " ".join(v[v.verdict == "consistent"].dataset),
                    "not_supported_tables": " ".join(v[v.verdict == "not supported by density"].dataset)})
    pd.DataFrame(out).round(3).to_csv(SUMMARY / "A_21_verdicts_by_representation.csv", index=False)

sy = read("21_detectability_synth")
if sy is not None:
    g = sy.groupby(["case", "truth", "n"], sort=False)
    t = g.agg(runs=("seed", "size"), k_tree=("k_tree", "mean"), detectability=("detectability", "mean"),
              sim_k_median=("sim_k_median", "mean"), sim_ari=("sim_ari_mean", "mean"))
    for v in ("consistent", "inconclusive: resolution too low", "not supported by density"):
        t[v] = g.verdict.apply(lambda x, v=v: int((x == v).sum()))
    t.round(3).to_csv(SUMMARY / "A_21_synthetic.csv")

# ------------------------------------------------------------------ 14: R with base-R matrix search
cm = read("14_cross_language_matrix")
if cm is not None:
    tc = [c for c in cm.columns if c.startswith("t_")]
    cm.groupby(["dataset", "n", "d", "m", "impl"], sort=False)[["total"] + tc + ["max_abs_df"]].mean().round(5) \
        .to_csv(SUMMARY / "A_14_matrix_time.csv")
    cm.groupby(["dataset", "n", "m"], sort=False)[["max_abs_df", "same_edges", "same_peaks", "labels_identical",
                                                   "ari_py_r"]].first().to_csv(SUMMARY / "A_14_matrix_agreement.csv")
print("report A written")
