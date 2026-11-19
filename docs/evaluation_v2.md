# Evaluation of the second design (Phase 2)

Numbers come from `results/raw/07*`, `08*`, `09*`, `10*` through
`benchmarks/11_report_v2.py` (`results/summary/08_*`, `results/figures/08_*`).
Phase-1 results are unchanged.

Setting unless stated: K = 100, τ = 0.5, graph of 10 neighbours, single thread.
**Final seeds 300–309 were never used before**; design and parameters were fixed
on seeds 200–214 ([`refinement_v2.md`](refinement_v2.md) §4). The final runs were
the only process on the machine; wall time of the uniform and v2 families is the
median of three runs.

Abbreviations: **U(m)** uniform refinement on m vertices; **v2(m₀)** = U(m₀) +
ascent of near-threshold peaks + new vertices at near-threshold merge locations
(A+S); **+T** = sample splitting with held-out re-estimation of each candidate
event; **v1** = the Phase-1 algorithm (q = 0.1, optimistic saddles).

## 1. Audit of the evaluation (before the experiments)

| Item | Finding | Action |
|---|---|---|
| True tree discretisation | recomputed with 4× the reference points and 2× the neighbours: true peak heights change by ≤ 0.0001, merge heights by ≤ 0.056 (`hetero`), same modes at τ ∈ {0.3, 0.5, 1.0} for all 14 data sets; true labels agree at ARI 0.98–1.00 (`09_audit_truth`) | none; ARI above ≈ 0.98 is within the noise of the truth |
| Peak evaluation | Phase 1 counted a mode as found when a peak lay *anywhere* in its basin | added `top_deficit` (true log-density at the mode minus at the estimated peak), `loc_err`, and `f1_top` (a true positive must be within 0.25 of the top) — `metrics.evaluate_plus` |
| `circles` | the mode is a ring: its location is arbitrary, so `loc_err` is meaningless there | `top_deficit` is used instead |
| HDBSCAN | noise points count as errors in the all-point ARI; zero clusters possible | already reported both ways in Phase 1 (`03b`); here it is a reference only |
| Budget | Phase 1 matched "neighbour evaluations", which ignores index construction, graph work, Python loops and labelling | **wall time is the criterion**; total and without labelling are both reported; counts are secondary |
| Subsample size vs resolution | Phase 1 matched cost by shrinking q | the uniform family is swept over m as a frontier |
| Seeds | 0–9 and 100–104 had been seen | development 200–214, final 300–309 |
| Random streams | found by a test: the split was correlated with data generated from the same integer seed | separate seed sequences |

## 2. Main result: v2 against uniform refinement

n = 20 000, 12 data sets × 10 seeds = 120 runs per row (`gauss3_border` and
`many64` are reported apart, as announced before the run).

| Method | Exact | FP | FN | Peak F1 (strict) | Saddle MAE | Merge-order recall | ARI | Wall [s] | Wall w/o labelling [s] |
|---|---|---|---|---|---|---|---|---|---|
| U(200) | 0.858 | 0.10 | 0.07 | 0.788 | 1.27 | 0.43 | 0.914 | 0.024 | 0.012 |
| U(400) | 0.917 | 0.03 | 0.08 | 0.813 | 0.86 | 0.64 | 0.946 | 0.032 | 0.018 |
| U(1000) | 0.942 | 0.04 | 0.03 | 0.854 | 0.75 | 0.71 | 0.957 | 0.056 | 0.038 |
| U(2000) | 0.933 | 0.06 | 0.01 | 0.877 | 0.77 | 0.85 | 0.958 | 0.096 | 0.072 |
| U(4000) | 0.950 | 0.08 | 0.00 | 0.894 | 0.77 | 0.83 | 0.962 | 0.174 | 0.143 |
| **v2(200)** | 0.917 | 0.06 | 0.03 | 0.882 | 1.28 | 0.52 | 0.930 | 0.031 | 0.020 |
| **v2(400)** | 0.967 | 0.02 | 0.02 | 0.884 | 0.81 | 0.70 | 0.960 | 0.043 | 0.030 |
| v2(1000) | 0.933 | 0.07 | 0.01 | 0.881 | 0.75 | 0.78 | 0.955 | 0.080 | 0.060 |
| v2+T(400) | 0.850 | 0.06 | 0.09 | 0.827 | 0.91 | 0.55 | 0.934 | 0.039 | 0.025 |
| T only (1000) | 0.825 | 0.08 | 0.11 | 0.812 | 0.68 | 0.62 | 0.927 | 0.043 | 0.025 |
| v1 | 0.875 | 0.11 | 0.06 | 0.855 | 0.61 | 0.67 | 0.934 | 0.047 | 0.026 |
| full data | 0.950 | 0.07 | 0.00 | 0.901 | 0.84 | 0.83 | 0.971 | 0.659 | 0.659 |

Paired differences (same data set and seed; 95 % bootstrap interval resampling
seeds within data sets; `08_v2_final_main_paired.csv`):

| Comparison | Wall [s] | Δ exact | wins / ties / losses | Δ ARI | Δ strict peak F1 |
|---|---|---|---|---|---|
| **v2(200) − U(400)** *(primary: equal time)* | 0.031 vs 0.032 | **0.000** [−0.050, +0.050] | 5 / 110 / 5 | −0.016 [−0.043, +0.005] | **+0.068** [+0.044, +0.092] |
| v2(200) − U(1000) | 0.031 vs 0.056 | −0.025 [−0.075, +0.025] | 4 / 109 / 7 | −0.027 [−0.050, −0.007] | +0.027 [+0.010, +0.045] |
| v2(400) − U(1000) | 0.043 vs 0.056 | +0.025 [−0.017, +0.067] | 5 / 113 / 2 | +0.004 [−0.006, +0.014] | +0.029 [+0.004, +0.056] |
| v2(1000) − U(2000) | 0.080 vs 0.096 | 0.000 [−0.050, +0.042] | 4 / 112 / 4 | −0.003 [−0.016, +0.009] | +0.004 [−0.011, +0.019] |
| v2(1000) − U(4000) | 0.080 vs 0.174 | −0.017 [−0.050, +0.008] | 1 / 116 / 3 | −0.008 [−0.019, +0.003] | −0.013 [−0.025, −0.002] |
| v2+T(400) − v2(400) | 0.039 vs 0.043 | **−0.117** [−0.175, −0.058] | 1 / 104 / 15 | −0.027 [−0.039, −0.015] | −0.056 [−0.077, −0.036] |
| T(1000) − U(1000) | 0.043 vs 0.056 | −0.117 [−0.183, −0.050] | 3 / 100 / 17 | −0.030 [−0.047, −0.013] | −0.043 [−0.072, −0.013] |
| v1 − U(1000) | 0.047 vs 0.056 | −0.067 [−0.117, −0.017] | 2 / 108 / 10 | −0.023 [−0.036, −0.011] | +0.001 [−0.018, +0.020] |
| v2(200) − v1 | 0.031 vs 0.047 | +0.042 [−0.025, +0.108] | 12 / 101 / 7 | −0.004 [−0.030, +0.020] | +0.026 [+0.005, +0.049] |

Reading:

* **At equal wall time v2 does not recover the tree better than uniform
  refinement.** The primary comparison is an exact tie. The best cell, v2(400)
  at 0.967, is not distinguishable from U(1000) at 0.942 (interval contains 0;
  5 wins, 2 losses in 120 runs), and v2(1000) falls back to 0.933.
* **What v2 does improve is where the peak sits.** With 200–400 initial vertices
  the estimated peaks are as close to the true modes as U achieves with ~3 000
  vertices (strict F1 0.88 vs 0.85 at 1 000 and 0.89 at 4 000; mean deficit 0.108
  vs 0.133 and 0.107). This difference is significant and is the ascent at work.
* **It does not improve the saddles.** Saddle error follows the number of
  vertices, v2 or not (1.28 at m₀ = 200, 0.81 at 400, 0.75 at 1 000).
* **v2 is better than v1** in time (0.031 s vs 0.047 s) with exact recovery not
  worse (+0.04 [−0.03, +0.11]); v1 is significantly worse than U(1000).
* **The held-out test lowers accuracy on these data** (−0.12 exact,
  significant), see §4.

`results/figures/08_frontier_wall_main.png` shows the same thing as curves: one
frontier for exact recovery and ARI; v2 above it only in the strict-peak panel.

### Larger n

n = 200 000, 8 data sets × 5 seeds = 40 runs per row (`08_v2_final_n200k`).

| Method | Exact | Strict peak F1 | Saddle MAE | ARI | Wall [s] |
|---|---|---|---|---|---|
| U(400) | 0.900 | 0.855 | 0.56 | 0.939 | 0.172 |
| U(1000) | 0.925 | 0.902 | 0.35 | 0.944 | 0.212 |
| U(2000) | 1.000 | 0.953 | 0.36 | 0.972 | 0.263 |
| U(4000) | 0.975 | 0.943 | 0.23 | 0.965 | 0.331 |
| v2(200) | 0.900 | 0.925 | 1.32 | 0.902 | 0.173 |
| v2(400) | 0.875 | 0.940 | 0.52 | 0.932 | 0.190 |
| v2(1000) | 0.950 | 0.970 | 0.33 | 0.952 | 0.247 |
| v2+T(400) | 0.825 | 0.876 | 0.45 | 0.917 | 0.201 |
| v1 | 0.925 | 0.949 | 0.61 | 0.951 | 0.547 |
| full data | 0.925 | 0.963 | 0.68 | 0.960 | 5.865 |

v2(200) − U(400): Δ exact 0.000 [−0.10, +0.10] at equal time (0.173 vs 0.172 s);
v2(400) − U(1000): −0.050 [−0.15, +0.05]. With only 40 runs the intervals are
wide, but nothing points towards v2, and the best row is plain U(2000). At this
size 0.11 s of every method is labelling and 0.06 s the index: the whole uniform
family from 400 to 4 000 vertices spans 0.17–0.33 s.

### Reported apart

| | U(400) | U(1000) | U(2000) | U(4000) | v2(200) | v2(400) | v2(1000) | v2+T(400) | T(1000) | v1 | full |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gauss3_border`, exact (n = 20 000) | 0.6 | 0.2 | 0.2 | 0.0 | 0.6 | 0.5 | 0.2 | 0.6 | **0.8** | 0.3 | 0.0 |
| `many64`, peak F1 (n = 20 000) | 0.32 | 0.80 | 0.91 | 0.93 | 0.22 | 0.37 | 0.82 | 0.33 | 0.74 | 0.85 | 0.95 |
| `many64`, wall [s] | 0.020 | 0.031 | 0.049 | 0.085 | 0.027 | 0.044 | 0.087 | 0.039 | 0.027 | 0.034 | 0.330 |
| `many64`, peak F1 (n = 200 000) | 0.35 | 0.90 | 0.99 | 1.00 | 0.27 | 0.44 | 0.92 | 0.32 | 0.84 | 1.00 | 1.00 |

* `many64` (64 modes): the structure needs vertices, and adaptive insertion near
  the events it already sees cannot supply the ones it does not see. v2 is
  dominated by U at every time budget (v2(1000): F1 0.82 in 0.087 s; U(2000):
  0.91 in 0.049 s). v1, with its 2 000-vertex coarse graph, does better here
  than v2 and about as well as U at the same time.
* `gauss3_border` (true prominence 0.40–0.45 < τ): fewer vertices means fewer
  false peaks for every method; the held-out test is the only thing that gets it
  mostly right at 1 000 vertices (0.8 against 0.2).

## 3. Ablation: which ingredient does what

Development seeds, n = 20 000, 12 data sets (`07_v2_dev`, 60 runs per cell;
`07b_v2_dev2`, 120 runs per cell). Times under parallel load.

| Added to U(400) | Exact | Strict peak F1 | Saddle MAE | Extra density evaluations | Wall [s] |
|---|---|---|---|---|---|
| nothing | 0.90 | 0.81 | 0.63 | 0 | 0.035 |
| ascent (A) | 0.92 | 0.89 | 0.62 | ≈ 40 | 0.040 |
| saddle vertices (S) | 0.95 | 0.82 | 0.57 | ≈ 17 | 0.042 |
| hidden-peak checks (H) | 0.90 | 0.81 | 0.63 | ≈ 26 | 0.037 |
| edge verification (E) | 0.90 | 0.80 | 0.57 | ≈ 370 | 0.051 |
| A + S + E + H | 0.97 | 0.90 | 0.52 | ≈ 570 | 0.075 |
| U(400), round 2 | 0.89 | 0.80 | 0.74 | 0 | 0.034 |
| A + S, round 2 | 0.93 | 0.89 | 0.71 | ≈ 55 | 0.047 |
| A + S + T, round 2 | 0.87 | 0.83 | 0.85 | (half the data per query) | 0.044 |

* The peak-location gain is the ascent's; nothing else moves it.
* Edge verification and saddle vertices lower the saddle error by about 0.06
  each. Edge verification needs as many density evaluations as the graph has
  vertices, which is why U with twice the vertices is the cheaper way to the
  same saddle accuracy.
* The complete graph refinement (A+S+E+H) had the best exact recovery in round 1
  (0.97) at twice the time of U(400) and the same time as U(1000) (0.92) — a
  5-seed-per-data-set difference that was not pursued because the lean variant
  came close in round 1 (0.95 with the test) and the full one is several times
  slower than U on `many64`.

On `many64` at m₀ = 200, edge verification alone finds 30 modes where U finds 6:
it is the one ingredient that repairs a graph that is much too sparse. U(2000)
finds 53 in less time.

## 4. The held-out test

The mechanism works as designed: on a flat density the in-sample prominence of
the selected events is positive by construction, the held-out one is centred at
zero with the predicted standard deviation, and fewer than 3 % exceed τ
(`tests/test_v2.py`). On the benchmark it trades errors rather than removing
them:

| n = 20 000, final seeds | FP | FN | Exact |
|---|---|---|---|
| U(1000) | 0.04 | 0.03 | 0.942 |
| T(1000) | 0.08 | 0.11 | 0.825 |
| v2(400) | 0.02 | 0.02 | 0.967 |
| v2+T(400) | 0.06 | 0.09 | 0.850 |

Per data set (exact, v2+T(400) vs v2(400)): `gauss3_close` 0.7 vs 1.0, `hier`
0.8 vs 1.0, `hier_noise3` 0.4 vs 0.8, `circles` 0.5 vs 0.9; `gauss3_border`
0.6 vs 0.5.

Why: the in-sample prominence carries two opposite errors — upward selection
bias (the peak is a maximum and the saddle a minimum of noisy values) and
downward bias from smoothing and from a peak that is not at the top. For modes
whose true prominence is 0.7–1.1 against τ = 0.5 they roughly cancel. Removing
only the first leaves an estimate that is honest about the two *locations* and
too low for the *mode*, with a standard error of 0.2 at K₂ = 50; a true
prominence of 0.9 then fails about one time in ten. The test also halves the
data behind the tree (K₁ = 50), which adds candidates. Its one clear success is
the case built to have no true split.

So the uncertainty-aware rule is not wrong, but at a fixed bandwidth it has less
power than the biased rule it replaces. It would need a bias correction for the
smoothing, i.e. a statistical refinement of the estimator itself.

## 5. Time and memory

One process per run, mean of 3 seeds (`10_v2_scaling`).

| `hier`, d = 2 | n = 20 000 | n = 100 000 | n = 1 000 000 | memory above the data, n = 10⁶ |
|---|---|---|---|---|
| U(400) | 0.021 s | 0.076 s | 0.77 s | 4 MB |
| U(1000) | 0.033 s | 0.097 s | 0.85 s | 6 MB |
| U(4000) | 0.092 s | 0.158 s | 0.96 s | 16 MB |
| v2(200) | 0.023 s | 0.078 s | 0.71 s | 5 MB |
| v2(400) | 0.031 s | 0.089 s | 0.78 s | 5 MB |
| v2+T(400) | 0.030 s | 0.086 s | 0.75 s | 21 MB |
| v1 | 0.030 s | 0.176 s | 2.91 s | 47 MB |
| ToMATo (GUDHI) | 0.251 s | 1.33 s | 16.3 s | 3 061 MB |

At n = 10⁶ every subsample method spends 0.33 s building the index and
0.36–0.53 s labelling; the part that differs between U(400) and U(4000) is 0.06 s
of queries, and between U and v2 a few milliseconds. **There is no time left for
an adaptive strategy to save.** v1's cost grows with n because its coarse graph
does (q fixed); the others keep m fixed.

Where a density evaluation is expensive, the picture changes in degree, not in
kind. `gauss3_noise8` (d = 10, n = 100 000): U(1000) 0.82 s, U(4000) 2.82 s,
v2(200) 0.29 s, v2(400) 0.48 s, GUDHI 40 s. Here v2(200) is 2.8× faster than
U(1000) at the same exact recovery (1.0 vs 1.0 on the final seeds at n = 20 000,
0.9–1.0 for U). This is the only regime where the selective part bought time,
and it is one data set.

Not measured: approximate neighbours, several threads, real data, K ≠ 100 on the
final seeds.

## 6. Answers

1. **Why v1 lost.** Overhead, not accuracy of the idea: it saved 36 % of the
   density evaluations and spent more than that on a coarse stage, a Python
   loop and labelling against five times more vertices; a density evaluation
   costs ~10 µs. Second, its estimator on frozen coarse basins was worse than a
   consistent field on the same vertices (0.93 vs 0.98 exact).
2. **What was designed.** Start from uniform refinement on a few hundred
   vertices with one fidelity; grow the graph where the tree is decided
   (ascent of near-threshold peaks, vertices at merge locations, lazy
   verification of long edges with batched cuts, a rule for peaks hidden by
   long edges); optionally re-estimate every candidate event on a held-out half
   of the data.
3. **What improved.** Against v1: 35 % less time at n = 20 000 and 4× less at
   n = 10⁶ with exact recovery not worse. Against U: peak location only.
4. **Equal budget.** No. Exact recovery at equal wall time is a tie
   (0.917 vs 0.917; Δ = 0.000 [−0.05, +0.05] at n = 20 000; 0.90 vs 0.90 at
   n = 200 000). No configuration of v2 is significantly better than a uniform
   one that takes the same or less time; on `many64` v2 is significantly worse.
5. **Where the gain is.** In the peak estimate (location and height: strict F1
   +0.07 [+0.04, +0.09] at equal time). Not in the density estimate, which is
   the same estimator; not in the saddles, which follow the vertex count; not
   in the merge order.
6. **Data.** Better or equal: Gaussian mixtures, `hetero`, `hier`, the
   noise-dimension sets (`hier_noise3`: v2(400) 0.8 vs U(400) 0.3, U(1000) 0.7).
   Worse: `many64` at every budget; `moons` slightly (0.8–0.9 vs 0.9–1.0).
   Unsolved for everyone: `circles`, `gauss3_border`.
7. **Worth the overhead?** At d ≤ 5, no: the same milliseconds spent on more
   uniform vertices give the same tree and better saddles. At d = 10 the ascent
   pays (2.8× less time than the uniform run of equal accuracy), on one data
   set.
8. **Further design changes?** Not along the computational axis. Two rounds of
   design now agree: the tree needs a few hundred to a few thousand
   consistently estimated vertices, those are nearly free, and wall time is
   dominated by the index and the labelling. What still limits accuracy is
   statistical — bandwidth, smoothing bias, noise maxima along ridges — and the
   held-out test shows that removing one bias without the other makes things
   worse.

## 7. Judgement: **PIVOT**

Selective / adaptive refinement has a real but narrow benefit (peak location;
time in high dimension) and no benefit for the quantity it was meant to improve,
the recovery of the hierarchy per unit of time. The method that both phases
kept failing to beat is three lines long:

> take ~1 000–2 000 random points, give each its kNN density in the full data,
> run ToMATo on their kNN graph, label everything else by nearest vertex.

At n = 10⁶ it takes 0.85 s and 6 MB against 16.3 s and 3 GB for GUDHI's ToMATo
on all points, with exact recovery 0.94 vs 0.95 at n = 20 000 and 1.00 vs 0.93
at n = 200 000. That is a useful fast estimator of a density cluster tree. It is
not new (DBSCAN++ density on a subsample, ToMATo's tree), and the honest
framing for any follow-up is an implementation and a characterisation, not an
algorithm.

Recommended direction:

* Stop developing selective refinement as the contribution (v1 and v2).
* If the project continues, make the uniform subsample tree the object: how many
  vertices are needed as a function of the number and size of modes
  (`many64` shows the dependence), stratified vertex selection, and exploration
  tooling around the tree (prominence diagram, τ sweep at no extra cost since
  the tree is computed once).
* Keep from v2 only the ascent, as an optional polish of the reported peak
  locations.
* A statistical line (bias-corrected or adaptive-bandwidth estimates at the
  events, with the held-out design for honesty) remains scientifically open; it
  is a different project from "selective computation" and should be judged
  against DADApy's PAk, which already does adaptive bandwidth everywhere.

GO is not supported (no advantage at equal time). REVISE is not chosen because
two independent designs failed for the same structural reason, which further
redesign of the refinement does not address. STOP would discard the one result
that is practically useful.

## 8. Not done

* K other than 100 and τ other than 0.5 on the final seeds (Phase-1 sensitivity
  showed both dominate all methods equally).
* The full graph refinement (A+S+E+H) on final seeds.
* Real data; approximate nearest neighbours; multi-threaded timing.
* densityCut (R only); AuToMATo and DADApy were not rerun in Phase 2.
* Stratified vertex selection and multi-resolution sampling (not built).
