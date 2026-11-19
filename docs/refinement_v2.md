# Selective refinement, second design (v2)

Phase 2 asks one question: does adaptive, selective computation have an
intrinsic advantage over uniform refinement of a subsample for recovering a
density cluster tree? This document records why v1 lost, which designs were
considered, what was built, and what was dropped. Results are in
[`evaluation_v2.md`](evaluation_v2.md).

"Uniform refinement" (**U**) throughout: m random points become graph vertices,
each gets the kNN log-density computed in the full data (K neighbours), and the
merge tree of their kNN graph is pruned at τ. It is the Phase-1 control that v1
could not beat.

## 1. Why v1 did not beat uniform refinement

Re-analysis of the Phase-1 runs (`results/raw/01_*`, `05_*`; n = 20 000,
12 data sets × 10 seeds unless stated).

| # | Suspected cause | Evidence | Verdict |
|---|---|---|---|
| 1 | Variance / bias of the density | Every method at full-data fidelity has the same error against the truth (peak MAE ≈ 0.35, saddle MAE 0.6–0.9). | Not a difference between v1 and U. Statistical, set by K. |
| 2 | Peaks missed by the coarse sample | v1 FN 0.04 per run, U at equal cost 0.08. | Minor, and not in U's favour. |
| 3 | Saddles / connectivity of the kNN graph | U at small m: saddle bias +0.34 to +0.81 (m = 100–200), peak bias −0.26; both lower prominence → FN 0.28 at m = 100, 0 from m = 800. | The real computational error of **U at small m** — and therefore the place to look for an advantage. |
| 4 | Frozen basins | v1 with *everything* verified scores 0.93 exact; U on the same 2 000 vertices 0.98. | Costs ~5 points: the top-5 estimator on fixed basins is worse than a consistent field. |
| 5 | Selection bias of peak / saddle candidates | Coarse maxima are biased upward by +0.53 (peak height); unverified values compete with verified ones. | Forces the lazy loop to peel many items; source of 4 more points (0.89 vs 0.93). |
| 6 | Uncertainty rule | z (candidate margin) did not help; optimistic saddles did. On reporting seeds the gain shrank. | Heuristic, weakly informative. |
| 7 | Overhead | n = 20 000: v1 0.052 s = labelling 0.023 + coarse stage 0.012 + fine queries 0.011 + loop 0.004 + index 0.005. U at equal cost 0.031 s = labelling 0.012 + queries 0.014 + index 0.005. n = 200 000: v1 0.53 s, of which loop 0.10 and coarse stage 0.14; U 0.33 s. | **Main cause.** v1 saved 36 % of the fine evaluations (276 vs 430) and paid more than that for the coarse stage, the loop, and labelling against 5× more vertices. |
| 8 | Dependence on τ and K | K = 30: every method fails (exact ≤ 0.5); K = 200: every method ≥ 0.94. | Dominates everything, equally for all methods. |

Three conclusions shaped v2.

* **A fine density evaluation is too cheap to be worth saving.** One query
  costs ~10 µs in 2-D; 1 000 of them cost less than labelling the data. Any
  per-event logic in Python loses to evaluating a few hundred more vertices.
  Only in high dimension (0.17 ms per query at d = 10) or for genuinely
  expensive statistics is there something to save.
* **Wall time has a floor that no refinement strategy touches**: building the
  full-data index and labelling all points (12 ms of 31 ms at n = 20 000;
  ~0.2 s of 0.33 s at n = 200 000).
* **The number of vertices a problem needs does not grow with n.** The fair
  uniform baseline at large n keeps m fixed, so its cost approaches the floor.

So the remaining room for an adaptive method is narrow: reach, with few vertices,
the accuracy that U reaches only with many — by repairing the specific errors of
a small graph (row 3) — or exceed U's plateau by better *statistics* at the
events that decide the tree.

## 2. Candidate designs

| | Idea | Targets | Extra computation | Closest prior work | Outcome |
|---|---|---|---|---|---|
| **A** | Hill-climb in the full data from near-threshold peaks; visited points become vertices | peak deficit of a sparse graph (row 3) | a few density queries per peak | mean shift; modal EM (Li, Ray, Lindsay 2007) | **kept** |
| **S** | New vertices at the location where a near-threshold event attains its merge level | pass not sampled → saddle too low | 4 queries per event and round | saddle search (Comaniciu et al. 2002); NEB paths | **kept** |
| **E** | Lazy verification of long edges on the path between two peaks: density at interior points, batched over the cut of each suspicious merge, bisection along very long edges | edges that jump a valley → saddle too high, clusters merged | hundreds of queries when the graph is sparse | LazySP (Dellin & Srinivasa 2016); interior-point edge density in SigMA (Ratzenböck et al. 2023) | correct, **dropped by default**: costs more than the vertices it replaces |
| **H** | Verify the long ascent edge of every vertex that is a local maximum among its short edges | peaks with zero prominence, invisible to any event-based rule | one query per such vertex and round | — | correct, **dropped by default**, same reason |
| **T** | Sample splitting: tree, ascent and saddles from half H1 of the data; prominence of each candidate event re-estimated at the selected peak and saddle locations from the disjoint half H2 | selection bias of maxima/minima (row 5); calibrated uncertainty (row 6) | 2 queries per event; two indices instead of one | data splitting for mode significance (Genovese et al. 2016, on Hessians); same-sample kNN saddle tests (Burman & Polonik 2009; DPA; SigMA) | **kept as an option**: trades false positives for false negatives |
| B | Nested subsamples m₀ < m₁ < … | missed peaks | re-estimation at each level | progressive persistence (Vidal & Tierny) | not built: U with larger m does the same in one batch |
| — | Stratified (index-guided) vertex selection | rare clusters | none | density-biased sampling (Palmer & Faloutsos 2000) | not built: improves the *baseline*, not selectivity |
| — | Bootstrap of events | uncertainty | B tree recomputations | AuToMATo | not built: T gives an unbiased estimate for 2 queries |

All of v2 starts from U on a small m₀ and has **one fidelity**: every vertex
carries a full-data density. That removes v1's coarse stage, its mixed
fidelities and its frozen basins in one step; the graph grows, edges are never
removed, and the tree is recomputed from the whole graph.

## 3. The algorithm (`src/atdc/v2.py`)

```
V  ← m₀ random points;  f(v) ← kNN log-density in the data;  G ← kNN graph of V
[H] repeat ≤ 4×: verify the long ascent edge of hidden-peak candidates
T  ← merge tree of (G, f, edge caps)
[A] peaks with prominence ≥ τ − margin: hill-climb ≤ 3 steps, 6 candidates per step
    → new vertices; rebuild G
repeat ≤ 4×:
    events with prominence ≥ τ − margin
    [E] long unverified edges on their forest paths: cap by interior densities
    [S] 4 new vertices at each merge location; rebuild G
    stop when nothing changed;  T ← merge tree
prune:  keep peaks with prominence ≥ τ
        [T] or: keep events whose held-out prominence on H2 is ≥ τ
label:  ToMATo rule on V, then 1-NN for all points
```

Details that matter:

* **Long edge**: longer than the bandwidth radius of its endpoints. A shorter
  edge cannot hide a dip that the estimator could resolve at this bandwidth.
* **Edge caps**: a verified edge keeps `w = min(f_u, f_v, min interior density)`.
  When no long unverified edge remains on the paths of the selected events,
  their merge levels are exact for the verified weights (lazy Kruskal).
* **Why the path and not only the bottleneck**: a slope vertex can be attached
  to the wrong cluster by a long edge and to its own by a short one; the event's
  critical edge is then a perfectly good short edge. A test case exposed this.
* **Why hidden peaks need their own rule**: a cluster top whose highest
  neighbour lies across a valley has prominence 0 and no event refers to it.
* **Ascent by evaluation, not by mean-shift direction**: the mean-shift vector of
  one K-ball is as long as the sampling error of its centroid. Candidates are
  evaluated and the best is taken.
* **Held-out test.** H1 (fraction ρ) and H2 are disjoint; K₁ = ρK and
  K₂ = K − K₁ give the same expected ball radius, hence the same smoothing bias.
  Everything that *selects* uses H1. For an event with peak location p and
  merge location s, `R₂ = f̂₂(p) − f̂₂(s)` is independent of the selection, with
  standard error √(ψ′(K₂)·2) under local uniformity. The rule is `R₂ ≥ τ`.
  It is an unbiased estimate of the smoothed density difference **between
  those two locations** — not of the true prominence: if the graph has no vertex
  at the true pass, s is too low and the test confirms a real difference between
  two points that is not a real saddle (seen at d = 10; S repairs it).
  It is a heuristic decision rule, not a test with a stated level.

Mechanics are verified against the true density (`tests/test_v2.py`): a missed
peak is recovered by ascent; a false connection through long edges is removed;
a merge order that the sparse graph gets wrong becomes the true one; the merge
level of a two-Gaussian mixture is within 0.06 of the analytic saddle; unimodal
and flat densities keep one mode; on a flat density the in-sample prominence of
selected events is positive while the held-out one is centred at zero with the
predicted spread.

Two bugs were found by these tests: (i) the random split was correlated with a
data set generated from the same integer seed (one half contained a single
cluster) — the generators now use separate seed sequences; (ii) verifying only
the cut around the dying component misses false edges inside the surviving one.

## 4. What the development rounds decided

Seeds 200–204 (`07_v2_dev`) and 205–214 (`07b_v2_dev2`), n = 20 000, 12 data
sets; none of these seeds is used for the reported comparison.

| Variant (m₀) | Exact | FP | FN | Wall [s] | Vertices |
|---|---|---|---|---|---|
| U (400) | 0.89 | 0.09 | 0.05 | 0.034 | 400 |
| U (1000) | 0.91 | 0.08 | 0.03 | 0.061 | 1 000 |
| U (4000) | 0.93 | 0.09 | 0.01 | 0.193 | 4 000 |
| **A+S (200)** | 0.93 | 0.03 | 0.04 | 0.033 | 252 |
| A+S (400) | 0.93 | 0.08 | 0.02 | 0.047 | 455 |
| A+S+T (400) | 0.87 | 0.07 | 0.08 | 0.044 | 470 |
| T only (1000) | 0.88 | 0.09 | 0.08 | 0.047 | 1 000 |
| A+S+E+T (400) | 0.88 | 0.04 | 0.09 | 0.068 | 489 |

(round 2, 120 runs per row; times taken under parallel load)

* **E and H**: 350–750 extra density evaluations at m₀ = 200–400 — more than the
  graph itself — for no gain in exact recovery in round 1. They help only when
  the graph is far too sparse for the structure (`many64`: 30 modes found
  instead of 6 at m₀ = 200), and there simply using 2 000 vertices is both better
  (53 modes) and faster.
* **T**: on `gauss3_border` it turns 0.0 into 0.6–0.9 exact; on the other 12 it
  lowers exact recovery, because it removes the upward selection bias that, in
  the in-sample rule, happens to compensate the downward bias from smoothing and
  from peaks that are not at the top (`hier_noise3`: 0.3 instead of 0.7–0.9).
* **A+S** at m₀ = 200 looked as good as U with 20× more vertices.

Fixed before the final run: **v2 = A+S, primary m₀ = 200**, compared with
U(400) (same wall time on development seeds) and U(1000) (twice the time);
A+S+T and T are reported as ablations.

## 5. Not built, and why

* Nested multi-resolution sampling — U with a larger m is one batched query.
* Discovery of modes with no vertex nearby — impossible without sampling there;
  stratified vertex selection would help U and v2 equally.
* Adaptive bandwidth at the tested events — the one remaining statistical
  refinement; see the judgement in `evaluation_v2.md`.
