# Evaluation

All numbers come from `results/raw/*.csv` through `benchmarks/06_report.py`
(`results/summary/`, `results/figures/`). Setting unless stated: n = 20 000,
bandwidth K = 100, subsample rate q = 0.1 (k₀ = 10), graph of 10 neighbours,
threshold τ = 0.5 (log-prominence), single thread, seeds 0–9.
Stage B parameters (z = 0, z_saddle = 2, t_top = 5) were fixed on seeds 100–104.

Ground truth is the merge tree of the **true density** pruned at τ; labels are
its basins, not the generating components.

**Reading the tables.** "Exact" = fraction of runs whose retained peaks are
exactly the true modes (one per true basin, none extra). Averages exclude
`gauss3_border`, a data set built to be undecidable (true prominence 0.40–0.45,
just below τ); it is listed separately. Cost = neighbours returned by kNN
queries. Wall times of the accuracy runs were taken while other benchmark
processes were running, so compare them within a table only; the scaling runs
(§5) were timed with at most one other process.

## 1. Correctness

| Check | Result |
|---|---|
| Merge tree vs hand-computed graphs (1 peak, 2 peaks, hierarchical 3 peaks, elder rule, ties, plateaus, −∞ and 1e−300 values, disconnected, capped edges) | pass |
| Merge heights and persistence pairs vs an independent brute-force implementation, 1 200 random graphs (with/without ties, with/without edge caps) | identical |
| Tree of the basin graph vs vertex-level tree, 400 random graphs | identical |
| Persistence diagram and labels vs GUDHI ToMATo on the same graph and values | identical diagram, ARI > 0.999 |
| kNN log-density: mean and variance under a uniform density | unbiased, variance = ψ′(k) within 10 % |
| True tree vs analytic saddle of a two-Gaussian mixture | within 0.03 |
| Refinement when fine ≡ coarse (q = 1) | tree unchanged, all three modes |

34 tests, `tests/`. One bug was found this way: cluster labels below a retained
saddle did not follow ToMATo's rule; fixed and covered by a test.

## 2. Is selective refinement effective? (main question)

12 data sets × 10 seeds (`01_ablation_main`).

| Method | Exact | FP | FN | ARI | Saddle MAE | Triplet recall | Cost | Wall [s] |
|---|---|---|---|---|---|---|---|---|
| coarse (Stage A only) | 0.12 | 10.87 | 0.02 | 0.405 | 0.62 | 0.69 | 40 000 | 0.046 |
| random refinement, same budget | 0.07 | 8.29 | 0.02 | 0.439 | 0.62 | 0.73 | 67 447 | 0.048 |
| coarse, larger subsample, same cost | 0.15 | 6.58 | 0.02 | 0.482 | 0.83 | 0.73 | 67 088 | 0.063 |
| **ATDC selective** | **0.89** | 0.10 | 0.04 | **0.938** | **0.62** | 0.71 | 67 196 | 0.052 |
| **uniform refinement, smaller subsample, same cost** | **0.89** | 0.05 | 0.08 | **0.941** | 0.80 | 0.67 | 67 196 | **0.031** |
| ATDC without selection (everything verified) | 0.93 | 0.06 | 0.02 | 0.952 | 0.62 | 0.74 | 123 660 | 0.058 |
| uniform refinement, same subsample | 0.98 | 0.03 | 0.00 | 0.969 | 0.62 | 0.78 | 240 000 | 0.105 |
| full data (ToMATo-type) | 0.94 | 0.08 | 0.00 | 0.971 | 0.86 | 0.88 | 2 000 000 | 0.727 |

What this shows:

1. **Selection is not arbitrary.** At the same number of fine evaluations,
   random verification leaves the tree as bad as the coarse one (exact 0.07);
   selective verification reaches 0.89.
2. **Against uniform refinement of the same subsample**: 28 % of the cost for a
   lower exact rate (0.89 vs 0.98) and the same saddle error.
3. **Against uniform refinement at equal cost — no advantage.** Evaluating the
   full-data density at every point of a subsample of rate ≈ 0.02 gives the same
   exact rate and ARI (paired over 120 runs: ATDC better in 5, worse in 5) and
   is faster in wall time, because it is one batched query with no loop. ATDC is
   better only on saddle heights (0.62 vs 0.80).
4. The whole cost–accuracy curve says the same
   (`results/figures/05_tradeoff_cost_main.png`, `…_wall_main.png`): the uniform
   family and ATDC lie on one frontier for F1 and ARI, and the uniform family
   extends to lower cost.

| Uniform refinement, rate q | 0.005 | 0.01 | 0.02 | 0.04 | 0.1 | 0.4 |
|---|---|---|---|---|---|---|
| cost | 31 000 | 42 000 | 64 000 | 108 000 | 240 000 | 1 140 000 |
| exact | 0.78 | 0.87 | 0.93 | 0.94 | 0.98 | 0.93 |
| ARI | 0.78 | 0.91 | 0.95 | 0.96 | 0.97 | 0.97 |

| ATDC, rate q | 0.04 | 0.07 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|
| cost | 53 415 | 62 748 | 67 196 | 123 068 | 355 358 |
| exact | 0.85 | 0.90 | 0.89 | 0.93 | 0.97 |
| ARI | 0.92 | 0.94 | 0.94 | 0.95 | 0.97 |

Other sample sizes:

| Setting | ATDC exact / ARI | uniform, same cost | uniform, same subsample | full |
|---|---|---|---|---|
| n = 5 000, 6 data sets × 3 seeds | 0.89 / 0.947 | 0.72 / 0.902 | 0.94 / 0.960 | 0.78 / 0.929 |
| n = 20 000, 12 × 10 | 0.89 / 0.938 | 0.89 / 0.941 | 0.98 / 0.969 | 0.94 / 0.971 |
| n = 200 000, 7 × 5 | 0.89 / 0.944 | 0.97 / 0.975 | 0.97 / 0.979 | 0.80 / 0.923 |

ATDC is ahead of the cost-matched uniform control only at n = 5 000, where that
control is left with about 100 graph vertices — and where everything runs in
milliseconds anyway. At n = 200 000 it is behind.

Many clusters (`many64`, 64 modes; added afterwards as a case that should need
graph resolution): at n = 100 000, ATDC at q = 0.07 (cost 372 100) has F1 0.998
and ARI 0.931; uniform refinement at q = 0.02 (320 000) has 0.987 / 0.905 and at
q = 0.04 (540 000) 1.000 / 0.939 — the same frontier again. At n = 20 000 no
method recovers the 64 modes at K = 100 (full data finds 58): the bandwidth is
too wide for clusters of ~300 points, which refinement along the computational
axis cannot repair. There ATDC (F1 0.86 at q = 0.1) is *worse* than the coarse
tree it starts from (0.90) and than uniform refinement (0.91): verified peaks
lose their upward bias and fall below τ.

**Development-seed optimism.** On the seeds used to choose z_saddle, ATDC scored
exact 0.98 against 0.94 for the cost-matched control; on the reporting seeds it
is 0.89 against 0.89. The apparent advantage did not replicate.

### Parameter sensitivity (development seeds, `02_sensitivity_dev`)

| Change from base | Exact | ARI | Cost |
|---|---|---|---|
| base (z = 0, z_saddle = 0, t_top = 5) | 0.86 | 0.941 | 64 564 |
| z = 1 (verify events up to 1 s.e. below τ) | 0.84 | 0.934 | 94 426 |
| z_saddle = 1 / **2** / 3 | 0.90 / **0.98** / 0.94 | 0.953 / 0.969 / 0.964 | 67 534 / 70 180 / 71 252 |
| t_top = 1 / 10 / 20 | 0.84 / 0.88 / 0.88 | 0.916 / 0.942 / 0.940 | 46 762 / 82 402 / 109 134 |
| edge-midpoint check | 0.86 | 0.940 | 72 942 |
| K = 30 / 50 / 200 | 0.16 / 0.52 / 0.98 | 0.470 / 0.640 / 0.977 | — |
| τ = 0.3 / 0.75 / 1.0 | 0.60 / 0.96 / 0.90 | 0.696 / 0.968 / 0.914 | — |

The statistical parameters K and τ move the result far more than any Stage B
parameter, for every method alike (at K = 30 full data scores 0.00 exact and
uniform refinement 0.57): a threshold of 0.5 is inside the noise of a
30-neighbour estimate. Stage B does not change that.

## 3. Statistical versus computational accuracy

Every method that reaches full-data fidelity carries the same error against the
*true* tree: peak log-density MAE ≈ 0.35 and saddle MAE ≈ 0.6–0.9 for full data,
uniform refinement and ATDC alike (deep saddles are smoothed: in `hier` the
lowest saddle is estimated at −6.3 against a true −7.35 by all of them,
`results/figures/06_trees_hier_main.png`). Refinement removed the *computational*
error of the coarse tree (11 spurious peaks → 0.1) and nothing else. The
hypothesis' second half — improved *statistical* accuracy — was not tested by
this design and is not supported by any result here.

Two side observations:

* More graph vertices at the same fidelity produce more spurious peaks, because
  more noisy values compete (full data, n vertices: exact 0.94; uniform on
  2 000 vertices: 0.98; at n = 200 000: 0.80 vs 0.97). Higher resolution is not
  monotonically better at a fixed τ.
* The saddle error is *lower* on a moderately sized subsample graph than on the
  full graph (0.62 vs 0.86), mostly from `circles` and `hetero`, whose deep
  saddles lie in sparse regions.

## 4. Existing methods

n = 20 000, 12 data sets × 10 seeds (`03_competitors_main`, `03b_hdbscan_main`).

| Method | Exact | FP | FN | ARI (all points) | ARI (labelled points) | Unlabelled | Wall [s] |
|---|---|---|---|---|---|---|---|
| ATDC | 0.89 | 0.10 | 0.04 | 0.938 | — | 0 | 0.048 |
| uniform refinement, same cost | 0.89 | 0.05 | 0.08 | 0.941 | — | 0 | 0.029 |
| full data, log-kNN (ToMATo-type) | 0.94 | 0.08 | 0.00 | 0.971 | — | 0 | 0.668 |
| ToMATo, GUDHI, log-DTM, same τ | 0.88 | 0.15 | 0.00 | 0.957 | — | 0 | 0.572 |
| HDBSCAN, mcs = n/100, min_samples = 10 | 0.55 | 1.11 | 0.53 | 0.448 | 0.632 | 29 % | 0.763 |
| HDBSCAN, same + allow_single_cluster | 0.58 | 0.00 | 1.06 | 0.356 | 0.624 | 27 % | 0.764 |
| HDBSCAN, best of 18 settings by labelled-point ARI (oracle) | 0.83 | 0.01 | 0.44 | 0.435 | 0.906 | 23 % | 0.878 |
| DADApy ADP, PAk, Z = 1.65 (default) | 0.48 | 1.43 | 0.07 | 0.699 | — | 0 | 3.49 |
| DADApy ADP, PAk, Z = 3 | 0.77 | 0.01 | 0.46 | 0.814 | — | 0 | 3.48 |
| AuToMATo, defaults (n = 5 000, 6 data sets × 3 seeds) | 0.39 | 0.11 | 1.11 | 0.652 | — | 0 | 21.0 |

Caveats that matter for fairness:

* The ground truth is "modes of the true density with log-prominence ≥ 0.5".
  ToMATo-type methods are given that τ; HDBSCAN, ADP and AuToMATo select
  clusters by their own criteria, so part of their deficit is a different
  definition of a cluster, not an error. HDBSCAN leaves 23–36 % of the points
  unlabelled by design; on the points it labels its oracle setting reaches ARI
  0.91.
* HDBSCAN recovers the two-level structure of `hier` in none of the settings
  (it returns the two super-clusters) and needs `allow_single_cluster` for
  unimodal data, which in turn breaks `gauss3_close` and the noisy-dimension
  sets. On `circles` it is the best method (exact 1.0 vs 0.3–0.7 for all
  prominence-based ones).
* DADApy at Z = 3 is the only method that decides `gauss3_border` correctly
  (1.0) and is good in 2-D, but fails with 8 noise dimensions (exact 0.0).
* AuToMATo was run only at n = 5 000 because of its 1 000 bootstrap ToMATo runs.
* densityCut was not run (R only).

Where the subsample-based methods are better: time and memory (§5), with
accuracy close to full-data ToMATo. Where they are worse: `circles`, and the
5-dimensional `hier_noise3` (ATDC exact 0.6, cost-matched uniform 0.2, against
1.0 for full data).

## 5. Time and memory

`hier`, d = 2, one process per run, mean of 3 seeds (`04_scaling`).

| n | coarse | ATDC | uniform (same subsample) | full | ToMATo GUDHI | HDBSCAN |
|---|---|---|---|---|---|---|
| 10 000 | 0.013 s | 0.016 s | 0.025 s | 0.16 s | 0.13 s | 0.15 s |
| 100 000 | 0.16 s | 0.19 s | 0.31 s | 2.5 s | 1.4 s | 3.1 s |
| 1 000 000 | 2.3 s | 3.1 s | 3.9 s | 30.7 s | 17.9 s | 139 s |
| peak memory above the data, n = 10⁶ | 122 MB | 47 MB | 131 MB | 2 709 MB | 3 061 MB | 347 MB |

At n = 10⁶ ATDC spends 0.78 s labelling all points, 0.53 s on the coarse graph
and tree, 0.92 s recomputing the tree over 47 iterations, 0.60 s on fine queries
and 0.34 s building the full-data index; uniform refinement spends 1.76 s on
fine queries instead. So ATDC is 5.8× faster than GUDHI's ToMATo and uses 65×
less memory, but only 1.3× faster than uniform refinement of the same subsample
— and the cost-matched uniform control, which is faster still at n = 20 000 and
200 000, was not part of the scaling run. The Python loop of Stage B grows with
n (6 → 47 iterations) and would be the first thing to fix.

With noise dimensions at n = 100 000: d = 5, ATDC 0.41 s, uniform 0.84 s, GUDHI
4.9 s; d = 10, ATDC 2.8 s, uniform 6.9 s, GUDHI 40.9 s. KD-trees degrade with
dimension for every method; the ratio between them stays similar.

Not measured: approximate nearest neighbours, multi-threading, real data.

## 6. Failures and their causes

| Data | What fails | Cause |
|---|---|---|
| `circles` | 1 spurious peak on the outer ring in 70 % of ATDC runs (also 50 % of full-data runs) | density is flat along a ring, so noise bumps along the ridge have prominence close to τ = 0.5; ATDC adds to it by estimating a saddle from 5 crossing edges |
| `moons` | spurious peak in 20 % of runs | same mechanism; was 60 % before the optimistic-saddle rule |
| `hier_noise3` (5-D) | a true mode is missed in half of the runs | its coarse prominence falls below τ, so the event is never verified (z = 0); raising z did not help on development seeds and costs 50 % more |
| `gauss3_border` | every method except ADP (Z = 3) reports 2–3 clusters | true prominence 0.40–0.45 vs τ = 0.5: undecidable at this sample size and bandwidth |
| `many64`, n = 20 000 | ATDC below its own coarse stage | K too wide for small clusters; verification lowers peaks |
| any, K ≤ 50 | many spurious peaks for all methods | τ inside the noise of the estimator |
| deep saddles | error 1–5 in log-density for all methods | smoothing bias and sparse graphs; not addressed by refinement |

Design-level causes on ATDC's side: the basin partition is frozen at the coarse
stage; peaks and saddles are estimated from 5 candidates chosen with noisy
values; the uncertainty rule ignores the selection bias of maxima; unverified
values, biased upward, compete with verified ones.

## 7. Judgement

1. **Novelty.** No prior work was found that directs full-data density queries
   to undecided merge events. The components are all known (ToMATo's tree, DPA's
   test, DBSCAN++'s subsample density, AnyDBC's lazy queries); the claim is the
   combination only.
2. **Correctness.** The merge-tree machinery is correct against three
   independent references. With refinement, the true peak set is recovered
   exactly in 89 % of runs (full data 94 %); merge order between modes is
   recovered less well by all subsample methods (triplet recall 0.67–0.78 vs
   0.88 for full data); peak and saddle heights carry the bandwidth's bias.
3. **Selective refinement.** Better than random (0.89 vs 0.07 exact). About as
   good as uniform refinement of the same subsample at 28 % of its cost. **Not
   better than uniform refinement at equal cost**, and slower in wall time; the
   only gain is a lower saddle error.
4. **HDBSCAN and others.** Subsample-based ToMATo-type estimation is 6–45×
   faster than GUDHI ToMATo / HDBSCAN at n = 10⁶ with far less memory and
   comparable accuracy under a prominence-based ground truth. That advantage
   belongs to the subsample-plus-full-data-density step, not to selection.
   HDBSCAN is better on rings; ADP is better on the borderline case.
5. **Limits.** Synthetic data only, d ≤ 10, exact KD-trees, one τ and K per
   run supplied by the user, no real data, no approximate neighbours.
6. **Decision: REVISE** — and not GO.
   * The current Stage B should **not** be packaged or developed further as it
     stands: the cheaper and simpler uniform alternative matches it.
   * The idea is not dismissed, because the place where selection could pay was
     not tested: the *statistical* axis. Fine density queries turned out to be
     cheap; what is expensive is statistics — per-point adaptive bandwidth
     (PAk), bootstrap significance (AuToMATo costs 21 s at n = 5 000). Running
     those only at the handful of undecided events is the natural redesign, and
     the failures above (`circles`, `gauss3_border`, `many64`, K sensitivity) are
     all statistical ones that such a step would target.
   * A stop criterion for that next step: if selectively applied adaptive
     bandwidth or bootstrap does not beat (a) the uniform subsample pipeline
     with a fixed τ and (b) DADApy at Z = 3 on those four cases at lower cost
     than applying it everywhere, the line should be **stopped**.
   * PIVOT is not proposed: nothing here shows value as an exploration or
     visualisation tool.

## 8. Not done

* densityCut (R only); AuToMATo beyond n = 5 000; real data.
* Nested multi-level refinement and adaptive bandwidth (designs E, F).
* The cost-matched uniform control in the scaling run; scaling of the
  `many64` case beyond n = 100 000.
* Point-level merge distortion over all pairs (only mode-level merge heights
  were evaluated).
* Approximate nearest neighbours; multi-threaded timing.
