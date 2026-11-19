# Algorithm design: selective refinement for density cluster trees

Status: research prototype. Everything below is implemented in `src/atdc/`
unless marked *not implemented*. No consistency or error guarantee is claimed.

## 1. Objects (kept separate on purpose)

Let p be the density, f = log p.

| Task | Definition used here |
|---|---|
| **Peak detection** | local maxima of an estimate f̂ on a neighbourhood graph |
| **Saddle estimation** | merge height m(a,b) = max over a–b paths of the minimum of f̂ along the path |
| **Hierarchy recovery** | the merge tree: peaks as leaves, joined at their merge heights (elder rule: the lower peak dies) |
| **Cluster selection** | keep peaks with log-prominence R_j = f̂(peak_j) − f̂(saddle_j) ≥ τ |
| **Labelling** | each point follows its highest neighbour to a retained peak (ToMATo's rule); non-subsample points take the label of their nearest subsample point |

R_j is the 0-dimensional persistence of the superlevel filtration of f̂ — an
existing quantity (see `literature_review.md`). τ is an input; ATDC does not
choose it.

The discrete definition (`src/atdc/mergetree.py`): a graph with vertex values
f(v) and edge weights w(e) ≤ min of the endpoint values (equality by default).
The superlevel graph at t keeps vertices with f ≥ t and edges with w ≥ t;
m(u,v) is the largest t at which u and v are connected. Computed by Kruskal's
algorithm with a union-find. Verified against hand-computed cases, an
independent brute-force implementation of the definition, and GUDHI's ToMATo
(`tests/test_mergetree.py`).

## 2. Two kinds of accuracy

For a kNN estimate with k neighbours among N points,
`f̂(x) = ψ(k) − log N − log V_d − d log r_k(x)`:

* **variance** ≈ ψ′(k) ≈ 1/k (checked in `tests/test_core.py`);
* **smoothing bias** depends on the ball radius r_k, i.e. on k/N: peaks are
  flattened, valleys filled. Raising k lowers the variance and *raises* this
  bias. So "more neighbours" is not "more accurate".

This gives two separate axes:

| Axis | Knob | What it changes |
|---|---|---|
| **statistical** | bandwidth K (neighbours in the full data) | bias–variance trade-off of the target estimator |
| **computational** | how much of the data is used at a fixed radius | variance only; the bias is that of K |

A subsample of rate q with k₀ = qK neighbours has, in expectation, the same
ball radius as the full data with K neighbours: same bias, variance larger by
1/q. ATDC refines **only along the computational axis**. Its target is the tree
the full-data estimator with bandwidth K would give; it cannot be statistically
better than that estimator, and the experiments are read accordingly.

## 3. Candidate designs considered

| # | Idea | Uncertainty | Verdict |
|---|---|---|---|
| A | Grow k locally around uncertain events, all points in the graph from the start | 1/k per point | Rejected as the main design: changing k changes the bias, so refined and unrefined values are not comparable, and the coarse stage (all n points) is already expensive. |
| **B** | **Subsample tree, then full-data density at the peaks and saddles of undecided events** | 1/k₀ vs 1/K at equal bias | **Chosen.** Refinement changes variance only; the coarse stage costs q² of the full kNN work. |
| C | Lazy verification of edges by the density at their midpoint (the saddle is a bottleneck-path problem; cheap endpoint minima are upper bounds) | edge may jump a valley | Kept as an option (`midpoint=True`); with kNN edges no longer than the bandwidth it has little to detect. |
| D | Local bootstrap of the events (AuToMATo restricted to candidates) | resampling | Rejected: many tree recomputations per event; cost not competitive. |
| E | Nested subsample ladder q₀ < q₁ < … < 1 with successive rejection | 1/k_ℓ | *Not implemented.* Would only pay if most candidates could be rejected at an intermediate level. |
| F | Per-point adaptive bandwidth at refined points (PAk-style test) | bias onset | *Not implemented.* It is the one variant that could improve *statistical* accuracy. |

A further obstacle shaped the design. Putting refined values back into the
field and recomputing the tree does not work: an unrefined neighbour, whose
coarse value is the maximum of noisy values, overtakes a refined peak, becomes a
new "peak", and refinement spreads over the whole cluster. Design B therefore
does not mix fidelities in one field. It keeps the **coarse basins** as a fixed
partition and re-estimates two kinds of numbers on the graph of basins.

## 4. The chosen design

### Stage A — coarse tree
1. Random subsample S, |S| = m = qn. kNN among S with k₀ = max(4, round(qK)).
2. Coarse values f₀ on S; kNN graph (k_graph neighbours, symmetrised).
3. Ascent basins: every vertex points to its highest neighbour. Basin graph:
   node value h_a = peak value of basin a; edge value s_ab = best crossing edge
   between adjacent basins a and b. Its merge tree equals the vertex-level merge
   tree (proved by a path argument, tested in `test_basin_graph_tree_equals_vertex_tree`).

The coarse stage is deliberately over-segmented (m/(k₀+1)-ish basins): a true
mode almost always owns at least one coarse basin, which is how missed peaks are
avoided. What it cannot see is structure narrower than the bandwidth — neither
can the full-data estimator.

### Stage B — selective refinement
*Fine value* of a subsample point: kNN log-density in the full data with K
neighbours. Verifying

* a **peak** a: fine values at the t_top highest coarse vertices of basin a;
  h_a := the largest fine value seen in the basin;
* a **saddle** (a,b): fine values at the endpoints of the t_top best coarse
  crossing edges; s_ab := max over those edges of the min of the two fine values
  (optionally also the fine value at the midpoint).

```
compute the tree of the basin graph from the current (h, s)
repeat
    for each merge event j (peak a dies at adjacency e, prominence R_j):
        u_j   := number of unverified items among {a, e}
        cand  := R_j + z·se₀·sqrt(u_j) ≥ τ  and  u_j > 0
    also needed: the surviving peak of every verified event with R_j ≥ τ,
                 and every root (a peak that never dies)
    optional (z_saddle > 0): for every retained peak, the adjacency where it
                 would die if unverified saddles were s + z_saddle·se₀
    stop if nothing is needed, or the budget is exhausted
    verify the needed items, most prominent first (one batched kNN query)
    recompute the tree
```

Answers to the six design questions:

1. **Uncertainty.** An item (peak or saddle) is either coarse (standard error
   se₀ = √ψ′(k₀)) or verified (√ψ′(K)). An event is uncertain when its
   prominence could reach τ given the coarse items it depends on. Coarse maxima
   are also biased upward (maximum of noisy values); this is why a peak is never
   *retained* without verification.
2. **Selection.** Events whose upper bound reaches τ, ordered by that bound.
   With z = 0 this is "every event currently above τ".
3. **What improves.** The variance of the peak and saddle heights of those
   events, and the upward selection bias of coarse maxima. Not the smoothing
   bias.
4. **Update.** The tree of the basin graph is recomputed from scratch (it is
   small). A verified peak usually drops; another coarse basin of the same
   cluster then becomes the elder and is verified next. The same happens along a
   boundary: the critical adjacency drops and the next one is verified. The loop
   therefore peels the top of each retained cluster and the best part of each
   retained boundary.
5. **Stopping.** (i) fixpoint: every retained event has its peak, its saddle and
   its survivor verified; (ii) budget on the number of fine evaluations;
   (iii) iteration cap.
6. **Missed peaks / connections.** Peaks: over-segmentation of Stage A (above)
   plus z > 0, which also verifies events slightly below τ. Connections: the
   optimistic saddle rule (z_saddle > 0) verifies adjacencies whose coarse value
   may be an underestimate; the midpoint option caps edges that cross a valley.
   An adjacency absent from the subsample graph is never created — a real limit.

### Controls implemented alongside
* `coarse` — Stage A only.
* `global` — fine values at **all** subsample vertices, vertex-level tree
  (uniform refinement; DBSCAN++-style density, ToMATo tree).
* `rag_all` — every peak and adjacency verified (ATDC's estimator without
  selection).
* `random` — random peaks/adjacencies verified, same number of fine evaluations.
* `global_matched` — `global` on a smaller subsample with the same total cost.
* `coarse_matched` — `coarse` on a larger subsample with the same total cost.
* `full` — no subsampling: ToMATo with a log-kNN density.

## 5. Cost

Unit: one neighbour returned by a kNN query (Σ k over queries).

| Step | Units |
|---|---|
| Stage A kNN | m·max(k₀, k_graph) |
| Stage B | r·K, r = number of refined vertices |
| labelling all points | n |
| uniform refinement | m·K |
| full data | n·K |

Tree work is O(m k_graph) for the basin graph plus O(B log B) per iteration for
B basins. The fine queries need an index over the full data (O(n log n) build),
which the coarse stage alone does not.

The units ignore index construction and Python overhead; wall time is reported
next to them and the two do not always agree (`evaluation.md`).

## 6. Known limits of the design

* Target is the full-data estimator at bandwidth K, not the true tree.
* τ and K are inputs.
* The basin partition is frozen at the coarse stage; refined values are only
  compared through per-basin maxima and per-adjacency maxima.
* Standard errors ignore the selection bias of maxima and the dependence between
  neighbouring estimates. The rule is a heuristic, not a test with a level.
* Subsample labels are propagated by 1-NN; cluster boundaries have the
  resolution of the subsample.
