# Prior work and where ATDC differs

Scope: the methods ATDC competes with directly, and the work closest to its
one candidate contribution (selective refinement). This is not a systematic
review: it rests on reading the primary sources listed below plus about fifteen
targeted web searches (November 2026).

Verification marks: **[V]** read in the paper, official documentation or
source code; **[A]** abstract or bibliographic record only; **[U]** standard
statement not re-read. No theoretical guarantee is claimed for ATDC anywhere in
this repository.

## 1. What already exists (and is therefore not new in ATDC)

| Ingredient used by ATDC | Where it already exists |
|---|---|
| Density peaks, prominence, hierarchical merging with a union-find | ToMATo |
| Log-density peak-minus-saddle as the merge statistic | 0-dim persistence of a log-density; ToMATo with a log density; DPA |
| Comparing peak-minus-saddle with its statistical error | DPA / ADP |
| Choosing the significant peaks from the data | AuToMATo (bottleneck bootstrap), DPA (Z), Kim et al. 2016 |
| Coarse then refined density, valley-to-peak ratio, stable interval | densityCut |
| Density of only a subsample evaluated against the full data | DBSCAN++ |
| Tree of kNN-density level sets | Robust Single Linkage, HDBSCAN |
| Error measure for cluster trees | merge distortion (Eldridge et al. 2015) |

`R_j = log p_peak,j − log p_saddle,j` is the persistence of peak *j* in the
superlevel filtration of the log-density. It is not a new statistic and is not
presented as one.

## 2. Direct competitors

### ToMATo
Chazal, Guibas, Oudot, Skraba. *Persistence-Based Clustering in Riemannian
Manifolds.* J. ACM 60(6):41, 2013, doi:10.1145/2535927 (SoCG 2011 version read) **[V]**.

Input: a neighbourhood graph, a density estimate at the points, a merge
parameter τ. Points are visited in decreasing density; each joins the cluster of
its highest neighbour; when a point connects two clusters, the one with the
lower root is merged into the other if its prominence (root density minus the
current density) is below τ. τ = ∞ yields the 0-dimensional persistence diagram,
from which τ is chosen by a gap. No statistical test; no coarse-to-fine step.

Implementation: `gudhi.clustering.tomato.Tomato(graph_type='knn',
density_type='logDTM', n_clusters=None, merge_threshold=None, **params)`
(GUDHI 3.13) **[V]**. Density types are `DTM`, `logDTM`, `KDE`, `logKDE`,
`manual`; the default is `logDTM`. Used here unmodified, and as a positive
control for the merge-tree code (`tests/test_mergetree.py`).

**Relation.** ATDC's Stage A *is* ToMATo on a subsample with a log-kNN density.
The reference called `full` in the experiments is ToMATo on all points.

### densityCut
Ding, Shah, Condon. *densityCut: an efficient and versatile topological approach
for automatic clustering of biological data.* Bioinformatics 32(17):2567–2576,
2016, doi:10.1093/bioinformatics/btw227 **[V]**.

kNN density with K = log2(N), then a "refinement" that is a random walk with
restart on the kNN graph (α = 0.9) — a **global** smoothing of every point's
density. Modes are points denser than their in-neighbours; valley height over
the lower peak height (a ratio in [0,1]) decides merging; the number of clusters
is the one stable over the longest range of the ratio threshold. No statistical
test.

Implementation: R only (`bitbucket.org/jerry00/densitycut_dev`, GitHub mirror
`SamGG/densitycut`); no Python implementation found. **Not run in this
repository** for that reason (no re-implementation was attempted).

**Relation.** densityCut's refinement changes *all* densities by diffusion.
ATDC's refinement recomputes *selected* densities from more data. The
peak/valley ratio and the stability interval are densityCut's and are not
claimed.

### DPA / ADP (PAk, DADApy)
d'Errico, Facco, Laio, Rodriguez. *Automatic topography of high-dimensional data
sets by non-parametric density peak clustering.* Information Sciences
560:476–492, 2021, doi:10.1016/j.ins.2021.01.010 **[V]**. PAk: Rodriguez,
d'Errico, Facco, Laio, JCTC 14:1206–1215, 2018 **[A]**. DADApy: Glielmo et al.,
Patterns 2022, doi:10.1016/j.patter.2022.100589 **[A]**.

PAk chooses, per point, the largest k for which a likelihood-ratio test cannot
reject a constant density (threshold 23.928, p = 1e-6) and returns log ρ with an
error ε. DPA merges cluster c into c′ when
`log ρ_c − log ρ_cc′ < Z (ε_c + ε_cc′)`, with ρ_cc′ the border (saddle) density.
Z is the only free parameter.

Implementation: `dadapy` 0.3.4, `Data.compute_density_PAk()`,
`compute_clustering_ADP(Z=1.65)` **[V]**. It requires numpy < 2, so it runs in
its own environment (`benchmarks/dadapy_worker.py`).

**Relation.** The significance test for a peak against its saddle is DPA's.
What DPA does not do is use the *undecided* outcome of that test to choose where
further computation is spent: densities and errors are computed once, for all
points. DPA's per-point adaptive bandwidth is a statistical refinement that ATDC
does **not** have (see `algorithm_design.md`, limits).

### AuToMATo
Huber, Kališnik, Schnider. *AuToMATo: An Out-Of-The-Box Persistence-Based
Clustering Algorithm.* TMLR 2025, arXiv:2408.06958 **[V]**.

ToMATo diagram on the data; B bootstrap resamples; the (1−α) quantile of the
bottleneck distance between bootstrap and original diagrams gives a global
threshold τ. Defaults α = 0.35, B = 1000. Implementation: `automato` on PyPI.

**Relation.** A global threshold at the cost of B full ToMATo runs — the
opposite of a budget-aware method. ATDC takes τ as an input and does not choose
it.

### HDBSCAN
Campello, Moulavi, Sander, PAKDD 2013 **[A]**; Campello et al., TKDD 2015 **[A]**;
McInnes & Healy, ICDMW 2017 **[A]**; `hdbscan` package documentation **[V]**.

Single linkage on the mutual reachability distance
`max(core_k(a), core_k(b), d(a,b))`; with λ = 1/distance the tree is condensed
by `min_cluster_size` and flat clusters maximise stability. Since the core
distance is a monotone transform of the kNN density, the hierarchy estimates the
tree of the level sets of a kNN density. No statistical test; no refinement.

### Robust Single Linkage
Chaudhuri & Dasgupta, NIPS 2010 **[A]**; Chaudhuri, Dasgupta, Kpotufe, von
Luxburg, IEEE Trans. IT 60:7900–7912, 2014, arXiv:1406.1546 **[V]**.

For each r: nodes with kNN radius ≤ r, edges of length ≤ αr; components are the
clusters. Consistent in Hartigan's sense for √2 ≤ α ≤ 2 and k ≳ d log n. The
2014 paper adds a pruning step that reconnects components that are joined at a
slightly lower level, removing spurious clusters. Global, not adaptive.

### Density cluster tree, merge distortion, inference
Hartigan 1981 **[A]**. Eldridge, Belkin, Wang, COLT 2015, arXiv:1506.06422 **[V]**:
merge height m(x,x′) = the highest level at which x and x′ are in the same
component of {f ≥ λ}; merge distortion = sup over pairs of |m − m̂|. Kim, Chen,
Balakrishnan, Rinaldo, Wasserman, NIPS 2016, arXiv:1605.06416 **[V]**: bootstrap
confidence set for the tree of a smoothed density and pruning of branches
shorter than twice the quantile.

The evaluation in this repository uses merge heights between modes, i.e. merge
distortion restricted to the modes, plus peak detection and merge order.

### 0-dimensional persistent homology
Edelsbrunner, Letscher, Zomorodian 2002 **[A]**; Edelsbrunner & Harer 2010
**[U]**; stability: Cohen-Steiner, Edelsbrunner, Harer 2007 **[A]/[U]**;
confidence sets: Fasy et al. 2014 **[A]**, Chazal et al. JMLR 2018 **[A]**.

The merge tree of a superlevel filtration computed with a union-find and the
elder rule is the standard algorithm; `src/atdc/mergetree.py` implements exactly
that and nothing more.

## 3. Work closest to selective refinement

| Work | What it does | Overlap | Difference |
|---|---|---|---|
| **DBSCAN++** (Jang & Jiang, ICML 2019, arXiv:1810.13105) **[V]** | core-point status for m ≪ n sampled points, each against the full data; the rest assigned to the nearest core point | "density of a subsample, computed on the full data, is enough" — this is exactly the **uniform refinement** baseline of this repository | one level, one shot, nothing selective |
| **AnyDBC** (Mai et al., KDD 2016, doi:10.1145/2939672.2939750) **[A]** | anytime DBSCAN that actively chooses which range queries to issue | budgeted, lazy, active choice of neighbourhood queries — closest in spirit | single level, resolves connectivity only, no peaks/saddles/significance |
| **Vidal, Guillou, Tierny** (TVCG 2021, doi:10.1109/TVCG.2021.3060500; LDAV 2021, arXiv:2108.05766) **[A]** | progressive, coarse-to-fine persistence diagrams of scalar fields with a bottleneck error bound | coarse-to-fine merge-tree computation — closest on the topology side | deterministic fields on grids; no sampling noise, no statistics |
| Active level-set estimation (Gotovos et al., IJCAI 2013) **[A]** | sample until every point's confidence interval clears the threshold | the same stopping logic | one level set of a black-box function |
| SNG-DBSCAN (Jiang, Jang, Łącki, NeurIPS 2020) **[A]** | subsample the edges of the ε-graph | subsampled connectivity | non-adaptive |
| Phillips, Wang, Zheng (SoCG 2015, arXiv:1307.7760) **[A]** | KDE coresets preserve the persistence diagram | justifies a coarse stage | uniform error, no adaptivity |

Not found after targeted searching: refinement directed at saddles or merge
events; multi-fidelity or bandit allocation for cluster-tree significance; a
subsampled ToMATo.

## 4. The claim that remains

Only this is a candidate contribution:

> Use the undecided significance of merge events in a cheap subsample tree to
> decide where full-data density queries are spent, and re-derive the tree
> lazily until every retained event is verified or the budget is exhausted.

It is a combination: ToMATo's tree, DPA's kind of test, DBSCAN++'s
subsample-against-full-data density, AnyDBC's lazy query selection. Whether the
combination adds anything over the simplest member of that list — DBSCAN++-style
uniform refinement of a smaller subsample — is an empirical question, answered
in `evaluation.md`.
