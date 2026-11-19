# ATDC common specification — draft 0.1 (Phase A)

Status: **draft for review**. This is the single reference that the R package
and the Python package are to implement. Where the two languages would
otherwise have to guess, the text fixes the answer. Items marked **[A]** are
defaults chosen from the Phase A experiments
([`phaseA_report.md`](phaseA_report.md)); they can still change before 1.0.

Nothing here is a new method: the tree is the 0-dimensional superlevel
persistence of a kNN log-density on a neighbourhood graph (ToMATo; Chazal et
al. 2013), the density of a subsample is computed against all data
(DBSCAN++; Jang & Jiang 2019), prominence on a log-density is a density ratio
(DPA; densityCut), the stability diagnostics are resampling stability
(Ben-Hur et al. 2002; Hennig 2007).

## 1. Notation

`X`: n × d data matrix. `V`: the m vertices (a subset of the rows of `X`).
`K`: neighbours for the density. `k_graph`: neighbours for the graph.
`f(v)`: estimated log-density of vertex v. Indices in this document are
1-based; an implementation uses its language's convention and converts at the
boundary (exported files are **1-based** in both languages).

## 2. Input

* `X` is a real matrix with n ≥ 2 rows (observations) and d ≥ 1 columns.
  Data frames with only numeric columns are accepted and converted. A vector
  is an error (the caller must say whether it is one column or one row).
* Missing, NaN or infinite values: **error**. No imputation.
* Constant columns are allowed (they do not change distances).
* The distance is Euclidean, on the columns as given. Standardisation is the
  caller's decision; neither package scales silently. The summary prints the
  column standard deviations' ratio (max/min) and warns above 100.
* Duplicate rows are allowed (see 4.3).
* n < 2 K + 2 after the default rule for K (3.2): error with the message that
  the data are too few for a density tree.

## 3. Parameters and defaults

| Name (spec) | Meaning | Default |
|---|---|---|
| `K` | neighbours for the density estimate | `min(100, max(10, floor(n / 20)))` **[A]** |
| `m` | number of vertices | `min(n, 4000)` **[A]** |
| `k_graph` | neighbours for the vertex graph | 10 |
| `dim` | dimension in the density formula: `"auto"`, `"ambient"` or a positive number | `"auto"` **[A]** |
| `threshold` | prominence threshold used for the default partition: a number, or `"auto"` | `"auto"` **[A]** |
| `vertices` | explicit vertex indices (overrides `m` and the seed) | none |
| `seed` | seed of the language's own generator | none (non-deterministic) |

The R and Python argument names are given in the API documents; this table is
the meaning they must have.

## 4. Density

### 4.1 Neighbours

For a vertex v, `r_j(v)` is the Euclidean distance from v to its j-th nearest
**other** row of `X` (the row of v itself is excluded; a different row with
the same coordinates is not). Neighbours are searched in **all** n rows, not
only among the vertices. The search is exact by default. An approximate search
is an explicit option and is recorded in the result.

### 4.2 Log-density

```
f(v) = digamma(K) − log(n − 1) − log V_D − D · log r_K(v)
V_D  = π^(D/2) / Γ(D/2 + 1)
```

`D` is the value of `dim`: the number of columns (`"ambient"`), a number given
by the caller, or (`"auto"`) the intrinsic-dimension estimate

```
D̂ = 1 / mean_v [ (1 / (K − 1)) · Σ_{j=1}^{K−1} log( r_K(v) / r_j(v) ) ]
```

(Levina & Bickel 2005 with the averaging of MacKay & Ghahramani 2005; terms
with r_j = 0 are skipped; vertices whose sum is 0 or not finite are left out of
the mean). `D̂` uses only distances that the density step computes anyway.

Facts the documentation must state:

* For any D > 0, f is a decreasing function of `r_K`. The tree as a
  combinatorial object — which vertices are peaks, which peak merges into
  which, in what order, and every partition obtained by keeping the j most
  prominent peaks — is therefore **the same for every D**. D only sets the
  unit of heights and prominences.
* Under local uniformity the standard error of f(v) is
  `(D / D₀) · sqrt(trigamma(K))`, D₀ being the true local dimension. With
  D = D₀ it is `sqrt(trigamma(K))` (0.100 for K = 100) in any dimension. With
  the ambient dimension of data that lie near a lower-dimensional set it is
  larger by the factor d / D₀, and so is every prominence. This is why
  `"auto"` is the default.
* f is a biased estimate of the log-density (it averages over a ball of
  radius r_K). The bias depends on K; trees for different K are different
  estimates, not refinements of one another.

### 4.3 Degenerate radii

If `r_K(v) = 0` (more than K rows coincide with v), f(v) would be +∞. The
packages replace such radii by the smallest positive `r_K` among the vertices
and **warn** once ("… rows have more than K exact duplicates; their density is
capped"). If every radius is 0: error.

## 5. Vertices

`m ≥ n` or `m = NULL/None` with n ≤ default: every row is a vertex, in row
order. Otherwise m distinct rows are drawn uniformly without replacement by
the language's generator and **sorted increasingly**. The two languages do not
share a random number generator; identical results across languages are
obtained by passing the same `vertices`. Every fitted object stores the vertex
indices, so any fit can be reproduced exactly in the other language.

Resolution statement to be printed by `summary`: a cluster can only be found
if more than `k_graph` vertices fall into it; with m vertices this needs a
share of the data of about `2 (k_graph + 1) / m` for a detection probability
above 0.95 **[A]**. Smaller groups are absorbed by a neighbour, silently and
reproducibly (see `phaseA_report.md`, section 1).

## 6. Graph

Undirected graph on the vertices: {u, v} is an edge if v is among the
`min(k_graph, m − 1)` nearest **other vertices** of u, or u among those of v.
Edge weight: `w(u, v) = min(f(u), f(v))`.

Equal distances (duplicates, lattices) make the neighbour set ambiguous. The
specification does not fix the choice; results are guaranteed identical across
languages only for data without such ties or when the graph is supplied. Both
packages expose a low-level entry point that takes `f` and the edge list
(section 12).

## 7. Merge tree

### 7.1 Order

* Vertices are ranked by decreasing f; ties by increasing vertex index. "Higher"
  below always means earlier in this order (a strict total order).
* Edges are processed by decreasing weight; ties by the rank of the lower
  endpoint (higher first), then by the rank of the higher endpoint (higher
  first). This is a strict total order of the edges.

### 7.2 Algorithm (reference; any implementation with the same output is valid)

Union–find over vertices, each component remembering its highest vertex (its
**peak**). For each edge in order: if the endpoints are in different
components A and B with peaks a higher than b, record the **merge event**
`(dying = b, surviving = a, level = w, edge = (u, v))` and unite them (peak a).
Vertices that are never `dying` are the **roots**, one per connected component
of the graph.

### 7.3 Peaks, saddles, prominence

* A **peak** is a vertex with no higher neighbour (a local maximum of f on the
  graph). Every root is a peak.
* The **saddle level** of a non-root peak p is the level of the event in which
  p dies; its **saddle edge** is that event's edge; its **parent** is the
  event's surviving peak. Roots have saddle level −∞ and no parent.
* **Prominence** `prom(p) = f(p) − saddle_level(p)`; +∞ for roots.
  Because f is a log-density, prom(p) is the logarithm of the ratio between
  the density at the peak and at the pass that connects it to higher ground.
  This is the "relative" prominence; the two packages report it as
  `prominence` and its exponential as `density_ratio`. A prominence on the
  density scale (f_peak − f_saddle without the logarithm) is **not** offered:
  it depends on the unit of every column and on the dimension.
* A vertex that is not a peak dies at its own level (prominence 0); these
  events are not part of the exported tree.

Caveat to be stated wherever saddles are shown: the saddle level is the best
pass **on the graph of the m vertices** for the estimated density. It is not
the saddle of the continuous density: the graph can lack a vertex at the pass
(level too low) and a long edge can jump a valley (level too high), and the
estimate itself is smoothed over the K-neighbour ball.

### 7.4 Basins

`basin(v)`: follow from v the edge to its highest higher neighbour until a
peak is reached. Every vertex belongs to exactly one basin.

## 8. Partition at a threshold

For a threshold τ > 0, replay the merge events of non-root peaks in decreasing
level (ties: the order of 7.1), with a second union–find over **peaks**,
initially all separate. For an event with saddle edge (u, v): let A, B be the
current clusters of basin(u) and basin(v), a higher than b their highest
peaks. If A ≠ B and `f(b) − level < τ`, B is merged into A. (This is ToMATo's
rule.) After the replay:

* the clusters are in one-to-one correspondence with the peaks of prominence
  ≥ τ together with the roots;
* a vertex has the cluster of its basin; a data row has the cluster of its
  nearest vertex (Euclidean; ties by smaller vertex index);
* cluster labels are integers 1 … k numbered by **decreasing peak height**
  (ties by vertex index). The same numbering in both languages. No row is
  left unlabelled; there is no noise label.

τ = +∞ gives the connected components of the graph. Partitions for
increasing τ are nested (coarser as τ grows).

Requesting a number of clusters k: the partition at the largest τ that yields
at least k clusters, i.e. τ = the k₀-th largest finite prominence where k₀ =
k − (number of roots); error if k is smaller than the number of roots, warning
if k exceeds the number of peaks or if prominences tie at the cut.

## 9. Prominence path

The table with one row per peak, sorted by decreasing prominence (roots first,
by decreasing height):

`rank, peak (vertex), row (data index), height, saddle_level, prominence,
density_ratio, parent (rank), n_clusters_if_cut_here, size_vertices, size_rows`

`n_clusters_if_cut_here` is the number of clusters for τ equal to this row's
prominence. The **prominence path plot** shows the number of clusters against
τ as a step function (τ on a logarithmic axis), the chosen threshold, and the
noise floor of 10.1. `size_rows` is the number of data rows whose nearest
vertex lies in the peak's basin.

## 10. Thresholds, stability, recommendation

### 10.1 Noise floor **[A]**

`τ₀ = 3.5 · sqrt(2 · trigamma(K))` (0.50 for K = 100): 3.5 standard errors of
a difference of two independent kNN log-density estimates, valid when `dim`
is the true local dimension. It is a rule of thumb, not a test: the peak and
the saddle are a maximum and a minimum over many vertices, the two estimates
share neighbours, and many peaks are examined. `threshold = "auto"` means τ₀.

### 10.2 Stability to resampling

`B` replicates (default 20). Replicate b draws a subsample of
`ceil(fraction · n)` rows without replacement (default fraction 0.8) and, in
it, vertices as in section 5; fits the tree with the same K, k_graph and D
(D fixed to the value of the original fit); labels **all** n rows by nearest
vertex. For a threshold τ:

* `stability(τ)` = mean adjusted Rand index over the pairs (b, b + 1 mod B);
* `count_mode(τ)`, `count_agreement(τ)` = most frequent number of clusters and
  the fraction of replicates that have it.

The **stability path** is this table on a grid of τ (default: 25 values,
geometric from τ₀ / 2 to the largest finite prominence of the original fit)
and its plot. It needs B additional fits and is never computed implicitly.

What it does and does not show: a partition that changes when rows or
vertices are redrawn should not be reported as structure. The converse fails:
a group too small for the chosen m (section 5) or for K is missed in every
replicate, and the partition without it is perfectly stable.

### 10.3 Stable thresholds **[A]**

`threshold = "auto"` is always the noise floor τ₀; resampling never changes
the reported partition silently. After a stability path has been computed:

* `summary` and the report print the stability at the reported threshold and
  **warn below 0.8**;
* `recommend` lists the grid values τ ≥ τ₀ whose partition has at least two
  clusters and `stability ≥ 0.8`, finest first, each with its number of
  clusters and stability. If the list is empty the message is "no partition
  into several clusters is stable at this resolution". A single cluster is
  trivially stable and is never listed.

The value 0.8 comes from the Phase A development runs (correct partitions had
stabilities of 0.83–0.89, wrong ones 0.4–0.65) and is in the range used for
related indices (prediction strength 0.8–0.9, Tibshirani & Walther 2005;
clusterwise Jaccard 0.75–0.85, Hennig 2007). It is a convention, not a test.
A first version of the rule (stability ≥ 0.9 and the same number of clusters
in 80 % of the replicates, the finest such threshold replacing the default
partition) was rejected: on 16 public tables it replaced good partitions by
coarser ones or by a single cluster (`phaseA_report.md`, section 5). The
documentation must not call any listed number of clusters "the" number of
clusters.

## 11. Comparison with an existing partition

Input: a vector of n labels from any method (k-means, a published
annotation). Output, for a threshold or on the τ grid:

* adjusted Rand index (Hubert & Arabie 1985) and normalised mutual
  information (arithmetic normalisation);
* the contingency table;
* per reference cluster: the density cluster holding most of its rows and
  that share; per density cluster: the same in the other direction;
* **split** (a reference cluster spread over several density clusters with at
  least 10 % of its rows in a second one) and **merge** (the reverse) flags;
* the τ at which agreement is maximal, and the agreement at the recommended τ.

Interpretation text is fixed by the specification, in both languages:
agreement says the reference partition follows density structure visible at
this bandwidth; disagreement says it does not, which may be a property of the
reference method (k-means cuts through uniform regions; it also separates
groups that are not separated by a density valley) and is not by itself an
error of either.

## 12. Low-level entry points (for tests and for reuse)

Both packages export the tree construction on given inputs:
`merge_tree(f, edges)` → peaks table and events; `cut(tree, τ)` → vertex
labels. With `f`, `edges` and the vertex indices fixed, the two languages must
agree exactly (section 14).

## 13. Errors and warnings

| Condition | Behaviour |
|---|---|
| non-numeric, NaN/Inf, fewer than 2 rows, zero columns | error |
| n < 2K + 2 with user K; K < 2; k_graph < 1; m < k_graph + 2 | error |
| `vertices` out of range or duplicated | error |
| `threshold` ≤ 0 or NaN | error |
| more than K exact duplicates of a row | warning (4.3) |
| graph has more than one connected component | message, with the count (each is a cluster at every τ) |
| `dim = "auto"` gives D̂ < 1 or not finite | warning; ambient dimension used |
| d > 50 with `dim = "ambient"` | warning (prominences scale with d) |
| column scale ratio > 100 | warning |
| requested k not attainable | error / warning (section 8) |
| stability below 0.8 at the reported threshold | warning in `summary` and in the report |

Messages have the same wording in both languages (kept in the specification
repository as a table).

## 14. Numerical agreement between the languages

* Given the same `f` and edge list: peaks, parents, events and labels are
  **identical**; levels and prominences agree to 1e-12 (they are copies or
  differences of the inputs).
* Given the same `X` and `vertices` without distance ties: `r_K` agrees to a
  relative 1e-10, f to an absolute 1e-9 (digamma and log-gamma
  implementations differ in the last digits), `D̂` to a relative 1e-9; the
  graph is identical; the tree is identical unless two vertex densities
  differ by less than 1e-9, which the conformance data avoid.
* With different random draws (different `seed`s, or the same seed in the two
  languages) results agree only statistically; the conformance suite checks
  the adjusted Rand index between the two partitions against the
  within-language replicate spread.

Conformance data (JSON, in the specification repository): input matrix,
vertices, expected `r_K`, f, D̂, edges, peaks table, labels at three
thresholds, comparison tables; produced by one implementation, checked by
hand on the small cases, and verified by both in continuous integration.

## 15. Exported files (same names and columns in both languages)

| File | Content |
|---|---|
| `peaks.csv` | the prominence path table (section 9) |
| `labels.csv` | row, cluster at the reported threshold |
| `vertices.csv` | vertex, row, r_K, f, basin |
| `stability.csv` | τ, n_clusters (original fit), stability, count_mode, count_agreement |
| `comparison.csv`, `contingency.csv` | section 11 |
| `parameters.json` | all parameters, D̂, τ₀, the vertex indices, seed, package and dependency versions, date |
| `prominence_path.(pdf|png)`, `merge_tree.(pdf|png)`, `stability_path.(pdf|png)` | figures |
| `methods.md` | a Methods paragraph generated from `parameters.json`; it states what was computed and nothing else |

## 16. Out of scope for 0.1

Other distances; adaptive bandwidth; out-of-sample prediction beyond the
nearest-vertex rule; confidence sets for the tree; the selective refinement of
Phases 1–2 (kept in this repository as a record; the ascent step is listed as
a possible later option).

## References

Chazal, Guibas, Oudot, Skraba (2013) JACM 60(6). Jang & Jiang (2019) ICML.
d'Errico, Facco, Laio, Rodriguez (2021) Information Sciences 560. Ding, Shah,
Condon (2016) Bioinformatics 32(17). Levina & Bickel (2005) NIPS 17. MacKay &
Ghahramani (2005), note on the Levina–Bickel estimator. Ben-Hur, Elisseeff,
Guyon (2002) PSB. Hennig (2007) CSDA 52(1). Hubert & Arabie (1985) Journal of
Classification 2. Full entries: [`phaseA_software_survey.md`](phaseA_software_survey.md).
