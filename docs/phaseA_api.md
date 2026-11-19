# ATDC: architecture, API and conformance plan — proposal (Phase A)

Companion of the specification draft ([`phaseA_spec_draft.md`](phaseA_spec_draft.md)).
The specification says what is computed; this document says how the two
packages are built and what the user types. Everything here is a proposal to
be confirmed before Phase B; code examples are designs, not tested code.

## 1. Architecture

| | A. two independent implementations | B. shared compiled core (C++/Rust) | C. shared specification and conformance data, native numerics in each language |
|---|---|---|---|
| Install | trivial | compiled: wheels for every platform on PyPI, a compiler or binaries on CRAN | trivial: no compiled code of our own |
| Same results | nothing enforces it | by construction for the core | enforced by conformance tests in CI |
| Speed | — | best | sufficient (measured, below) |
| Maintenance | two code bases drift | one core, two bindings, two build systems | two small code bases, one specification |
| CRAN / PyPI risk | low | medium (toolchains, `cibuildwheel`, CRAN compiler notes) | low |

**Proposed: C.** Reasons, from measurements in `phaseA_report.md`:

* The computation is two neighbour searches (delegated to mature libraries in
  both languages), sorting, and one union–find loop over about m edges after
  a vectorised reduction. With m = 4 000 that loop takes a few hundredths of a
  second in plain R and in plain Python. There is nothing for a compiled core
  to speed up that the neighbour libraries do not already do.
* The R prototype reproduced the Python reference exactly on the same data and
  vertices in 15 of 15 configurations (densities to 9e-14, identical graph,
  peaks and labels), with neither sharing code. Agreement is achievable
  through the specification.
* The price, measured: in ten dimensions and more R is two to three times
  slower than Python (seconds), because of the neighbour search, not the
  tree. One small compiled routine in the R package would close the gap; it is
  not proposed for version 1 (report §4.3).
* No compiled code means: `pip install atdc` and `install.packages("atdc")`
  never need a compiler, and neither package needs the other language.

Dependencies. Python: numpy, scipy, scikit-learn, pandas, matplotlib. R:
`Imports:` one k-d tree package (RANN proposed: no dependencies of its own,
within 10 % of FNN; `dbscan::kNN` is faster from ten dimensions but slower in
two), `stats`, `graphics`, `utils`; above about 15 dimensions a matrix search
in base R; `Suggests:` ggplot2, knitr,
rmarkdown, testthat, jsonlite. No approximate neighbour search in version 1:
measured in Phase A, an index on all rows costs more than the m exact queries
it would replace (report §4), so pynndescent, hnswlib and RcppHNSW are not
dependencies, not even optional ones.

### Repositories

Proposed: **one repository** (`atdc`) for the specification, both packages,
the conformance data and the paper; `ATDC-experiments` stays as the research
record and keeps the benchmarks.

```
atdc/
  spec/            specification.md, messages.md, CHANGELOG
  conformance/     cases/*.json (inputs + expected), generate.py, README
  python/          pyproject.toml, src/atdc/, tests/, docs/
  r/               DESCRIPTION, R/, man/, tests/testthat/, vignettes/
  examples/        the same analyses in both languages, executed in CI
  paper/           SoftwareX manuscript, figures built from examples/
  .github/workflows/   python.yml, r.yml, conformance.yml
```

A single repository makes "the specification changed" one commit that CI checks
against both packages. The price: CRAN and pkgdown work from a sub-directory
(supported: `R CMD build r`, `remotes::install_github("…/atdc", subdir = "r")`).
The existing sibling projects use three repositories (`-r`, `-py`,
`-experiments`) with the specification in the R repository; that layout also
works and is the alternative if consistency with them matters more. **This is
a decision for the maintainers** (it also fixes the repository name and its
visibility).

Versions: both packages carry the same `MAJOR.MINOR`, equal to the
specification version they implement; `PATCH` is per language.

## 2. Python API (scikit-learn estimator)

```python
import numpy as np
from sklearn.cluster import KMeans
from atdc import ATDC

km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)     # the existing analysis

model = ATDC(random_state=0).fit(X)          # density cluster tree with default settings
model.labels_                                # partition at model.threshold_
model.n_clusters_
model.peaks_                                 # DataFrame: the prominence path table
model.plot_path()                            # number of clusters against the threshold
model.plot_tree()                            # merge tree of the peaks

stab = model.stability(n_replicates=20)      # DataFrame: the stability path (B more fits)
model.plot_stability()
model.recommend()                            # thresholds whose partition is stable

cmp = model.compare_labels(km.labels_)       # agreement, contingency, splits and merges
cmp.summary()
model.report("atdc_report", reference_labels=km.labels_)   # files of spec section 15
```

| Constructor argument | Spec name | Default |
|---|---|---|
| `n_neighbors` | K | `None` → default rule |
| `n_vertices` | m | `None` → `min(n, 4000)` |
| `n_graph_neighbors` | k_graph | 10 |
| `dim` | dim | `"auto"` |
| `threshold` | threshold | `"auto"` |
| `n_clusters` | alternative to `threshold` | `None` |
| `algorithm` | neighbour search: `"auto"`, `"kd_tree"`, `"brute"` | `"auto"` |
| `random_state` | seed | `None` |

Methods: `fit(X)`, `fit_predict(X)`, `predict(X_new)` (nearest-vertex
assignment of new rows — an assignment, not a re-estimation of the tree; the
docstring says so), `get_labels(threshold=None, n_clusters=None)`,
`get_params()`, `set_params()` (inherited from `BaseEstimator`, `ClusterMixin`),
`stability(...)`, `recommend(...)`, `compare_labels(labels, threshold=None)`,
`plot_path(ax=None)`, `plot_tree(ax=None)`, `plot_stability(ax=None)`,
`plot_diagnostics()` (the three panels), `summary()` (text), `report(dir, ...)`.

Fitted attributes: `labels_`, `threshold_`, `n_clusters_`, `peaks_`,
`vertices_` (DataFrame), `vertex_indices_`, `dim_`, `n_neighbors_`,
`n_features_in_`, `stability_` (after `stability`). Functions:
`atdc.merge_tree(f, edges)`, `atdc.cut_tree(tree, threshold)`.

Plot methods return the Matplotlib `Axes`/`Figure` so that the caller can
restyle them. Tables are pandas DataFrames with the column names of the
specification. The estimator passes scikit-learn's `check_estimator` where the
checks apply to a clusterer.

## 3. R API (S3, in the style of `stats` and `cluster`)

```r
km <- kmeans(x, centers = 4, nstart = 10)        # the existing analysis

set.seed(1)
fit <- atdc(x)                                   # density cluster tree with default settings
fit                                              # print: clusters, threshold, resolution
summary(fit)
fit$cluster                                      # partition at fit$threshold (as in kmeans)
clusters(fit, k = 4)                             # another cut; or clusters(fit, threshold = 1)
as.data.frame(fit)                               # the prominence path table

plot(fit)                                        # prominence path
plot(fit, which = "tree")                        # merge tree of the peaks

stab <- stability(fit, B = 20)                   # stability path (B more fits)
plot(stab)
recommend(stab)

cmp <- compare(fit, km$cluster)                  # agreement, contingency, splits and merges
cmp
report(fit, dir = "atdc_report", reference = km$cluster)
```

| Argument of `atdc()` | Spec name | Default |
|---|---|---|
| `x` | X: numeric matrix or data frame | — |
| `neighbors` | K | `NULL` → default rule |
| `vertices` | m, or a vector of row indices | `NULL` → `min(n, 4000)` |
| `graph.neighbors` | k_graph | 10 |
| `dim` | dim | `"auto"` |
| `threshold` | threshold | `"auto"` |
| `k` | alternative to `threshold` | `NULL` |

Randomness follows R's convention: no `seed` argument, the caller uses
`set.seed()`; the drawn vertices are stored in the object.

Class `"atdc"`: a list with `cluster`, `threshold`, `peaks` (data frame),
`vertices` (data frame), `dim`, `neighbors`, `call`. S3 methods: `print`,
`summary` (and `print.summary.atdc`), `plot` (`which = "path" | "tree"`; base
graphics), `predict(object, newdata)` (nearest-vertex assignment of new rows),
`as.data.frame`, `as.hclust` and `as.dendrogram` (see below). Generics defined
by the package: `clusters()`, `stability()`, `recommend()`, `compare()`,
`report()`. Classes `"atdc_stability"` and `"atdc_comparison"` have `print`,
`plot` and `as.data.frame` methods. If ggplot2 is installed,
`ggplot2::autoplot()` methods return ggplot objects for the same figures; the
package works without it.

`cutree()` is not generic in `stats`, so it is not overloaded; `clusters()` is
the accessor. `as.hclust(fit)` returns the tree **of the peaks** (leaves =
peaks, height = highest peak minus merge level), which base R can draw and
which `cutree(h = )` cuts at a density level. Its help page states that this
is a level-set cut, not the prominence cut of `clusters()`, and that the
object is not a hierarchical clustering of the observations.

## 4. The two APIs side by side

| Task | Python | R |
|---|---|---|
| fit | `ATDC().fit(X)` | `atdc(x)` |
| labels | `model.labels_` | `fit$cluster` |
| labels for k clusters | `model.get_labels(n_clusters=4)` | `clusters(fit, k = 4)` |
| labels at a threshold | `model.get_labels(threshold=1.0)` | `clusters(fit, threshold = 1)` |
| peaks table | `model.peaks_` | `as.data.frame(fit)` |
| prominence path | `model.plot_path()` | `plot(fit)` |
| merge tree | `model.plot_tree()` | `plot(fit, which = "tree")` |
| stability path | `model.stability()`; `model.plot_stability()` | `s <- stability(fit)`; `plot(s)` |
| recommendation | `model.recommend()` | `recommend(s)` |
| compare with a partition | `model.compare_labels(labels)` | `compare(fit, labels)` |
| new rows | `model.predict(X_new)` | `predict(fit, newdata)` |
| report | `model.report(dir)` | `report(fit, dir)` |
| seed | `ATDC(random_state=0)` | `set.seed(0)` |
| same vertices in both | `ATDC(vertices=idx0)` (0-based) | `atdc(x, vertices = idx0 + 1L)` |

This table is part of the documentation of both packages, so that a person or
an assistant translating an analysis does not have to guess.

## 5. Conformance tests

Three layers, all run in CI for both packages.

1. **Tree on given inputs** (exact). JSON cases with `f` and `edges`; expected
   peaks table, events and labels at several thresholds. Hand-checked small
   graphs: a path with two peaks; a plateau (equal densities); a disconnected
   graph; a star; ties in edge weights that exercise the order of 7.1; a
   single vertex; random graphs generated once. Both packages must return the
   expected output exactly.
2. **Pipeline on given data and vertices** (tolerances of spec section 14).
   Cases: one Gaussian; separated Gaussians; unequal densities; two moons; a
   hierarchy; a 1 % cluster; background noise; duplicated rows; a constant
   column; columns of very different scale; n just above the minimum.
   Expected `r_K`, f, D̂, edges, peaks, labels.
3. **Behaviour** (same in both): errors and warnings of spec section 13 with
   the same message text; the default-rule values; label numbering; file names
   and column names of the report.

Plus one statistical check per release: independent seeds in the two
languages on three data sets, ARI between languages within the spread between
replicates of one language.

The expected values are generated by the Python package (`conformance/generate.py`),
stored in the repository and never regenerated silently: a change of an
expected value is a specification change with a changelog entry. The
hand-checked cases are verified against a brute-force tree (all pairs,
definition of the merge height) that shares no code with either
implementation.

`conformance.yml` installs both packages, runs both test suites against the
same files and additionally runs each example of `examples/` in both languages
and compares the exported CSV files.
