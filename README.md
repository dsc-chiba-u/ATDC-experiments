# ATDC-experiments

Proof-of-concept implementation and reproducible benchmarks for Adaptive
Topological Density Clustering (ATDC).

**Current state (Phase A, November 2026).** The research question below was
answered in the negative (Phases 1–2). The project then turned to research
software: is the method that selective refinement could not beat — the uniform
subsample density cluster tree — a sound basis for an R and a Python package?
Phase A measured that before any package was written. Its report is
[`docs/phaseA_report.md`](docs/phaseA_report.md). The judgement: the software
is sound, the purpose first stated for it is not supported — **GO as a density
cluster tree toolkit** for data with at least about a thousand rows and about
ten dimensions or fewer; **not as a validator of k-means analyses**. The
decision is pending with the maintainers. No package exists yet; when built, the packages will live in their own
repository and this one remains the research record.

## Question

> Accurate recovery of density cluster hierarchies may not require uniformly
> accurate estimation of the underlying density function. Selective refinement
> of density and connectivity estimates near uncertain merging events may
> improve computational efficiency and statistical accuracy of hierarchical
> clustering.

This repository tests that hypothesis with one concrete algorithm:

* **Stage A** — merge tree (0-dimensional persistence, as in ToMATo) of a kNN
  log-density on a random subsample;
* **Stage B** — full-data density re-estimation only at the peaks and saddles of
  merge events whose log-prominence may still reach the threshold τ, with the
  tree re-derived lazily until every retained event is verified.

## Result

**Phase 2 judgement: PIVOT** ([`docs/evaluation_v2.md`](docs/evaluation_v2.md)).
Phase 1 ended in REVISE ([`docs/evaluation.md`](docs/evaluation.md)).

Two independent designs of selective refinement were built and compared with
the simplest alternative, *uniform refinement*: take m random points, give each
its kNN density in the full data, run ToMATo on their kNN graph, label the rest
by nearest vertex.

* **v1** (coarse subsample tree, then full-data densities at undecided peaks and
  saddles) beat random refinement but not uniform refinement at equal cost. Its
  overhead exceeded the density evaluations it saved.
* **v2** (uniform refinement on a few hundred vertices, then ascent of
  near-threshold peaks, new vertices at merge locations, optional verification
  of long edges and a held-out test on a disjoint half of the data) ties with
  uniform refinement at equal wall time on unused seeds: exact recovery of the
  true modes 0.917 vs 0.917, difference 0.000 [−0.05, +0.05] over 120 paired
  runs. It places peaks closer to the true modes (+0.07 strict peak F1) and
  loses on a 64-cluster data set.
* The method neither design could beat is the useful result: at n = 10⁶ uniform
  refinement on 1 000 vertices takes 0.85 s and 6 MB against 16.3 s and 3 GB for
  GUDHI's ToMATo on all points, with comparable recovery. It combines known
  ideas (DBSCAN++-style subsample density, ToMATo's tree) and is not claimed as
  new.

## Phase A in brief

* **Subsample size.** From 4 000 vertices the subsample tree is as accurate as
  the tree on all 20 000 points (exact recovery of the true modes 0.86 against
  0.84). What it loses is predictable: a cluster is found when about 12 or
  more vertices fall into it, so a 0.5 % cluster needs m ≥ 4 000. The loss is
  invisible to a stability diagnostic.
* **Dimension.** Right up to d ≈ 5, usable at 10, wrong from 20 when the extra
  coordinates are noise — with all points as well. On 50-d expression data and
  384-d text embeddings as given the tree is unstable; after reduction to 5–10
  dimensions it is level with or above HDBSCAN and at or below k-means.
* **Existing tools.** Same partition as GUDHI's ToMATo (ARI 0.909 against
  0.908 on the true basins), level with densityCut (0.931 over 18 sets; 0.927
  against 0.957 for the tree without the one set built to be undecidable), far
  above HDBSCAN at its defaults (0.464).
* **Sample size.** The tree needs 100–400 rows per cluster where k-means needs
  a few dozen in total; on small tables it reports one cluster where k-means is
  right.
* **As a check of a k-means partition.** On 16 public tables the tree
  confirmed the partition on 2, contradicted it on 1 and could say nothing on
  13, as given and after reduction by principal components alike. A parametric
  check keeps that silence from being read as a refutation; it cannot turn it
  into an answer.
* **Cost.** A million rows in 2-D: 0.9 s and 0.2 GB against 16 s and 3.3 GB
  for ToMATo on all points. Approximate neighbour search does not pay (the
  index costs more than the m exact queries) and is dropped.
* **R and Python.** An R prototype written from the description reproduces the
  Python reference exactly on shared vertices (densities to 9e-14, identical
  graph, peaks and labels) at 1.0–1.3 times the Python time in 2-D and two to
  three times from ten dimensions.

## Documents

| File | Content |
|---|---|
| [`docs/phaseA_report.md`](docs/phaseA_report.md) | Phase A: measurements, judgement, decisions needed |
| [`docs/phaseA_spec_draft.md`](docs/phaseA_spec_draft.md) | common specification for both languages, draft 0.1 |
| [`docs/phaseA_api.md`](docs/phaseA_api.md) | architecture, R and Python APIs, conformance plan (proposal) |
| [`docs/phaseA_software_survey.md`](docs/phaseA_software_survey.md) | existing software, literature, SoftwareX requirements |
| [`docs/literature_review.md`](docs/literature_review.md) | competing methods, what is not new, the remaining claim |
| [`docs/algorithm_design.md`](docs/algorithm_design.md) | candidate designs, the chosen one, pseudocode, cost, limits |
| [`docs/evaluation.md`](docs/evaluation.md) | Phase 1: results, failure cases, judgement |
| [`docs/refinement_v2.md`](docs/refinement_v2.md) | Phase 2: why v1 lost, candidate designs, v2 |
| [`docs/evaluation_v2.md`](docs/evaluation_v2.md) | Phase 2: audit, final comparison, ablation, judgement |

## Implementation status

| Part | Status |
|---|---|
| Merge tree on graphs (`src/atdc/mergetree.py`) | done; tested against hand computation, brute force and GUDHI ToMATo |
| v1: Stage A, Stage B, uniform controls (`src/atdc/core.py`) | done |
| v2: adaptive graph refinement, held-out test (`src/atdc/v2.py`) | done; graph logic tested against the true density |
| Exact-density synthetic data, true trees, metrics | done |
| Wrappers: GUDHI ToMATo, HDBSCAN, DADApy (ADP), AuToMATo | done (their own packages) |
| Uniform subsample tree, reference implementation (`src/atdc/subsample.py`) | done (Phase A); equal to `ATDC2` without refinement on the same vertices |
| True trees of Gaussian mixtures in any dimension (`src/atdc/truth_gmm.py`) | done (Phase A) |
| Stability path and listing of stable thresholds (`src/atdc/diagnostics.py`) | prototype (Phase A) |
| R prototype of the estimator (`benchmarks/r/atdc_proto.R`) | done (Phase A); used only to measure agreement and speed |
| densityCut | run in Phase A through its R package (Bitbucket, 0.0.1); not run in Phases 1–2 |
| Leiden | run in Phase A (`leidenalg`) |
| AuToMATo | Phase 1 only (n = 5 000); **not rerun in Phase A** — minutes per fit |
| Nested multi-level refinement, adaptive bandwidth, stratified vertex selection | **not implemented** (see design documents) |
| Real data | Phase A: 16 public tables, 68 579 cells (PCA-50), 18 221 documents (384-d and 768-d embeddings) |
| R and Python packages, manuscript | **not started** — waits for the decision on the Phase A report |

## Running

```sh
# environments (project-local; the global Python is untouched)
mamba env create -p ./.conda-env -f environment.yml
mamba create -p ./.conda-env-dadapy -c conda-forge python=3.11 "numpy<2" scipy scikit-learn pip
./.conda-env-dadapy/bin/pip install dadapy==0.3.4        # optional, for the ADP comparison

PY=./.conda-env/bin/python
$PY -m pytest tests -q                    # 53 tests, half a minute

$PY benchmarks/02_sensitivity.py          # development seeds: parameter choice
$PY benchmarks/01_ablation.py             # coarse / uniform / random / selective / full
$PY benchmarks/01_ablation.py --n 200000 --seeds 5 --tag n200k \
     --datasets gauss3_close hetero moons circles hier unimodal gauss3_noise3
$PY benchmarks/03_competitors.py          # + ToMATo (GUDHI), HDBSCAN, DADApy
$PY benchmarks/03_competitors.py --n 5000 --seeds 3 --automato --tag automato_n5k \
     --datasets gauss3_close hetero moons circles hier unimodal
$PY benchmarks/03b_hdbscan.py             # HDBSCAN with noise-aware scoring
$PY benchmarks/05_tradeoff.py             # accuracy against cost
$PY benchmarks/05_tradeoff.py --tag many --datasets many64
$PY benchmarks/05_tradeoff.py --n 100000 --seeds 5 --tag many_n100k --datasets many64
$PY benchmarks/04_scaling.py              # time by stage and peak memory (run alone)
$PY benchmarks/06_report.py               # results/summary/*.csv, results/figures/*.png

# Phase 2
$PY benchmarks/09_audit.py                # stability of the ground-truth tree
$PY benchmarks/07_v2_dev.py               # development seeds 200-204
$PY benchmarks/07b_v2_dev_lean.py         # development seeds 205-214
$PY benchmarks/08_v2_final.py             # final seeds 300-309 (run alone: wall time)
$PY benchmarks/08_v2_final.py --n 200000 --seeds 5 --tag n200k --no_hdbscan \
     --datasets gauss3_close hetero moons circles hier unimodal gauss3_noise3 hier_noise3 many64
$PY benchmarks/10_v2_scaling.py           # time by stage and memory (run alone)
$PY benchmarks/11_report_v2.py            # results/summary/08_*, results/figures/08_*

# Phase A (R environment and data environment: see environment.yml)
./.conda-env-data/bin/python benchmarks/data_prep.py pbmc      # data/pbmc68k_pca50.npy
./.conda-env-data/bin/python benchmarks/data_prep.py text      # data/ng20_*.npy
$PY benchmarks/12_subsample.py --tag dev --variants base kg5 kg20 mid     # seeds 400-404
$PY benchmarks/12_subsample.py --tag final --seed0 500 --seeds 10         # seeds 500-509
$PY benchmarks/13_dimension.py --tag dev2                                 # seeds 400-404
$PY benchmarks/13_dimension.py --tag final --seed0 500 --seeds 10
$PY benchmarks/15_competitors_A.py --tag main
$PY benchmarks/17_tabular.py --tag main2            # 16 public tables (scikit-learn, OpenML)
$PY benchmarks/19_small_n.py --tag main
$PY benchmarks/20_real_data.py --tag main
$PY benchmarks/21_detectability.py --tag main        # also --tag pca10 --pca 10, --tag pca5 --pca 5
$PY benchmarks/21_detectability.py --synthetic       # the same check where the answer is known
$PY benchmarks/14_cross_language.py --tag main --repeat 3   # R against Python (run alone)
$PY benchmarks/14_cross_language.py --tag matrix --nn matrix dbscan --backend auto \
     --configs hier_noise8:20000:1000 hier_noise8:200000:1000 blobs5_d20:20000:1000 \
               blobs5_d20:20000:4000 hier_noise48:20000:4000      # R matrix search (run alone)
$PY benchmarks/16_cost.py --tag main                # time by stage, memory (run alone; hours)
$PY benchmarks/16_cost.py --tag brute2 --only sub:brute
$PY benchmarks/18_report_A.py                       # results/summary/A_*, results/figures/A_*
```

All methods run single-threaded. Seeds: Phase 1 reported on 0–9 with Stage B
parameters chosen on 100–104; Phase 2 developed on 200–214 and reported on
300–309; Phase A developed on 400–404 and reported on 500–519, with 700–702
for cost, 800 and 810 for the public tables, 811–815 for the controlled cases
of the detectability check and 900 for the real data. `results/raw/01_ablation_v1.csv` is the first Phase-1 run, made before
its parameter choice. The development rounds 07 and 07b were run as four
processes and concatenated.

## Layout

```
src/atdc/      mergetree.py  core.py (v1)  v2.py  density.py  datasets.py  truth.py  metrics.py  baselines.py
               subsample.py  truth_gmm.py  diagnostics.py                      (Phase A)
tests/         correctness of the tree, the estimators and the ground truth
benchmarks/    numbered scripts; each writes results/raw/NN_*.csv
benchmarks/r/  R prototype, Rcpp loop used for timing only, densityCut runner
data/          real data prepared by data_prep.py (not in git; provenance in data/*.json)
results/raw/       one row per run
results/summary/   tables built by 06_report.py, 11_report_v2.py, 18_report_A.py
results/figures/   figures built by the same scripts
docs/          literature review, design, evaluations, Phase A report and proposals
```

## Minimal example (Phase A reference implementation)

```python
import numpy as np, sys
sys.path.insert(0, "src")
from atdc import datasets
from atdc.subsample import SubsampleTree
from atdc.diagnostics import default_K, noise_floor

X = datasets.make("hier").sample(20000, np.random.default_rng(0))
K = default_K(len(X))
fit = SubsampleTree(X, m=4000, K=K, seed=0, dim="auto")
est = fit.estimate(noise_floor(K))          # labels for all rows at the default threshold
print(len(est.peak_idx), "clusters;", fit.n_clusters(2.0), "at threshold 2")
```

## Minimal example (Phases 1–2)

```python
import numpy as np, sys
sys.path.insert(0, "src")
from atdc import datasets
from atdc.v2 import ATDC2

X = datasets.make("hier").sample(20000, np.random.default_rng(0))

# uniform refinement: 1000 vertices with full-data densities, ToMATo tree
est = ATDC2(X, K=100, m0=1000, seed=0).estimate(tau=0.5)

# v2: fewer vertices, then ascent and saddle vertices
est2 = ATDC2(X, K=100, m0=400, seed=0).refine(0.5, hidden=False, edges=False).estimate(0.5)
print(len(est.peak_idx), len(est2.peak_idx), "clusters")
```
