# ATDC-experiments

Proof-of-concept implementation and reproducible benchmarks for Adaptive
Topological Density Clustering (ATDC).

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

## Documents

| File | Content |
|---|---|
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
| densityCut | **not run** — R only, no Python implementation found |
| Nested multi-level refinement, adaptive bandwidth, stratified vertex selection | **not implemented** (see design documents) |
| Real data | **not run** |
| Packaging, R port, manuscript | out of scope |

## Running

```sh
# environments (project-local; the global Python is untouched)
mamba env create -p ./.conda-env -f environment.yml
mamba create -p ./.conda-env-dadapy -c conda-forge python=3.11 "numpy<2" scipy scikit-learn pip
./.conda-env-dadapy/bin/pip install dadapy==0.3.4        # optional, for the ADP comparison

PY=./.conda-env/bin/python
$PY -m pytest tests -q                    # 46 tests, a few seconds

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
```

All methods run single-threaded. Seeds: Phase 1 reported on 0–9 with Stage B
parameters chosen on 100–104; Phase 2 developed on 200–214 and reported on
300–309. `results/raw/01_ablation_v1.csv` is the first Phase-1 run, made before
its parameter choice. The development rounds 07 and 07b were run as four
processes and concatenated.

## Layout

```
src/atdc/      mergetree.py  core.py (v1)  v2.py  density.py  datasets.py  truth.py  metrics.py  baselines.py
tests/         correctness of the tree, the estimator and the ground truth
benchmarks/    numbered scripts; each writes results/raw/NN_*.csv
results/raw/       one row per run
results/summary/   tables built by 06_report.py
results/figures/   figures built by 06_report.py
docs/          literature review, design, evaluation
```

## Minimal example

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
