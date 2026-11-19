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

## Result in one paragraph

Judgement: **REVISE** (details and numbers in [`docs/evaluation.md`](docs/evaluation.md)).
Selective refinement works as designed: it clearly beats random refinement at
the same budget and reproduces uniform refinement of the same subsample at about
a quarter of the cost. But it does **not** beat the simplest alternative at
equal cost — evaluating the full-data density at *all* points of a *smaller*
subsample — on peak detection or ARI, and it is slower in wall time. Most of the
10–30× saving over full-data ToMATo comes from the subsample-plus-full-data-density
step, which is the known idea behind DBSCAN++. The first half of the hypothesis
is supported; the "selective" half is not, in the regimes tested.

## Documents

| File | Content |
|---|---|
| [`docs/literature_review.md`](docs/literature_review.md) | competing methods, what is not new, the remaining claim |
| [`docs/algorithm_design.md`](docs/algorithm_design.md) | candidate designs, the chosen one, pseudocode, cost, limits |
| [`docs/evaluation.md`](docs/evaluation.md) | results, failure cases, GO / REVISE / PIVOT / STOP |

## Implementation status

| Part | Status |
|---|---|
| Merge tree on graphs (`src/atdc/mergetree.py`) | done; tested against hand computation, brute force and GUDHI ToMATo |
| Stage A, Stage B, uniform controls (`src/atdc/core.py`) | done |
| Exact-density synthetic data, true trees, metrics | done |
| Wrappers: GUDHI ToMATo, HDBSCAN, DADApy (ADP), AuToMATo | done (their own packages) |
| densityCut | **not run** — R only, no Python implementation found |
| Nested multi-level refinement, adaptive bandwidth at refined points | **not implemented** (see design document) |
| Real data | **not run** |
| Packaging, R port, manuscript | out of scope |

## Running

```sh
# environments (project-local; the global Python is untouched)
mamba env create -p ./.conda-env -f environment.yml
mamba create -p ./.conda-env-dadapy -c conda-forge python=3.11 "numpy<2" scipy scikit-learn pip
./.conda-env-dadapy/bin/pip install dadapy==0.3.4        # optional, for the ADP comparison

PY=./.conda-env/bin/python
$PY -m pytest tests -q                    # 34 tests, a few seconds

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
```

All methods run single-threaded. Reported comparisons use seeds 0–9; the
parameters of Stage B were chosen on seeds 100–104 (`02_sensitivity.py`).
`results/raw/01_ablation_v1.csv` is the first run, made before that choice.

## Layout

```
src/atdc/      mergetree.py  core.py  density.py  datasets.py  truth.py  metrics.py  baselines.py
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
from atdc.core import ATDC

X = datasets.make("hier").sample(20000, np.random.default_rng(0))
a = ATDC(X, K=100, q=0.1, seed=0).stage_a()      # coarse tree on 2000 points
a.refine(tau=0.5, z_saddle=2.0)                  # selective refinement
est = a.estimate(tau=0.5)
print(len(est.peak_idx), "clusters;", est.info["frac_refined"], "of the subsample refined")
```
