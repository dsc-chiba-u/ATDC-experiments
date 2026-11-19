# CLAUDE.md — ATDC-experiments

## What this repository is

Research proof of concept and benchmarks for Adaptive Topological Density
Clustering (ATDC): does *selective* refinement of density estimates near
uncertain merge events recover a density cluster tree more cheaply than uniform
refinement? It is a hypothesis test, not a product.

Read first: `docs/evaluation.md` (results and the GO/REVISE/PIVOT/STOP
judgement), `docs/algorithm_design.md` (the design and its limits),
`docs/literature_review.md` (what is not new).

## Positioning (keep consistent in code comments and docs)

- Peaks, prominence, union-find merging are ToMATo; peak-vs-saddle significance
  is DPA; subsample density against the full data is DBSCAN++. None of these is
  claimed. `R = log p_peak − log p_saddle` is 0-dim persistence of a log-density.
- The only candidate contribution is the selective refinement loop. Having
  implemented it is not evidence that it helps: the claim needs the comparison
  against uniform refinement **at equal cost** (`global_matched`).
- Computational accuracy (how well the full-data estimator at bandwidth K is
  approximated) and statistical accuracy (how good that estimator is) are
  different axes. ATDC only moves along the first.
- Ground truth is the merge tree of the true density, never the generating
  mixture components.

## Hard rules

- No fabricated or hand-edited results. Everything under `results/` is written
  by a script in `benchmarks/`. A method that was not run is recorded as not run
  with the reason (e.g. densityCut: R only).
- Competitors are used through their own packages (`gudhi`, `hdbscan`,
  `dadapy`, `automato`); do not re-implement them.
- Design choices are made on development seeds (`02_sensitivity.py`, seeds
  100+); reported comparisons use seeds 0–9.
- No packaging, R port, or manuscript until asked.
- Progress reports to the user are in Japanese; code, comments and docs in
  English.

## Environment and execution

```sh
mamba env create -p ./.conda-env -f environment.yml      # main environment
./.conda-env/bin/python -m pytest tests -q               # correctness
./.conda-env/bin/python benchmarks/01_ablation.py        # see README for the rest
```

DADApy needs numpy < 2 and lives in `./.conda-env-dadapy` (see README). All
benchmarks run single-threaded.

## Layout

- `src/atdc/mergetree.py` — merge tree / 0-dim persistence on graphs (tested
  against brute force and GUDHI)
- `src/atdc/core.py` — Stage A, Stage B, uniform controls
- `src/atdc/truth.py`, `datasets.py`, `metrics.py` — true trees, exact
  densities, evaluation
- `src/atdc/baselines.py` — wrappers around existing implementations
- `benchmarks/NN_*.py` — numbered, each writes `results/raw/NN_*.csv`;
  `06_report.py` builds `results/summary/` and `results/figures/`
