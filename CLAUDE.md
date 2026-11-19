# CLAUDE.md — ATDC-experiments

## What this repository is

Research proof of concept and benchmarks for Adaptive Topological Density
Clustering (ATDC): does *selective* refinement of density estimates near
uncertain merge events recover a density cluster tree more cheaply than uniform
refinement? It is a hypothesis test, not a product.

Read first: `docs/evaluation_v2.md` (Phase 2 result and judgement: **PIVOT**),
`docs/refinement_v2.md` (why v1 lost, the v2 design), `docs/evaluation.md` and
`docs/algorithm_design.md` (Phase 1), `docs/literature_review.md` (what is not
new).

## State of the research (after Phase 2)

- Two designs of selective refinement (v1 `core.py`, v2 `v2.py`) were tested.
  Neither beats **uniform refinement** (m random vertices with full-data kNN
  densities, ToMATo on their graph) at equal wall time. v2 vs uniform on unused
  seeds: exact mode recovery 0.917 vs 0.917, difference 0.000 [−0.05, +0.05].
- Structural reason: one density evaluation costs ~10 µs, a few hundred to a few
  thousand consistent vertices suffice, and wall time is dominated by the
  full-data index and the labelling. There is nothing left for an adaptive
  strategy to save in low dimension.
- What v2 does improve: peak location (ascent). What it does not: saddles, merge
  order, the 64-cluster case (worse).
- Sample splitting with a held-out prominence estimate is calibrated but lowers
  accuracy at fixed bandwidth: it removes the upward selection bias that was
  compensating the downward smoothing bias.
- Do not present selective refinement as the contribution. The useful object is
  the uniform subsample tree, which is a combination of known ideas.

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
- Design choices are made on development seeds, reported comparisons on seeds
  not seen before: Phase 1 dev 100–104 / final 0–9; Phase 2 dev 200–214 / final
  300–309. A new design needs new final seeds.
- Never seed a method with the plain integer that generated the data
  (`default_rng(seed)` twice gives correlated streams); use seed sequences.
- Wall time is the criterion for cost; run timing benchmarks alone.
- Existing result files are not overwritten; new evaluations get new names.
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
- `src/atdc/core.py` — v1: Stage A, Stage B, uniform controls
- `src/atdc/v2.py` — v2: uniform refinement, graph refinement, held-out test
- `src/atdc/truth.py`, `datasets.py`, `metrics.py` — true trees, exact
  densities, evaluation
- `src/atdc/baselines.py` — wrappers around existing implementations
- `benchmarks/NN_*.py` — numbered, each writes `results/raw/NN_*.csv`;
  `06_report.py` (Phase 1) and `11_report_v2.py` (Phase 2) build
  `results/summary/` and `results/figures/`
