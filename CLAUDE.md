# CLAUDE.md — ATDC-experiments

## What this repository is

Research record and benchmarks for Adaptive Topological Density Clustering
(ATDC). Phases 1–2 tested a hypothesis — does *selective* refinement of density
estimates near uncertain merge events recover a density cluster tree more
cheaply than uniform refinement? — and answered no. Phase A (after the pivot to
research software for SoftwareX, R and Python) measured whether the uniform
subsample tree is a sound basis for a package. This repository stays an
experiment repository; packages, when built, go into their own repository.

Read first: `docs/phaseA_report.md` (Phase A measurements and judgement:
**GO as a density cluster tree toolkit, not as a validator of k-means
analyses**; decisions pending), `docs/phaseA_spec_draft.md`
and `docs/phaseA_api.md` (proposals), `docs/evaluation_v2.md` (Phase 2:
**PIVOT**), `docs/refinement_v2.md`, `docs/evaluation.md`,
`docs/algorithm_design.md`, `docs/literature_review.md` and
`docs/phaseA_software_survey.md` (what is not new, what exists).

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

## State after Phase A (uniform subsample tree as software)

- The estimator is `src/atdc/subsample.py` (`SubsampleTree`): m uniform
  vertices, kNN log-density against all rows, merge tree on the vertex kNN
  graph, prominence threshold, nearest-vertex labels. Nothing in it is new.
- It is as accurate as the full-data tree from m = 4 000; it finds a cluster
  only when about 12 or more vertices fall into it; stability under resampling
  does not reveal a missed cluster ("stable" is not "complete").
- Density trees (this one, ToMATo, HDBSCAN) stop working above an intrinsic
  dimension of about 10–20 and need 100–400 rows per cluster. Small tables and
  embeddings as given are outside the working regime; never present them as
  successes, and never select examples to hide this.
- The use case the plan put first — defending the number of clusters of a
  k-means analysis — is not supported: on 16 public tables the comparison was
  informative on 3 (2 confirmed, 1 contradicted), with or without PCA. The
  detectability check (`21_detectability.py`) prevents reading silence as
  refutation but cannot say "not supported" when k-means cut one mode. Do not
  describe the software as a validator of k-means results.
- Approximate neighbour search does not pay (index on n rows for m queries)
  and is not part of version 1. R is 2–3 times slower than Python from ten
  dimensions (neighbour search, not the tree).
- Accuracy equals ToMATo and is level with densityCut (each ahead on
  different data; never claim to beat either). The candidate software
  contribution is the R + Python pair from one specification, the diagnostic
  workflow and the stated limits — not the estimator.
- Defaults chosen in Phase A: K = min(100, max(10, n // 20)), m = min(n, 4000),
  k_graph = 10, `dim="auto"`, threshold = noise floor 3.5·sqrt(2·trigamma(K)).
  The stability rule never replaces the default partition; it lists thresholds
  with stability ≥ 0.8.
- Proposed architecture: shared specification and conformance data, native
  numerics in each language, no compiled code of our own (the R prototype
  `benchmarks/r/atdc_proto.R` reproduces Python exactly on shared vertices).
- Phase B (package implementation) starts only after the user has accepted the
  Phase A report and decided the repository layout, name, organisation,
  visibility and licence. Creating a GitHub repository is the user's decision.

## Positioning (keep consistent in code comments and docs)

- Peaks, prominence, union-find merging are ToMATo; peak-vs-saddle significance
  is DPA; subsample density against the full data is DBSCAN++. None of these is
  claimed. `R = log p_peak − log p_saddle` is 0-dim persistence of a log-density.
- The selective refinement loop was the only candidate algorithmic
  contribution. It was compared against uniform refinement **at equal cost**
  (`global_matched`) twice and did not win; it is closed (no v3) and is not
  claimed as novelty anywhere.
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
  300–309; Phase A dev 400–404 / final 500–519, cost 700–702, public tables
  800 and 810, detectability controls 811–815, real-data vertex draws 900. A new design needs new final seeds.
  The 16 public tables were looked at while the stability rule was changed:
  they are development data for that rule.
- Never seed a method with the plain integer that generated the data
  (`default_rng(seed)` twice gives correlated streams); use seed sequences.
- Wall time is the criterion for cost; run timing benchmarks alone.
- Existing result files are not overwritten; new evaluations get new names.
- No package implementation or manuscript before the user's decision on the
  Phase A report. R and Python are equals: neither is a port of the other, and
  the R side must not require Python.
- Do not add features or select results for the sake of the submission.
- Progress reports to the user are in Japanese; code, comments and docs in
  English.

## Environment and execution

```sh
mamba env create -p ./.conda-env -f environment.yml      # main environment
./.conda-env/bin/python -m pytest tests -q               # correctness
./.conda-env/bin/python benchmarks/01_ablation.py        # see README for the rest
```

DADApy needs numpy < 2 and lives in `./.conda-env-dadapy` (see README). R
(prototype, densityCut) lives in `./.conda-env-r`; start `Rscript` with
`./.conda-env-r/bin` on `PATH`. Data preparation (scanpy, sentence-transformers)
lives in `./.conda-env-data`; `data/` is not in git. Recipes are in
`environment.yml`. All benchmarks run single-threaded; never run anything else
while a timing benchmark (`14`, `16`) runs.

## Layout

- `src/atdc/mergetree.py` — merge tree / 0-dim persistence on graphs (tested
  against brute force and GUDHI)
- `src/atdc/core.py` — v1: Stage A, Stage B, uniform controls
- `src/atdc/v2.py` — v2: uniform refinement, graph refinement, held-out test
- `src/atdc/subsample.py` — Phase A reference estimator; `diagnostics.py` —
  default rules, stability path, stable thresholds (prototype)
- `src/atdc/truth.py`, `truth_gmm.py`, `datasets.py`, `metrics.py` — true trees
  (2-D grid; Gaussian mixtures in any dimension), exact densities, evaluation
- `src/atdc/baselines.py` — wrappers around existing implementations
- `benchmarks/NN_*.py` — numbered, each writes `results/raw/NN_*.csv`;
  `06_report.py` (Phase 1), `11_report_v2.py` (Phase 2) and `18_report_A.py`
  (Phase A, files `A_*`) build `results/summary/` and `results/figures/`
- `benchmarks/r/` — R prototype and densityCut runner (called by `14`, `15`, `16`)
