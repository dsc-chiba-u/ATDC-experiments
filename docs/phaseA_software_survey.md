# Phase A: existing software and literature (survey of 2026-11-19)

Scope: what already exists for density cluster trees and their diagnostics, in
R and in Python; whether the package name is free; which definitions of
prominence and which stability diagnostics can be adopted without inventing
statistics; what SoftwareX requires. Every row was checked against the package
page, the repository or the paper on the date above; items that could not be
checked are listed at the end. Nothing in this file is a result of this
repository.

## 1. Name

`atdc` is unused on PyPI (JSON API 404 for `atdc` and `ATDC`), on CRAN (no
case-insensitive match among current packages or the archive), in the
Bioconductor release indexes and on conda-forge. Unrelated GitHub repositories
use the acronym.

## 2. Software

| Package | Where, latest release | Tree / diagnostic plots | Subsampling | Stability diagnostic | Compare with a partition |
|---|---|---|---|---|---|
| TDA `clusterTree` (R) | CRAN 1.9.4, 2025-02 | kNN/KDE cluster tree; lambda, kappa, alpha, r plots | no | not for the tree (bootstrap targets persistence diagrams) | no |
| dbscan `hdbscan`, `extractFOSC` (R) | CRAN 1.3.0, 2026-11 | dendrogram, simplified tree, reachability plot | no; `predict` by nearest fitted point | per-cluster stability score, membership probability | no |
| densityClust (R) | CRAN 0.3.3, 2024-01 | decision plot; no tree | no | no | no |
| densitycut (R) | Bitbucket `jerry00/densitycut_dev` only, 0.0.1, last commit 2016-04; GPL (>= 2) in DESCRIPTION | cluster tree over a saliency index | no (approximate kNN) | cluster number spanning the longest saliency range | `ComputeNMI` |
| pdfCluster (R) | CRAN 1.0-4, 2022-12 | mode function, cluster tree, density-based silhouette | no | no | `adj.rand.index` |
| mclust (R) | CRAN 6.1.4, 2026-11 | BIC plot, `combiTree`, `entPlot`, `gmmhd` | no | `MclustBootstrap` (parameters) | `adjustedRandIndex`, `classError` |
| fpc (R) | CRAN 2.2-15, 2026-08 | none for trees | n/a | `clusterboot`, `nselectboot`, `prediction.strength` | n/a |
| ks `kms`, LPCM `ms` (R) | CRAN | mean-shift scatter plots | no | no | no |
| clustree (R) | CRAN 0.5.1, 2023-11 | tree of user-supplied clusterings across resolutions | n/a | n/a | its purpose |
| hdbscan (Python) | PyPI 0.8.44, 2026-06 | condensed tree, single-linkage tree, MST plots | no; `approximate_predict` | `cluster_persistence_`, probabilities | no |
| scikit-learn `HDBSCAN` | 1.9.1 | none; no tree attribute, no predict | no | `probabilities_` | no |
| GUDHI `Tomato` (Python) | PyPI 3.13.0, 2026-07 | `plot_diagram()`; merge tree as arrays, no dendrogram | no | diagram gap by eye | no |
| automato (Python) | PyPI 0.1.1, 2025-08; AGPL-3.0; scikit-learn API | not checked | no | bottleneck bootstrap sets the threshold | no |
| DeBaCl (Python) | PyPI 1.1, 2016; Python 2.7 only | level set tree plot | no | no | no |
| DADApy (Python) | PyPI 0.3.4, 2026-05 | decision graph, ADP dendrogram | no | Z-test of peak against saddle | `return_label_overlap` |
| fast_hdbscan (Python) | PyPI 0.3.4, 2026-10 | condensed and single-linkage tree objects | no | not checked | no |
| persistable-clustering (Python) | PyPI 0.5.3, 2024-07 | interactive UI, prominence diagram, vineyards | **yes** (`subsample`) | no resampling | no |

Also seen: Modalclust and multimode (R), DPA and tomaster (Python, 2021),
`jenniferjang/dbscanpp` (DBSCAN++ code, GitHub).

**Reading.** Every ingredient exists somewhere: cluster trees (TDA, pdfCluster,
DeBaCl, hdbscan), prominence diagrams (GUDHI, persistable), subsampling
(persistable, DBSCAN++), resampling stability (fpc), comparison of partitions
(clustree, mclust). No package was found that (i) estimates a density cluster
tree on a subsample with full-data densities, (ii) shows the cluster count
against the prominence threshold together with a resampling stability
diagnostic, (iii) compares the result with a partition the user already has,
and (iv) does so in R and in Python from one specification.

## 3. The same analysis in R and in Python

None for density cluster trees. Closest: **genieclust** (and lumbermark,
deadwood, quitefastmst) — one C++ core, matching releases on CRAN and PyPI,
mutual-reachability spanning trees, with a SoftwareX paper (Gagolewski 2021),
which is a direct precedent for the format; mlpack bindings (`dbscan`,
`mean_shift`, not hierarchical); fastcluster (agglomerative); R `dbscan`
called from Python through rpy2 (a bridge, not a second implementation).

## 4. Prominence on a log-density

No paper was found that uses "relative prominence" as a named term for
log-density merge trees. The idea appears in four forms:

* additive prominence = birth − death of a peak in 0-dim superlevel persistence
  (L1); GUDHI's `Tomato` defaults to `logDTM`, so its threshold is already on a
  log scale;
* log-density difference between peak and saddle against its error (L2, DPA);
* valley-to-peak density ratio (L3, densityCut's saliency index);
* level set at a fraction of the peak height (L4, Quickshift++).

A difference of log-densities is the log of a density ratio, so "absolute
prominence of log f" and "relative prominence of f" are the same number.

## 5. Stability diagnostics that exist

* resampling of partitions: S1–S6 (S4–S6 implemented in `fpc`);
* gap in the persistence diagram: L1 and the GUDHI documentation;
* bootstrap confidence sets for diagrams and trees: S7–S9, L9.

## 6. SoftwareX guide for authors (page read 2026-11-19, no revision date shown)

* "Maximum word count: 4000 excluding: title, authors, affiliations,
  references, metadata tables and including: abstract, running text, captions,
  footnotes. Maximum six (6) figures." (Secondary sites still quote 3000.)
* The journal template is mandatory (Word or LaTeX); it also says "Maximum 6
  pages" and an abstract of about 100 words.
* Required sections: Motivation and significance; Software description;
  Illustrative examples; Impact; Conclusions.
* Mandatory code metadata table: C1 current code version, C2 permanent GitHub
  link, C3 licence, C4 versioning system, C5 languages/tools/services, C6
  compilation requirements and dependencies, C7 link to documentation, C8
  support e-mail.
* Repository on GitHub with `README.md`, `LICENSE.txt`, source under `src`;
  accepted code is copied to the journal's GitHub organisation.
* The code must be open source under a recognised licence.

## Not verified

SoftwareX's list of accepted licences; the Word template; the JACM wording of
L1 (the 2011 conference version was read); the definitions in the dip-test and
excess-mass papers; the wording of L5 (formula from the hdbscan documentation);
automato's plots; fast_hdbscan's stability attributes.

## References

**Prominence and mode significance**

* L1. Chazal, Guibas, Oudot, Skraba (2013). Persistence-Based Clustering in Riemannian Manifolds. *Journal of the ACM* 60(6). https://doi.org/10.1145/2535927
* L2. d'Errico, Facco, Laio, Rodriguez (2021). Automatic topography of high-dimensional data sets by non-parametric density peak clustering. *Information Sciences* 560, 476–492. https://doi.org/10.1016/j.ins.2021.01.010
* L3. Ding, Shah, Condon (2016). densityCut: an efficient and versatile topological approach for automatic clustering of biological data. *Bioinformatics* 32(17), 2567–2576. https://doi.org/10.1093/bioinformatics/btw227
* L4. Jiang, Jang, Kpotufe (2018). Quickshift++: Provably Good Initializations for Sample-Based Mean Shift. *ICML*, PMLR 80. https://proceedings.mlr.press/v80/jiang18b.html
* L5. Campello, Moulavi, Sander (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *PAKDD*. https://doi.org/10.1007/978-3-642-37456-2_14
* L6. Müller, Sawitzki (1991). Excess Mass Estimates and Tests for Multimodality. *JASA* 86(415), 738–746. https://doi.org/10.1080/01621459.1991.10475103
* L7. Genovese, Perone-Pacifico, Verdinelli, Wasserman (2016). Non-Parametric Inference for Density Modes. *JRSS B* 78(1), 99–126. https://doi.org/10.1111/rssb.12111
* L8. Bobrowski, Skraba (2023). A universal null-distribution for topological data analysis. *Scientific Reports* 13, 12274. https://doi.org/10.1038/s41598-023-37842-2
* L9. Kim, Chen, Balakrishnan, Rinaldo, Wasserman (2016). Statistical Inference for Cluster Trees. *NeurIPS*. https://arxiv.org/abs/1605.06416
* Hartigan, Hartigan (1985). The Dip Test of Unimodality. *Annals of Statistics* 13(1), 70–84. https://doi.org/10.1214/aos/1176346577

**Stability**

* S1. Ben-Hur, Elisseeff, Guyon (2002). A stability based method for discovering structure in clustered data. *Pacific Symposium on Biocomputing*, 6–17. https://doi.org/10.1142/9789812799623_0002
* S2. Lange, Roth, Braun, Buhmann (2004). Stability-Based Validation of Clustering Solutions. *Neural Computation* 16(6), 1299–1323. https://doi.org/10.1162/089976604773717621
* S3. von Luxburg (2010). Clustering Stability: An Overview. *Foundations and Trends in Machine Learning* 2(3), 235–274. https://doi.org/10.1561/2200000008
* S4. Hennig (2007). Cluster-wise assessment of cluster stability. *CSDA* 52(1), 258–271. https://doi.org/10.1016/j.csda.2006.11.025
* S5. Tibshirani, Walther (2005). Cluster Validation by Prediction Strength. *JCGS* 14(3), 511–528. https://doi.org/10.1198/106186005X59243
* S6. Fang, Wang (2012). Selection of the number of clusters via the bootstrap method. *CSDA* 56(3), 468–477. https://doi.org/10.1016/j.csda.2011.09.003
* S7. Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh (2014). Confidence sets for persistence diagrams. *Annals of Statistics* 42(6), 2301–2339. https://doi.org/10.1214/14-AOS1252
* S8. Chazal, Fasy, Lecci, Michel, Rinaldo, Wasserman (2018). Robust Topological Inference: Distance To a Measure and Kernel Distance. *JMLR* 18(159). https://www.jmlr.org/papers/v18/15-484.html
* S9. Huber, Kališnik, Schnider (2025). AuToMATo: An Out-Of-The-Box Persistence-Based Clustering Algorithm. *TMLR*. https://arxiv.org/abs/2408.06958

**Software and precedent**

* Gagolewski (2021). genieclust: Fast and robust hierarchical clustering. *SoftwareX* 15, 100722. https://doi.org/10.1016/j.softx.2021.100722
* Jang, Jiang (2019). DBSCAN++: Towards fast and scalable density clustering. *ICML*, PMLR 97. https://proceedings.mlr.press/v97/jang19a.html
* Scoccola, Rolle (2023). Persistable: persistent and stable clustering. *JOSS* 8(83), 5022. https://doi.org/10.21105/joss.05022
* Package pages: https://cran.r-project.org/package=TDA (likewise dbscan, densityClust, pdfCluster, mclust, fpc, ks, LPCM, clustree); https://bitbucket.org/jerry00/densitycut_dev; https://hdbscan.readthedocs.io; https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html; https://gudhi.inria.fr/python/latest/clustering.html; https://github.com/m-a-huber/automato; https://github.com/CoAxLab/DeBaCl; https://github.com/sissa-data-science/DADApy; https://github.com/TutteInstitute/fast_hdbscan; https://github.com/LuisScoccola/persistable
* SoftwareX guide: https://www.sciencedirect.com/journal/softwarex/publish/guide-for-authors
