"""Real data for Phase A, written to data/ (not tracked by git).

Run with the data environment:  ./.conda-env-data/bin/python benchmarks/data_prep.py [text] [pbmc]

text  20 Newsgroups (scikit-learn loader; headers, footers and quotes removed,
      empty documents dropped) embedded with two public sentence-transformers
      models: all-MiniLM-L6-v2 (384-d) and all-mpnet-base-v2 (768-d).
      Embeddings are L2-normalised by the models.
pbmc  68k PBMC, donor A (Zheng et al. 2017, 10x Genomics), with the cell-type
      annotation published with the paper.  Library-size normalisation to
      10 000, log1p, 2 000 highly variable genes, scaling (clip 10), PCA 50.

Every output has a .json next to it with the source, the steps and the package
versions.
"""
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
what = sys.argv[1:] or ["text", "pbmc"]


def meta(path, **kw):
    import sklearn
    kw["numpy"] = np.__version__; kw["sklearn"] = sklearn.__version__
    Path(str(path) + ".json").write_text(json.dumps(kw, indent=1))


if "text" in what:
    import torch
    from sentence_transformers import SentenceTransformer
    import sentence_transformers
    from sklearn.datasets import fetch_20newsgroups
    torch.set_num_threads(24)
    ng = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"),
                            data_home=str(OUT / "sklearn_cache"))
    keep = [i for i, t in enumerate(ng.data) if len(t.strip()) >= 20]
    texts = [ng.data[i] for i in keep]
    y = np.asarray(ng.target)[keep]
    np.save(OUT / "ng20_labels.npy", y)
    (OUT / "ng20_label_names.json").write_text(json.dumps(list(ng.target_names)))
    for model, tag in (("all-MiniLM-L6-v2", "minilm384"), ("all-mpnet-base-v2", "mpnet768")):
        m = SentenceTransformer(model, device="cpu")
        E = m.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        np.save(OUT / f"ng20_{tag}.npy", E.astype(np.float32))
        meta(OUT / f"ng20_{tag}.npy", source="sklearn.datasets.fetch_20newsgroups(subset='all', "
             "remove=('headers','footers','quotes')); documents shorter than 20 characters dropped",
             model=f"sentence-transformers/{model}", max_seq_length=int(m.max_seq_length),
             normalize_embeddings=True, shape=list(E.shape), torch=torch.__version__,
             sentence_transformers=sentence_transformers.__version__)
        print(tag, E.shape, flush=True)

if "pbmc" in what:
    import anndata
    import pandas as pd
    import scanpy as sc
    raw = OUT / "pbmc68k_raw"
    raw.mkdir(exist_ok=True)
    url = ("https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/"
           "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz")
    ann = ("https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/"
           "pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv")
    tgz = raw / "matrices.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(url, tgz)
    if not (raw / "annotation.tsv").exists():
        urllib.request.urlretrieve(ann, raw / "annotation.tsv")
    with tarfile.open(tgz) as t:
        t.extractall(raw, filter="data")
    mdir = next(p.parent for p in raw.rglob("matrix.mtx*"))
    ad = sc.read_10x_mtx(mdir, var_names="gene_symbols")
    a = pd.read_csv(raw / "annotation.tsv", sep="\t").set_index("barcodes")
    ad = ad[ad.obs_names.isin(a.index)].copy()
    ad.obs["celltype"] = a.loc[ad.obs_names, "celltype"].values
    sc.pp.filter_genes(ad, min_cells=3)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    ad = ad[:, ad.var.highly_variable].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=50, svd_solver="arpack", random_state=0)
    np.save(OUT / "pbmc68k_pca50.npy", ad.obsm["X_pca"].astype(np.float32))
    cats = sorted(ad.obs["celltype"].unique())
    np.save(OUT / "pbmc68k_labels.npy", np.array([cats.index(c) for c in ad.obs["celltype"]]))
    (OUT / "pbmc68k_label_names.json").write_text(json.dumps(cats))
    meta(OUT / "pbmc68k_pca50.npy", source=url, annotation=ann, steps="filter_genes(min_cells=3); "
         "normalize_total(1e4); log1p; highly_variable_genes(n_top_genes=2000); scale(max_value=10); "
         "pca(50, arpack, random_state=0)", shape=list(ad.obsm["X_pca"].shape),
         scanpy=__import__("importlib.metadata").metadata.version("scanpy"),
         anndata=__import__("importlib.metadata").metadata.version("anndata"),
         variance_ratio=[float(v) for v in ad.uns["pca"]["variance_ratio"]])
    print("pbmc", ad.obsm["X_pca"].shape, flush=True)
