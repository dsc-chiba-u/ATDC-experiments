# densityCut (Ding, Shah & Condon 2016) through its own R package, default
# parameters.  args: <dir> with X.bin (doubles, row-major) and meta.json (n, d).
suppressPackageStartupMessages({library(jsonlite); library(densitycut)})
dir <- commandArgs(trailingOnly = TRUE)[1]
meta <- fromJSON(file.path(dir, "meta.json"))
X <- matrix(readBin(file.path(dir, "X.bin"), "double", meta$n * meta$d), nrow = meta$n, byrow = TRUE)
set.seed(meta$seed)
t0 <- proc.time()[[3]]
res <- DensityCut(X = X, show.plot = FALSE)
wall <- proc.time()[[3]] - t0
writeBin(as.integer(res$cluster), file.path(dir, "labels.bin"))
write_json(list(wall = wall, n_clusters = length(unique(res$cluster)),
                version = as.character(packageVersion("densitycut"))), file.path(dir, "out.json"), auto_unbox = TRUE)
