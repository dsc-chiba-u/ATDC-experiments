# Driver for benchmarks/14_cross_language.py.
# args: <dir>  (contains X.bin [doubles, row-major], meta.json with n, d, K, k_graph, tau, idx (0-based))
suppressPackageStartupMessages({library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
here <- dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(here, "atdc_proto.R"))
dir <- args[1]
meta <- fromJSON(file.path(dir, "meta.json"))
X <- matrix(readBin(file.path(dir, "X.bin"), "double", meta$n * meta$d), nrow = meta$n, byrow = TRUE)
idx <- meta$idx + 1L
loops <- list(r = merge_loop_r)
if (isTRUE(meta$cpp)) { Rcpp::sourceCpp(file.path(here, "merge_loop.cpp")); loops$cpp <- merge_loop_cpp }
out <- list()
for (nn in meta$nn) for (lp in names(loops)) {
  best <- NULL
  for (rep in seq_len(meta$`repeat`)) {
    fit <- subsample_tree(X, idx, meta$K, meta$k_graph, nn = nn, loop = loops[[lp]])
    lab <- label_all(fit, X, meta$tau)
    tot <- sum(lab$times)
    if (is.null(best) || tot < best$total) best <- list(total = tot, times = as.list(lab$times), fit = fit, lab = lab)
  }
  key <- paste(nn, lp, sep = "_")
  out[[key]] <- list(times = best$times, total = best$total)
  if (nn == meta$nn[1] && lp == "r") {
    out$f <- best$fit$f
    out$edges <- best$fit$edges - 1L
    out$peaks <- sort(idx[tree_peaks(best$fit$tree, meta$tau)] - 1L)
    out$n_merges <- length(best$fit$tree$level)
    out$persistence <- sort(best$fit$tree$f[best$fit$tree$dying] - best$fit$tree$level, decreasing = TRUE)[1:20]
    writeBin(as.integer(best$lab$labels - 1L), file.path(dir, "labels_r.bin"))
  }
}
write_json(out, file.path(dir, "r_out.json"), digits = NA, auto_unbox = TRUE)
