# Phase A prototype of the uniform subsample density cluster tree in R.
# Same definitions as src/atdc/subsample.py; used to measure what an R
# implementation costs and whether it reproduces the Python numbers
# (benchmarks/14_cross_language.py).  Not the package.
#
# Vertex and edge indices are 1-based.  Total order on vertices: higher
# density first, ties by smaller index.

knn_log_density <- function(r, K, n, d) {
  digamma(K) - log(n) - (0.5 * d * log(pi) - lgamma(0.5 * d + 1)) - d * log(r)
}

unique_edges <- function(src, dst, n) {
  a <- pmin(src, dst); b <- pmax(src, dst)
  keep <- a != b
  key <- unique((a[keep] - 1) * n + (b[keep] - 1))   # exact in doubles for n < 9e7
  key <- sort(key)
  cbind(key %/% n + 1, key %% n + 1)
}

# Merge tree (0-dim superlevel persistence) by Kruskal with the elder rule.
# use_cpp: union-find loop in C++ (Rcpp) instead of R.
merge_tree <- function(f, edges, reduce = TRUE, loop = merge_loop_r) {
  n <- length(f)
  w <- pmin(f[edges[, 1]], f[edges[, 2]])
  eidx <- seq_len(nrow(edges))
  if (reduce && nrow(edges) > n) {
    # ascent forest + best crossing edge per pair of adjacent basins
    u <- c(edges[, 1], edges[, 2]); v <- c(edges[, 2], edges[, 1])
    ei <- c(eidx, eidx)
    ok <- (f[v] > f[u]) | (f[v] == f[u] & v < u)
    u <- u[ok]; v <- v[ok]; ei <- ei[ok]
    o <- order(u, -f[v], v)
    first <- !duplicated(u[o])
    sel <- o[first]
    parent <- seq_len(n); pedge <- integer(n)
    parent[u[sel]] <- v[sel]; pedge[u[sel]] <- ei[sel]
    repeat { nxt <- parent[parent]; if (identical(nxt, parent)) break; parent <- nxt }
    cu <- parent[edges[, 1]]; cv <- parent[edges[, 2]]
    crossing <- which(cu != cv)
    key <- (pmin(cu[crossing], cv[crossing]) - 1) * n + pmax(cu[crossing], cv[crossing])
    o <- order(key, -w[crossing])
    best <- crossing[o[!duplicated(key[o])]]
    eidx <- c(pedge[pedge > 0], best)
  }
  hi <- pmax(f[edges[eidx, 1]], f[edges[eidx, 2]])
  ord <- eidx[order(-w[eidx], -hi)]
  res <- loop(f, edges[ord, 1], edges[ord, 2], w[ord])
  res$n <- n; res$f <- f
  res
}

merge_loop_r <- function(f, eu, ev, w) {
  n <- length(f)
  p <- seq_len(n); peak <- seq_len(n)
  ne <- length(eu)
  dying <- integer(ne); surviving <- integer(ne); level <- numeric(ne)
  mu <- integer(ne); mv <- integer(ne)
  k <- 0L
  for (e in seq_len(ne)) {
    a <- eu[e]; while (p[a] != a) { p[a] <- p[p[a]]; a <- p[a] }
    b <- ev[e]; while (p[b] != b) { p[b] <- p[p[b]]; b <- p[b] }
    if (a == b) next
    pa <- peak[a]; pb <- peak[b]
    if (f[pa] < f[pb] || (f[pa] == f[pb] && pa > pb)) { t <- a; a <- b; b <- t; t <- pa; pa <- pb; pb <- t }
    p[b] <- a
    k <- k + 1L
    dying[k] <- pb; surviving[k] <- pa; level[k] <- w[e]; mu[k] <- eu[e]; mv[k] <- ev[e]
  }
  s <- seq_len(k)
  for (i in seq_len(n)) { a <- i; while (p[a] != a) { p[a] <- p[p[a]]; a <- p[a] }; p[i] <- a }
  list(dying = dying[s], surviving = surviving[s], level = level[s], mu = mu[s], mv = mv[s],
       roots = sort(unique(peak[p])))
}

# ToMATo labelling: replay the spanning merges, refuse when the lower peak has
# prominence >= tau at the merge level.
tree_labels <- function(tree, tau) {
  n <- tree$n; f <- tree$f
  p <- seq_len(n)
  for (e in seq_along(tree$level)) {
    a <- tree$mu[e]; while (p[a] != a) { p[a] <- p[p[a]]; a <- p[a] }
    b <- tree$mv[e]; while (p[b] != b) { p[b] <- p[p[b]]; b <- p[b] }
    if (a == b) next
    if (f[a] < f[b] || (f[a] == f[b] && a > b)) { t <- a; a <- b; b <- t }
    if (f[b] - tree$level[e] < tau) p[b] <- a      # root of a component is its peak
  }
  for (i in seq_len(n)) { a <- i; while (p[a] != a) { p[a] <- p[p[a]]; a <- p[a] }; p[i] <- a }
  p
}

tree_peaks <- function(tree, tau) {
  c(tree$roots, tree$dying[tree$f[tree$dying] - tree$level >= tau])
}

# Exact search by matrix products, base R only (for dimensions where k-d trees
# stop helping).  Work arrays stay below `block` doubles.  Each block is one
# BLAS call on augmented matrices, [x, |x|^2] . [-2q, 1] = |x|^2 - 2 x.q, so
# that no R-level arithmetic touches the n x block array.
#
# Distance from each vertex to its K-th nearest row of X.  The neighbours are
# found from |x|^2 - 2 x.q; their distances are then recomputed directly, as in
# the Python reference.
mat_kth_distance <- function(X, idx, K, block = 8e6) {
  n <- nrow(X); m <- length(idx)
  Xa <- cbind(X, rowSums(X^2))
  step <- max(1L, as.integer(block %/% n)); r <- numeric(m)
  for (s in seq(1L, m, by = step)) {
    cols <- s:min(m, s + step - 1L)
    Q <- X[idx[cols], , drop = FALSE]
    d2 <- tcrossprod(Xa, cbind(-2 * Q, 1))           # n x step; one query per column
    d2[cbind(idx[cols], seq_along(cols))] <- Inf     # the vertex itself
    for (j in seq_along(cols)) {
      v <- d2[, j]
      nb <- which(v <= sort.int(v, partial = K)[K])
      r[cols[j]] <- sqrt(max(rowSums((X[nb, , drop = FALSE] - rep(Q[j, ], each = length(nb)))^2)))
    }
  }
  r
}

# Index of the nearest row of V for every row of X: argmax of x.v - |v|^2 / 2.
mat_nearest <- function(X, V, block = 8e6) {
  n <- nrow(X); m <- nrow(V)
  Va <- cbind(V, -0.5 * rowSums(V^2))
  step <- max(1L, as.integer(block %/% m)); out <- integer(n)
  for (s in seq(1L, n, by = step)) {
    rows <- s:min(n, s + step - 1L)
    out[rows] <- max.col(tcrossprod(cbind(X[rows, , drop = FALSE], 1), Va), ties.method = "first")
  }
  out
}

subsample_tree <- function(X, idx, K = 100, k_graph = 10, nn = c("RANN", "dbscan", "FNN", "matrix"),
                           loop = merge_loop_r) {
  nn <- match.arg(nn)
  n <- nrow(X); d <- ncol(X); m <- length(idx)
  tm <- c()
  t0 <- proc.time()[[3]]
  V <- X[idx, , drop = FALSE]
  r <- switch(nn,
    RANN = RANN::nn2(X, V, k = K + 1)$nn.dists[, K + 1],
    FNN = FNN::get.knnx(X, V, k = K + 1)$nn.dist[, K + 1],
    dbscan = dbscan::kNN(X, k = K + 1, query = V)$dist[, K + 1],
    matrix = mat_kth_distance(X, idx, K))
  tm["query"] <- proc.time()[[3]] - t0
  f <- knn_log_density(r, K, n, d)
  t0 <- proc.time()[[3]]
  k <- min(k_graph, m - 1)
  ind <- RANN::nn2(V, V, k = k + 1)$nn.idx
  edges <- unique_edges(rep(seq_len(m), k), as.vector(ind[, -1]), m)
  tm["graph"] <- proc.time()[[3]] - t0
  t0 <- proc.time()[[3]]
  tree <- merge_tree(f, edges, loop = loop)
  tm["tree"] <- proc.time()[[3]] - t0
  list(f = f, r = r, edges = edges, tree = tree, idx = idx, V = V, times = tm, nn = nn)
}

label_all <- function(fit, X, tau) {
  t0 <- proc.time()[[3]]
  vlab <- tree_labels(fit$tree, tau)
  nnv <- if (identical(fit$nn, "matrix")) mat_nearest(X, fit$V) else RANN::nn2(fit$V, X, k = 1)$nn.idx[, 1]
  fit$times["label"] <- proc.time()[[3]] - t0
  list(labels = fit$idx[vlab][nnv], times = fit$times)   # label = data index of the peak
}
