to_symmetric <- function(mat) {
  mat[lower.tri(mat)] <- 0
  lower <- t(mat)
  diag(lower) <- 0
  mat + lower
}

compute_l1_norm <- function(mat) {
  sum(abs(mat))
}

# Eq. 4 (Sec. 3.1)
compute_mean <- function(x, m, A, w) {
  K <- length(m)
  M <- ncol(x)
  lapply(1:K, function(k) {
    tmp <- t(apply(x, 1, function(row) row - m[[k]])) %*% A[[k]]
    do.call(cbind, lapply(1:M, function(i) - tmp[,i] * w[[k]][i] + x[,i]))
  })
}

# Eq. 5 (Sec. 3.1)
compute_variance <- function(A) {
  lapply(A, function(a) 1 / diag(a))
}

# Eq. 17 (Sec. 3.2)
#' @importFrom stats dnorm
compute_gating_function <- function(x, theta, u, w, i) {
  K <- length(u)
  mat <- do.call(cbind, lapply(1:K, function(k) {
    theta[k] * dnorm(x[,i], mean = u[[k]][,i], sd = w[[k]][i])
  }))
  denom <- rowSums(mat)
  g <- apply(mat, 2, function(col) col / denom)
  list(g = g, mat = mat)
}
