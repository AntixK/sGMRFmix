#' @importFrom utils txtProgressBar setTxtProgressBar
GMRFmix <- function(x, pi, m, A, alpha = NULL, max_iter = 500,
                    tol = 1e-1, verbose = TRUE) {
  N <- nrow(x)
  M <- ncol(x)

  if (verbose) progress_bar <- txtProgressBar(0, M, style=3)
  K <- length(pi)

  if (is.null(alpha)) alpha <- rep(1, K)
  if (length(alpha) != K) stop("length of alpha must equal Kest")

  w <- compute_variance(A)
  u <- compute_mean(x, m, A, w)

  theta_mat <- matrix(nrow = M, ncol = K)
  for (i in 1:M) {
    Nk <- pi * N
    loglik <- -Inf
    n_iter <- 1
    while (TRUE) {
      # Eq. 15 (Sec. 3.2)
      a <- alpha + Nk
      a_bar <- sum(a)

      # Eq. 16 (Sec. 3.2)
      theta <- exp(digamma(a) - digamma(a_bar))

      gating <- compute_gating_function(x, theta, u, w, i)
      g <- gating$g
      mat <- gating$mat

      # Eq. 18 (Sec. 3.2)
      Nk <- colSums(g)

      last_loglik <- loglik
      # Eq. 10 (Sec. 3.2)
      alpha_bar <- sum(alpha)
      loglik <- sum(log(apply(mat, 1, function(row) max(row)))) -
        lgamma(alpha_bar) +
        sum(lgamma(alpha) + (alpha - 1) * log(theta))

      loglik_gap <- abs(last_loglik - loglik)
      if (loglik_gap < tol) {
        theta_mat[i, ] <- theta
        break
      }

      n_iter <- n_iter + 1
      if (n_iter > max_iter) {
        message <- sprintf("did not converge after %d iteration: gap: %f",
                           max_iter, loglik_gap)
        warning(message)
        theta_mat[i, ] <- theta
        break
      }
    }
    if (verbose) setTxtProgressBar(progress_bar, i)
  }
  list(theta = theta_mat)
}
