#' @importFrom glasso glasso
#' @importFrom mvtnorm dmvnorm
#' @importFrom stats cov kmeans
#' @importFrom utils txtProgressBar setTxtProgressBar
sparseGaussMix <- function(x, K, rho, kmeans = FALSE, m0 = rep(0, M),
                           lambda0 = 1, max_iter = 500L, tol = 1e-1,
                           verbose = TRUE) {
  if (verbose) progress_bar <- txtProgressBar(0L, max_iter, style = 3L)
  N <- nrow(x)
  M <- ncol(x)

  pi <- rep(1/K, K)
  lambda <- pi * N

  # In the context of industrial condition-based monitoring, initialization
  # of {mk, Lambda_k} can be naturally done by disjointly partitioning the data
  # along the time axis as D = D1 \cup ... \cup DK and apply e.g. the graphical
  # lasso algorithm [13] on each (Sec. 5)
  if (kmeans) {
    km <- kmeans(x, centers = K)
    split_block <- km$cluster
  } else {
    split_block <- rep(1:K, each = ceiling(N / K))[1:N]
  }
  splitted_data <- split(x, split_block)

  m <- lapply(splitted_data, colMeans)
  Sigma <- lapply(splitted_data, cov)

  glasso_fit <- lapply(1:K, function(k) glasso(Sigma[[k]], rho = 0.1))
  Lambda <- lapply(glasso_fit, function(fit) fit$wi)
  invLambda <- lapply(Lambda, function(Lam) to_symmetric(solve(Lam)))

  compute_loglik <- generate_compute_loglik(x, N, K, rho, m0, lambda0)

  loglik <- -Inf
  n_iter <- 1
  while (TRUE) {
    if (verbose) setTxtProgressBar(progress_bar, n_iter)
    # Eq. 29 (Sec. 4.2)
    ln_r <- lapply(1:K, function(k) {
      log(pi[k]) +
        dmvnorm(x, mean = m[[k]], sigma = invLambda[[k]], log = TRUE) -
        M / (2 * lambda[k])
    })

    # Eq. 30 (Sec. 4.2) using log-sum-exp Algorithm
    max_ln_r <- apply(as.data.frame(ln_r), 1, max)
    rs <- lapply(ln_r, function(x) exp(x - max_ln_r))
    denom <- Reduce("+", rs)
    r <- lapply(rs, function(r) r / denom + 1e-100)

    # Eq. 31 (Sec. 4.2)
    Nk <- vapply(r, sum, double(1))
    pi <- Nk / N

    # Eq. 32 (Sec. 4.2)
    x_bar <- lapply(1:K, function(k) {
      colSums(apply(x, 2, function(col) r[[k]] * col)) / Nk[k]
    })

    # Eq. 33 (Sec. 4.2)
    x_mat <- as.matrix(x)
    Sigma <- lapply(1:K, function(k) {
      tmp <- lapply(1:N, function(i) {
        row <- x_mat[i, ]
        r[[k]][i] * tcrossprod(row - x_bar[[k]])
      })
      tmp <- Reduce("+", tmp)
      tmp / Nk[[k]]
    })

    # Eq. 34 (Sec. 4.2)
    lambda <- lambda0 + Nk
    m <- lapply(1:K, function(k) {
      (lambda0 * m0 + Nk[k] * x_bar[[k]]) / lambda[k]
    })

    # Eq. 35 (Sec. 4.2)
    Q <- lapply(1:K, function(k) {
      Sigma[[k]] + tcrossprod(x_bar[[k]] - m0) * lambda0 / lambda[k]
    })

    # Eq. 36 (Sec. 4.2)
    # Notice that the VB equation for Lambda_k preserves the original
    # L1-regularized GGM formulation [13]. We see that the fewer samples
    # a cluster have, the more the L1 regularization is applied due to the
    # rho/Nk term (Sec. 4.2)
    glasso_fit <- lapply(1:K, function(k) glasso(Q[[k]], rho = rho / Nk[k]))
    Lambda <- lapply(glasso_fit, function(fit) fit$wi)
    invLambda <- lapply(Lambda, function(Lam) to_symmetric(solve(Lam)))

    last_loglik <- loglik
    loglik <- compute_loglik(r, m, invLambda, Lambda, pi)

    loglik_gap <- abs(loglik - last_loglik)
    if (is.finite(loglik) && loglik_gap < tol) break

    n_iter <- n_iter + 1
    if(n_iter > max_iter) {
      message <- sprintf("did not converge after %d iteration: gap: %f",
                         max_iter, loglik_gap)
      warning(message)
      break
    }
  }
  if (verbose) setTxtProgressBar(progress_bar, max_iter)
  A <- lapply(1:K, function(k) (lambda[k] / (1 + lambda[k])) * Lambda[[k]] )
  list(pi = pi, m = m, A = A)
}

#' @importFrom mvtnorm dmvnorm
generate_compute_loglik <- function(x, N, K, rho, m0, lambda0) {
  # Eq. 26 (Sec. 4.1
  function(r, m, invLambda, Lambda, pi) {
    r_df <- as.data.frame(r, col.names = 1:K)
    inds <- apply(r_df, 1, which.max)

    # Eq. 24 (Sec. 4.1)
    tmp <- vapply(1:K, function(k) compute_l1_norm(Lambda[[k]]), double(1))
    loglik1 <- sum(-rho * tmp / 2)

    # Eq. 25 (Sec. 4.1)
    loglik2 <- sum(vapply(1:K, function(k) {
      dmvnorm(m[[k]], mean = m0, sigma = invLambda[[k]] / lambda0, log = TRUE)
    }, double(1)))

    loglik3 <- sum(vapply(inds, function(k) log(pi[k]), double(1)))

    loglik_df <- as.data.frame(lapply(1:K, function(k) {
      dmvnorm(x, mean = m[[k]], sigma = invLambda[[k]], log = TRUE)
    }), col.names = 1:K)
    loglik4 <- sum(mapply(function(row, col) loglik_df[row, col], 1:N, inds))

    loglik1 + loglik2 + loglik3 + loglik4
  }
}
