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


sGMRFmix <- function(x, K, rho, kmeans = FALSE, m0 = rep(0, M), lambda0 = 1,
                     alpha = NULL, pi_threshold = 1/K/100, max_iter = 500,
                     tol = 1e-1, verbose = TRUE) {
  scaled_x <- scale(x)
  scaled_center <- attr(scaled_x, "scaled:center")
  scaled_scale <- attr(scaled_x, "scaled:scale")
  
  x <- data.frame(scaled_x)
  M <- ncol(x)
  colnames <- colnames(x)
  if (verbose) message("################## sparseGaussMix #######################")
  fit <- sparseGaussMix(x, K = K, rho = rho, kmeans = kmeans, m0 = m0,
                        lambda0 = lambda0, max_iter = max_iter,
                        tol = tol, verbose = verbose)
  pi <- fit$pi
  m <- fit$m
  A <- fit$A
  print(pi)
  print(m)
  print(A)
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
    #     print(Nk)
    pi <- Nk / N
    
    #     print(length(r[[1]]))
    
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

