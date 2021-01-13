#' Sparse Gaussian Markov Random Field Mixtures
#'
#' @param x data.frame. A training data.
#' @param K integer. Number of mixture components. Set a large enough number
#'          because the algorithm identifies major dependency patterns from
#'          the data via the sparse mixture model.
#' @param rho double. Constant that multiplies the penalty term. An optimal
#'          value should be determined together with the threshold on the
#'          anomaly score, so the performance of anomaly detection is maximized.
#' @param kmeans logical. If TRUE, initialize parameters with k-means method.
#'          You should set TRUE for non-time series data. Default FALSE.
#' @param m0 a numeric vector. Location parameter of Gauss-Laplace prior.
#'          Keep default if no prior information is available. Default 0.
#' @param lambda0 double. Coefficient for scale parameter of Gauss-Laplace
#'          prior. Keep default if no prior information is available. Default 1.
#' @param alpha double. Concentration parameter of Dirichlet prior.
#'          Keep default if no prior information is available. Default 1.
#' @param pi_threshold double. Threshold to decide a number of states.
#'          If pi < pi_threshold, the states are rejected in the sense of
#'          sparse estimation.
#' @param max_iter integer. Maximum number of iterations.
#' @param tol double. The tolerance to declare convergence.
#' @param verbose logical.
#'
#' @return sGMRFmix object
#'
#' @examples
#' library(sGMRFmix)
#'
#' set.seed(314)
#' train_data <- generate_train_data()
#' fit <- sGMRFmix(train_data, K = 7, rho = 10)
#' fit
#'
#' @export
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
  if (verbose) message("\n################## GMRFmix ##############################")
  ind <- pi >= pi_threshold
  pi <- pi[ind] / sum(pi[ind])
  names(pi) <- seq_along(pi)
  m <- m[ind]
  A <- A[ind]
  fit <- GMRFmix(x, pi = pi, m = m, A = A, alpha = alpha,
                 max_iter = max_iter, tol = tol, verbose = verbose)
  theta <- fit$theta
  if (verbose) message("\n################## Finished #############################")
  mode <- compute_mode(x, m, A)
  result <- list(x = x, pi = pi, m = m, A = A, theta = theta, mode = mode,
                 Kest = length(pi), K = K, rho = rho, m0 = m0, lambda0 = lambda0,
                 pi_threshold = pi_threshold, colnames = colnames,
                 scaled_center = scaled_center, scaled_scale = scaled_scale)
  class(result) <- "sGMRFmix"

  cl <- match.call()
  result$call <- cl

  result
}

#' @importFrom mvtnorm dmvnorm
compute_mode <- function(x, m, A) {
  K <- length(m)
  mat <- do.call(cbind, lapply(1:K, function(k) {
    sigma <- to_symmetric(solve(A[[k]]))
    dmvnorm(x, mean = m[[k]], sigma = sigma, log = TRUE)
  }))
  unname(apply(mat, 1, which.max))
}
