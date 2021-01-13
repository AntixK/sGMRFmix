#' @export
print.sGMRFmix <- function(x, digits = 3L, ...) {
  cat("\nCall:\n")
  print(x$call)

  N <- nrow(x$x)
  M <- ncol(x$x)
  cat("\nData:", N, "x", M, "\n")

  cat("Parameters:\n")
  cat("  K:   ", x$K, "\n")
  cat("  rho: ", x$rho, "\n")

  cat("Estimated:\n")
  cat("  Kest:", x$Kest, "\n")
  cat("  pi:  ", format(x$pi, digits = digits), "\n")
  cat("  m, A, theta, H, mode\n")
}
