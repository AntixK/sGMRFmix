get_mean <- function(type) {
  switch(type,
         "1" = c(-1,  1,  1, -1,  0),
         "2" = c( 1, -1, -1,  1,  0),
         "3" = c( 0,  0,  0,  0,  0)
  )
}

get_sigma <- function(type) {
  switch(type,
         "1" = matrix(c(  1, 0.8,    0,    0,   0,
                        0.8,   1,    0,    0,   0,
                          0,   0,    1, -0.8,   0,
                          0,   0, -0.8,    1,   0,
                          0,   0,    0,    0, 0.5), nrow = 5, byrow = TRUE),
         "2" = matrix(c(  1,   0,   0, 0.8,   0,
                          0,   1, 0.8,   0,   0,
                          0, 0.8,   1,   0,   0,
                        0.8,   0,   0,   1,   0,
                          0,   0,   0,   0, 0.5), nrow = 5, byrow = TRUE),
         "3" = matrix(c(  1, 0.8,   0,   0,   0,
                        0.8,   1,   0,   0,   0,
                          0,   0,   1,   0,   0,
                          0,   0,   0,   1,   0,
                          0,   0,   0,   0, 0.5), nrow = 5, byrow = TRUE)
  )
}

#' Generate train data
#'
#' @importFrom mvtnorm rmvnorm
#' @export
generate_train_data <- function() {
  train_data <- rbind(
    rmvnorm(250, get_mean(1), get_sigma(1)),
    rmvnorm(250, get_mean(2), get_sigma(2)),
    rmvnorm(250, get_mean(1), get_sigma(1)),
    rmvnorm(250, get_mean(2), get_sigma(2))
  )
  train_data <- data.frame(train_data)
  colnames(train_data) <- paste0("x", 1:5)

  train_data
}

#' Generate test data
#'
#' @importFrom mvtnorm rmvnorm
#' @export
generate_test_data <- function() {
  test_data <- rbind(
    rmvnorm(250, get_mean(1), get_sigma(1)),
    rmvnorm(250, get_mean(2), get_sigma(2)),
    rmvnorm(500, get_mean(3), get_sigma(3))
  )
  test_data <- data.frame(test_data)
  colnames(test_data) <- paste0("x", 1:5)

  test_data
}

#' Generate test labels
#'
#' @export
generate_test_labels <- function() {
  test_labels <- matrix(c(
    rep(FALSE, 500), rep(TRUE, 500),
    rep(FALSE, 500), rep(TRUE, 500),
    rep(FALSE, 500), rep(TRUE, 500),
    rep(FALSE, 500), rep(TRUE, 500),
    rep(FALSE, 1000)
  ), ncol = 5)
  test_labels <- data.frame(test_labels)
  colnames(test_labels) <- paste0("x", 1:5)

  test_labels
}
