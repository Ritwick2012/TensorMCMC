#' Posterior Mean Using C++
#'
#' This function calls `getmean_cpp` to compute the posterior mean from tensor X and beta list.
#'
#' @param X_vec Flattened tensor (numeric vector of length n*p*d)
#' @param beta List of p×d matrices (length = rank)
#' @param n Number of observations
#' @param p Number of rows in each beta matrix
#' @param d Number of columns in each beta matrix
#' @param rank Rank of tensor decomposition
#' @return Numeric vector of length n
#' @export
getmean_cpp <- function(X_vec, beta, n, p, d, rank)
  {
  .Call(`_TensorMCMC_getmean_cpp`, X_vec, beta, n, p, d, rank)
  }
