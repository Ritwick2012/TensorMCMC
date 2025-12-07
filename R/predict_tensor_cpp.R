#' Predict Response Using Tensor Regression C++
#'
#' This function calls the underlying C++ function `predict_tensor_cpp`
#' to compute predicted responses given a flattened tensor, low-rank
#' coefficient matrices, and scalar covariate coefficients.
#'
#' @param X_vec Flattened tensor (numeric vector of length n*p*d)
#' @param beta List of p×d matrices representing tensor coefficients
#' @param gam Numeric vector of scalar coefficients
#' @param n Number of observations
#' @param p Number of rows in each beta matrix
#' @param d Number of columns in each beta matrix
#' @param rank Rank of tensor decomposition
#' @return Numeric vector of length n
#' @export
predict_tensor_cpp <- function(X_vec, beta, gam, n, p, d, rank)
  {
  .Call(`_TensorMCMC_predict_tensor_cpp`, X_vec, beta, gam, n, p, d, rank)
  }

