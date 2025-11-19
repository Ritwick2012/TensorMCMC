#' Predict Response Using Tensor Regression C++
#'
#' Calls `predict_tensor_cpp` to compute predictions from tensor X, beta, and scalar coefficients.
#'
#' @param X_vec Flattened tensor (numeric vector of length n*p*d)
#' @param beta List of p×d matrices (length = rank)
#' @param gam Numeric vector of scalar coefficients
#' @param n Number of observations
#' @param p Number of rows in each beta matrix
#' @param d Number of columns in each beta matrix
#' @param rank Rank of tensor decomposition
#' @return Numeric vector of length n
#' @export
predict_tensor_cpp <- function(X_vec, beta, gam, n, p, d, rank) {
  .Call(`_TensorMCMC_predict_tensor_cpp`, X_vec, beta, gam, n, p, d, rank)
}
