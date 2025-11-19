#' Update Beta Matrices Using C++ MCMC Step
#'
#' Calls the C++ function `update_beta_cpp` to perturb beta matrices.
#'
#' @param beta List of p×d matrices (one per rank)
#' @param p Number of rows in each beta matrix
#' @param d Number of columns in each beta matrix
#' @param rank Rank of tensor decomposition
#' @param sigma Standard deviation of Gaussian noise
#' @return Updated list of beta matrices
#' @export
update_beta_cpp <- function(beta, p, d, rank, sigma) {
  .Call(`_TensorMCMC_update_beta_cpp`, beta, p, d, rank, sigma)
}

