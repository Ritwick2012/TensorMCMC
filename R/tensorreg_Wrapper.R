#' Tensor Regression using Rcpp
#'
#' Low-rank tensor regression with stochastic updates
#'
#' @param z.train Matrix of scalar covariates (n x pgamma)
#' @param x.train 3D array of tensor predictors (n x p x d)
#' @param y.train Response vector (length n)
#' @param nsweep Number of stochastic update iterations (default 50)
#' @param rank Rank of tensor decomposition (default 2)
#' @param scale whether to scale predictors and response (default TRUE)
#' @param alpha.lasso LASSO tuning parameter for initial estimate (default 1)
#'
#' @return A list with beta.store, gam.store, rank, p, d, and scaling info
#' @export
#' @useDynLib TensorMCMC, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats rnorm sd
tensor.reg <- function ( z.train, x.train, y.train, nsweep = 50, rank = 2, scale = TRUE, alpha.lasso = 1)
  {

## Coefficient tensor is represented via a low-rank expansion
## Iterative updates use simple random-walk perturbations

  n <- length(y.train)
  p <- dim(x.train)[2]
  d <- dim(x.train)[3]
  pgamma <- ncol(z.train)

  if(length(unique(y.train)) == 1)
  y.train <- y.train + rnorm(length(y.train), 0, 1e-3)

  # Standardize scalar covariates and response
  Zt <- z.train
  my <- mean(y.train)
  sy <- if(scale) sd(y.train) else 1
  obs <- (y.train - my) / sy

  # Standardize tensor predictors
  Xt <- x.train
  if(scale)
    {
    for(j in 1:p)
      {
      for(k in 1:d)
        {
        colx <- Xt[,j,k]
        range_val <- diff(range(colx))
        if(range_val > 0)
          {
          Xt[,j,k] <- (colx - mean(colx)) / range_val
          }
        }
      }
     }

  # Vectorize tensor for C++ usage
  vecXt <- as.numeric(aperm(Xt, c(1,2,3)))

  # Initial LASSO estimate (used for rough initialization only)
  vecX_lasso <- cbind(t(apply(Xt, 1, function(xx) as.vector(xx))), Zt)
  las <- glmnet::cv.glmnet(vecX_lasso, y.train, alpha = alpha.lasso, nfolds = min(5,n))
  las <- glmnet::glmnet(vecX_lasso, y.train, lambda = las$lambda.min, alpha = alpha.lasso)
  beta.init <- as.numeric(las$beta)
  gam <- beta.init[1:pgamma]

  # Initialize tensor coefficients
  beta <- replicate(rank, list(matrix(stats::rnorm(p*d), p, d)))

  # Storage
  beta.store <- array(NA, dim = c(nsweep, rank, p, d))
  gam.store <- array(NA, dim = c(nsweep, pgamma))

  # Iterative stochastic updates (random-walk)
  for(sweep in 1:nsweep)
    {

## Simple random walk update for tensor coefficients.
## Chosen mainly for numerical stability rather than exact inference.

    beta <- update_beta_cpp(beta, p, d, rank, sigma = 0.05) # C++ function

## Scalar coefficients updated separately

    gam <- gam + stats::rnorm(pgamma, 0, 0.01)

    beta.store[sweep,, ,] <- array(unlist(beta), dim = c(rank, p, d))
    gam.store[sweep,] <- gam
  }

## mx and sx store training scale summary statistics
## These are retained for possible future use but are not currently applied during prediction

  out <- list(
    beta.store = beta.store,
    gam.store = gam.store,
    rank = rank, p = p, d = d,
    my = my, sy = sy,
    mx = apply(Xt, 2, mean),
    sx = apply(Xt, 2, stats::sd)
  )

  class(out) <- "tensor.reg"
  return(out)
}

#' Prediction from tensor regression (S3 method)
#'
#' @param object tensor.reg object
#' @param x.new new tensor predictors (n x p x d)
#' @param z.new new scalar covariates (n x pgamma)
#' @param scale Logical; whether to scale predictors
#' @param ... additional arguments
#' @return Predicted response vector
#' @export
#' @method predict tensor.reg
predict.tensor.reg <- function(object, x.new, z.new, scale = TRUE, ...)
  {

  fit <- object

  n <- dim(x.new)[1]  # number of observations
  p <- dim(x.new)[2]  # tensor rows
  d <- dim(x.new)[3]  # tensor columns

  Xt <- x.new
  if(scale)
 {
    for(j in 1:p)
 {
    for(k in 1:d)
 {
  colx <- Xt[,j,k]
  range_val <- diff(range(colx))
  if(range_val > 0)
  {
    Xt[,j,k] <- (colx - mean(colx)) / range_val
  }
 }
 }
 }

  # Flatten the tensor into a vector for C++ computation
  vecXt <- as.numeric(aperm(Xt, c(1,2,3)))

  # Compute mean of the stored beta coefficient matrices
  beta.mean <- apply(fit$beta.store, c(2,3,4), mean)

  # Convert mean beta to list format expected by C++ function
  if(fit$rank == 1)
  {
    beta_list <- list(matrix(beta.mean[1,,], nrow = p, ncol = d))
  }
  else
  {
    beta_list <- lapply(1:fit$rank, function(r) matrix(beta.mean[r,,], nrow = p, ncol = d))
  }

  gam.mean <- colMeans(fit$gam.store)

  # C++ prediction
  pred <- predict_tensor_cpp(vecXt, beta_list, gam.mean, n, p, d, fit$rank)

  # Add scalar covariates contribution
  pred <- pred + rowSums(z.new * matrix(gam.mean, nrow = n, ncol = length(gam.mean), byrow = TRUE))

  return(pred)
 }

#' Predict tensor regression (wrapper)
#'
#' @param fit tensor.reg object
#' @param x.new new tensor predictors
#' @param z.new new scalar covariates
#' @param scale Logical; whether to scale predictors
#' @return Predicted response vector
#' @export
#' @importFrom stats predict
predict_tensor_reg <- function(fit, x.new, z.new, scale = TRUE)
 {
  predict(fit, x.new, z.new, scale = scale)
 }

#' root-mean-square error (RMSE)
#'
#' @param a Predicted values.
#' @param b True observed values.
#' @return RMSE value.
#' @export
rmse <- function(a, b)
 {
  sqrt(mean((a-b)^2))
 }

#' Inverse-gamma random number generator
#'
#' @param n Number of samples.
#' @param shape Shape parameter of the gamma distribution.
#' @param rate Rate parameter of the gamma distribution.
#' @return A numeric vector of inverse-gamma samples.
#' @export
rigamma <- function(n, shape, rate)
 {
   1 / stats::rgamma(n, shape, rate)
  }

#' posterior mean for tensor regression
#'
#' @param X Input tensor.
#' @param beta Estimated coefficient tensor.
#' @param rank Rank used in tensor decomposition.
#' @param rank.exclude Optional rank values to exclude.
#' @return Mean tensor.
#' @export
getmean <- function(X, beta, rank, rank.exclude = NULL)
 {
  idx <- setdiff(1:rank, rank.exclude)
  mu.B <- numeric(dim(X)[1])
  for(i in 1:dim(X)[1])
  {
    Bsum <- 0
    for(r in idx)
  {
      B <- as.vector(beta[[r]])
      Xvec <- as.vector(X[i,,])
      Bsum <- Bsum + sum(Xvec * B)
  }
    mu.B[i] <- Bsum
  }
  return(mu.B)
 }

#' Simple rank comparison via in-sample RMSE
#'
#' @param x.train Training tensor X data.
#' @param z.train Training scalar covariates.
#' @param y.train Training response vector.
#' @param ranks Vector of rank values to test.
#' @param nsweep Number of stochastic update iterations.
#' @return A data.frame with ranks and corresponding in-sample RMSE.
#' @export
cv.tensor.reg <- function(x.train, z.train, y.train, ranks = 1:3, nsweep = 50)
  {
  out <- numeric(length(ranks))
  for(i in seq_along(ranks)){
    fit <- tensor.reg(z.train, x.train, y.train, nsweep = nsweep, rank = ranks[i])
    pred <- predict.tensor.reg(fit, x.train, z.train)
    out[i] <- rmse(y.train, pred)
  }
  data.frame(rank = ranks, RMSE = out)
}
