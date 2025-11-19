#' Tensor Regression using Rcpp
#'
#' Bayesian-like tensor regression with MCMC-style updates.
#'
#' @param z.train Matrix of scalar covariates (n x pgamma)
#' @param x.train 3D array of tensor predictors (n x p x d)
#' @param y.train Response vector (length n)
#' @param nsweep Number of MCMC sweeps (default 50)
#' @param rank Rank of tensor decomposition (default 2)
#' @param scale Logical; whether to scale predictors and response (default TRUE)
#' @param alpha.lasso LASSO tuning parameter for initial estimate (default 1)
#'
#' @return A list with beta.store, gam.store, rank, p, d, and scaling info
#' @export
#' @useDynLib TensorMCMC, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom stats rnorm sd
tensor.reg <- function(z.train, x.train, y.train, nsweep=50, rank=2, scale=TRUE, alpha.lasso=1) {

  n <- length(y.train)
  p <- dim(x.train)[2]
  d <- dim(x.train)[3]
  pgamma <- ncol(z.train)

  if(length(unique(y.train)) == 1) y.train <- y.train + rnorm(length(y.train), 0, 1e-6)

  # Standardize scalar predictors
  Zt <- z.train
  my <- mean(y.train)
  sy <- ifelse(scale, stats::sd(y.train), 1)
  obs <- as.numeric(scale(y.train, center=my, scale=sy))

  # Standardize tensor predictors
  Xt <- x.train
  if(scale){
    for(j in 1:p){
      for(k in 1:d){
        colx <- Xt[,j,k]
        rng <- diff(range(colx))
        if(rng > 0) {
          Xt[,j,k] <- (colx - mean(colx)) / rng
        }
      }
    }
  }

  # Vectorize tensor for C++ usage
  vecXt <- as.numeric(aperm(Xt, c(1,2,3)))

  # Initial LASSO estimate
  vecX_lasso <- cbind(t(apply(Xt, 1, function(xx) as.vector(xx))), Zt)
  las <- glmnet::cv.glmnet(vecX_lasso, y.train, alpha=alpha.lasso, nfolds=min(5,n))
  las <- glmnet::glmnet(vecX_lasso, y.train, lambda=las$lambda.min, alpha=alpha.lasso)
  beta.init <- as.numeric(las$beta)
  gam <- beta.init[1:pgamma]

  # Initialize tensor coefficients
  beta <- replicate(rank, list(matrix(stats::rnorm(p*d), p, d)))

  # Storage
  beta.store <- array(NA, dim=c(nsweep, rank, p, d))
  gam.store <- array(NA, dim=c(nsweep, pgamma))

  # MCMC updates
  for(sweep in 1:nsweep){
    beta <- update_beta_cpp(beta, p, d, rank, sigma=0.05) # C++ function
    gam <- gam + stats::rnorm(pgamma, 0, 0.01)

    beta.store[sweep,, ,] <- array(unlist(beta), dim=c(rank, p, d))
    gam.store[sweep,] <- gam
  }

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
predict.tensor.reg <- function(object, x.new, z.new, scale=TRUE, ...) {

  fit <- object

  n <- dim(x.new)[1]
  p <- dim(x.new)[2]
  d <- dim(x.new)[3]

  Xt <- x.new
  if(scale){
    for(j in 1:p){
      for(k in 1:d){
        colx <- Xt[,j,k]
        rng <- diff(range(colx))
        if(rng > 0) Xt[,j,k] <- (colx - mean(colx)) / rng
      }
    }
  }

  vecXt <- as.numeric(aperm(Xt, c(1,2,3)))
  beta.mean <- apply(fit$beta.store, c(2,3,4), mean)


  if(fit$rank == 1) {
    beta_list <- list(matrix(beta.mean[1,,], nrow=p, ncol=d))
  } else {
    beta_list <- lapply(1:fit$rank, function(r) matrix(beta.mean[r,,], nrow=p, ncol=d))
  }

  gam.mean <- colMeans(fit$gam.store)

  # C++ prediction
  pred <- predict_tensor_cpp(vecXt, beta_list, gam.mean, n, p, d, fit$rank)

  # Add scalar covariates contribution
  pred <- pred + rowSums(z.new * matrix(gam.mean, nrow=n, ncol=length(gam.mean), byrow=TRUE))

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
predict_tensor_reg <- function(fit, x.new, z.new, scale=TRUE) {
  predict(fit, x.new, z.new, scale=scale)
}

#' root-mean-square error (RMSE)
#'
#' @param a Predicted values.
#' @param b True observed values.
#' @return RMSE value.
#' @export
rmse <- function(a, b) sqrt(mean((a-b)^2))

#' Inverse-gamma random number generator
#'
#' @param n Number of samples.
#' @param shape Shape parameter of the gamma distribution.
#' @param rate Rate parameter of the gamma distribution.
#' @return A numeric vector of inverse-gamma samples.
#' @export
rigamma <- function(n, shape, rate) 1 / stats::rgamma(n, shape, rate)

#' posterior mean for tensor regression
#'
#' @param X Input tensor.
#' @param beta Estimated coefficient tensor.
#' @param rank Rank used in tensor decomposition.
#' @param rank.exclude Optional rank values to exclude.
#' @return Mean tensor.
#' @export
getmean <- function(X, beta, rank, rank.exclude=NULL) {
  idx <- setdiff(1:rank, rank.exclude)
  mu.B <- numeric(dim(X)[1])
  for(i in 1:dim(X)[1]){
    Bsum <- 0
    for(r in idx){
      B <- as.vector(beta[[r]])
      Xvec <- as.vector(X[i,,])
      Bsum <- Bsum + sum(Xvec * B)
    }
    mu.B[i] <- Bsum
  }
  return(mu.B)
}

#' Cross-validation for tensor regression
#'
#' @param x.train Training tensor X data.
#' @param z.train Training scalar covariates.
#' @param y.train Training response vector.
#' @param ranks A vector of rank values to test.
#' @param nsweep Number of MCMC sweeps / iterations.
#' @return A data.frame containing the best rank and RMSE performance.
#' @export
cv.tensor.reg <- function(x.train, z.train, y.train, ranks=1:3, nsweep=50){
  out <- numeric(length(ranks))
  for(i in seq_along(ranks)){
    fit <- tensor.reg(z.train, x.train, y.train, nsweep=nsweep, rank=ranks[i])
    pred <- predict.tensor.reg(fit, x.train, z.train)
    out[i] <- rmse(y.train, pred)
  }
  data.frame(rank=ranks, RMSE=out)
}


