library(testthat)
library(TensorMCMC)

set.seed(123)

# example 1
n <- 20
p <- 4
d <- 3
pgamma <- 2

x <- array(rnorm(n*p*d), dim = c(n,p,d))
z <- matrix(rnorm(n*pgamma), n, pgamma)
y <- rnorm(n)

# Test 1: tensor.reg returns correct output
test_that("tensor.reg returns correct structure", {
  fit <- tensor.reg(z, x, y, nsweep = 5, rank = 2)

  expect_s3_class(fit, "tensor.reg")
  expect_true("beta.store" %in% names(fit))
  expect_true("gam.store" %in% names(fit))
  expect_equal(dim(fit$beta.store), c(5, 2, p, d))
  expect_equal(dim(fit$gam.store), c(5, pgamma))
})

# Test 2: predict tensor reg returns numeric vector
test_that("predict_tensor_reg returns numeric vector of correct length", {
  fit <- tensor.reg(z, x, y, nsweep = 5, rank = 2)

  pred <- predict_tensor_reg(fit, x, z)

  expect_type(pred, "double")
  expect_length(pred, n)
})

# Test 3: getmean returns correct output
test_that("getmean returns numeric vector of correct length", {
  beta_list <- replicate(2, list(matrix(rnorm(p*d), p, d)))
  mu <- getmean(x, beta_list, rank = 2)

  expect_type(mu, "double")
  expect_length(mu, n)
})

# Test 4: rmse works correctly
test_that("rmse returns correct value", {
  a <- rnorm(10)
  b <- a + rnorm(10, 0, 0.1)
  val <- rmse(a, b)

  expect_type(val, "double")
  expect_true(val >= 0)
})
