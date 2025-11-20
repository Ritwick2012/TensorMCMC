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
