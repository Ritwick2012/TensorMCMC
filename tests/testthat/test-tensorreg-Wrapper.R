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

# Test 5: rigamma returns numeric vector
test_that("rigamma returns numeric vector of correct length", {
  val <- rigamma(5, shape = 2, rate = 1)

  expect_type(val, "double")
  expect_length(val, 5)
  expect_true(all(val > 0))
})

# Test 6: cv.tensor.reg returns data frame with RMSE
test_that("cv.tensor.reg returns data frame of correct structure", {
  cvres <- cv.tensor.reg(x, z, y, ranks = 1:2, nsweep = 3)

  expect_s3_class(cvres, "data.frame")
  expect_equal(names(cvres), c("rank", "RMSE"))
  expect_equal(nrow(cvres), 2)
  expect_true(all(cvres$RMSE >= 0))
})

# Test 7: tensor.reg works with rank 1
test_that("tensor.reg works with rank 1", {
  fit <- tensor.reg(z, x, y, nsweep = 4, rank = 1)

  expect_s3_class(fit, "tensor.reg")
  expect_equal(dim(fit$beta.store), c(4, 1, p, d))
})

# Test 8: tensor.reg works without scaling
test_that("tensor.reg works without scaling", {
  fit <- tensor.reg(z, x, y, nsweep = 3, rank = 2, scale = FALSE)

  expect_s3_class(fit, "tensor.reg")
  expect_equal(dim(fit$beta.store), c(3, 2, p, d))
})

# Test 9: predict.tensor.reg returns reasonable values
test_that("predict.tensor.reg gives reasonable values", {
  fit <- tensor.reg(z, x, y, nsweep = 5, rank = 2)
  pred <- predict.tensor.reg(fit, x, z)

  expect_type(pred, "double")
  expect_length(pred, n)
  expect_false(any(is.na(pred)))
})
