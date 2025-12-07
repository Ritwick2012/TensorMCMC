# TensorMCMC

<!-- badges: start -->
<!-- badges: end -->

**TensorMCMC** implements low-rank tensor regression for tensor predictors and scalar covariates using
simple stochastic updates. It includes fast C++ routines for coefficient updates and prediction, and 
provides tools for cross-validation and error evaluation.

## Installation

You can install the development version of TensorMCMC like so:

``` r
# FILL THIS IN! HOW CAN PEOPLE INSTALL YOUR DEV PACKAGE?

install.packages("devtools") 
devtools::install_github("Ritwick2012/TensorMCMC")

```
## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(TensorMCMC)
## basic example code

x.train <- array(rnorm(n*p*d), dim = c(n, p, d))
z.train <- matrix(rnorm(n*pgamma), n, pgamma)
y.train <- rnorm(n)

## Fit the tensor regression model
fit <- fit_tensor(x.train, z.train, y.train, rank = 2, nsweep = 50)

# Predict on training data
pred <- predict_tensor_reg(fit, x.train, z.train)

# Calculating RMSE
rmse_val <- rmse(pred, y.train)

```
