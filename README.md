# TensorMCMC

<!-- badges: start -->
<!-- badges: end -->

The goal of TensorMCMC is to implement Bayesian style tensor regression for tensor predictors and 
    scalar covariates with MCMC updates, C++ acceleration, prediction, and cross-validation utilities.

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


```
I have already completed the C++ (Rcpp) implementation and the testing of the package. My next step is to create the package vignette.
