// tensor_reg.cpp
#include <Rcpp.h>
using namespace Rcpp;

// posterior mean vector from tensor X and low-rank beta matrices
// [[Rcpp::export]]
 NumericVector getmean_cpp(const NumericVector& X_vec, const List& beta,
                           int n, int p, int d, int rank)
   {
   NumericVector mu(n, 0.0); // initialize mean vector to zero

   for(int i = 0; i < n; i++)
   {
    double Bsum = 0.0;
    for(int r = 0; r < rank; r++)
    {
    NumericMatrix Br = beta[r];
    for(int j = 0; j < p; j++)
    {
    for(int k = 0; k < d; k++)
    {
    int idx = i * p * d + j * d + k; // vectorized index
    Bsum += X_vec[idx] * Br(j,k);
    }
    }
    }
     mu[i] = Bsum;
   }

   return mu;
 }

// Update beta matrices with small Gaussian random walk noise
// [[Rcpp::export]]
 List update_beta_cpp(List beta, int p, int d, int rank, double sigma)
   {
   for(int r = 0; r < rank; r++)
    {
     NumericMatrix Br = beta[r];
     for(int j = 0; j < p; j++)
    {
      for(int k = 0; k < d; k++)
    {
         Br(j,k) += R::rnorm(0.0, sigma);  //Gaussian noise
    }
    }
     beta[r] = Br;
   }

   return beta;
 }


//  Predict response vector using tensor X and beta coefficients + scalar gam
// [[Rcpp::export]]
 NumericVector predict_tensor_cpp(const NumericVector& X_vec, const List& beta,
                                  const NumericVector& gam,
                                  int n, int p, int d, int rank)
   {

   NumericVector pred(n, 0.0); // initialize predicted vector

   for(int i = 0; i < n; i++)
    {
     double val = 0.0;

     // Tensor contribution (low-rank expansion)
     for(int r = 0; r < rank; r++)
    {
       NumericMatrix Br = beta[r];
       for(int j = 0; j < p; j++)
    {
        for(int k = 0; k < d; k++)
        {
           int idx = i * p * d + j * d + k;
           val += X_vec[idx] * Br(j,k);
         }
       }
     }

     // Scalar covariates contribution (gam)
     for(int g = 0; g < gam.size(); g++)
    {
       val += gam[g];
     }

     pred[i] = val;
   }

   return pred;
 }
