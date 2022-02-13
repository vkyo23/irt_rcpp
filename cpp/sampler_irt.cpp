// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h> 
#include "alpha_sample.h"
#include "beta_sample.h"
#include "theta_sample.h"
using namespace Rcpp;
using namespace arma;

// SAMPLER
// [[Rcpp::export]]
List sampler_irt(arma::mat datamatrix, arma::vec alpha, arma::vec beta,
                 arma::vec theta, double a0, double A0, 
                 double b0, double B0,
                 double MH_alpha, double MH_beta, double MH_theta,
                 int iter, int warmup, int thin, int refresh) {
  
  int total_iter = iter + warmup; // total iteration
  int sample_iter = iter / thin; // # of samples to save
  
  arma::mat Y = datamatrix; // rename datamatrix to Y
  int I = Y.n_rows; // # of individuals
  int J = Y.n_cols; // # of items
  
  // rename
  arma::vec alpha_old = alpha;
  arma::vec beta_old = beta;
  arma::vec theta_old = theta;
  
  // create storages for parameters
  NumericMatrix theta_store(I, sample_iter);
  NumericMatrix alpha_store(J, sample_iter);
  NumericMatrix beta_store(J, sample_iter);
  
  
  // WARMUP
  Rcout << "Warmup:   " << 1 << " / " << total_iter << " [ " << 0 << "% ]\n";
  for (int g = 0; g < warmup; g++) {
    if ((g + 1) % refresh == 0) {
      double gg = g + 1;
      double ti2 = total_iter;
      double per = std::round((gg / ti2) * 100);
      Rcout << "Warmup:   " << (g + 1) << " / " << total_iter << " [ " << per << "% ]\n";
    }
    theta = theta_sample(Y, alpha_old, beta_old, theta_old, MH_theta);
    theta_old = theta;
    alpha = alpha_sample(Y, alpha_old, beta_old, theta_old, a0, A0, MH_alpha);
    alpha_old = alpha;
    beta = beta_sample(Y, alpha_old, beta_old, theta_old, b0, B0, MH_beta);
    beta_old = beta;
  }
  
  // SAMPLING
  double gg = warmup + 1;
  double ti2 = total_iter;
  double per = std::round((gg / ti2) * 100);
  Rcout << "Sampling: " << gg << " / " << total_iter << " [ " << per << "% ]\n";
  for (int g = warmup; g < total_iter; g++) {
    if ((g + 1) % refresh == 0) {
      double gg = g + 1;
      double ti2 = total_iter;
      double per = std::round((gg / ti2) * 100);
      Rcout << "Sampling: " << (g + 1) << " / " << total_iter << " [ " << per << " % ]\n";
    }
    theta = theta_sample(Y, alpha_old, beta_old, theta_old, MH_theta);
    theta_old = theta;
    alpha = alpha_sample(Y, alpha_old, beta_old, theta_old, a0, A0, MH_alpha);
    alpha_old = alpha;
    beta = beta_sample(Y, alpha_old, beta_old, theta_old, b0, B0, MH_beta);
    beta_old = beta;
    
    if (g % thin == 0) {
      double th = thin;
      double wu = warmup;
      double gg = g;
      double ggg = (g - warmup) / thin;
      
      // Reparameterization (fix theta with mean 0 and sd 1)
      
      NumericVector theta_nv = as<NumericVector>(wrap(theta_old));
      NumericVector alpha_nv = as<NumericVector>(wrap(alpha_old));
      NumericVector beta_nv = as<NumericVector>(wrap(beta_old));
      
      NumericVector theta_std = (theta_nv - mean(theta_nv)) / sd(theta_nv);
      NumericVector alpha_std = beta_nv * mean(theta_nv) - alpha_nv;
      NumericVector beta_std = beta_nv * sd(theta_nv);
      
      theta_store(_, ggg) = theta_std;
      alpha_store(_, ggg) = alpha_std;
      beta_store(_, ggg) = beta_std;
    }
  }
  
  List L = List::create(Named("alpha") = alpha_store,
                        Named("beta") = beta_store,
                        Named("theta") = theta_store);
  return(L);
}  
