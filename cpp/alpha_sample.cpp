// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loglik.h"
using namespace arma;
using namespace Rcpp;

// SAMPLING ALPHA
NumericVector alpha_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double a0, double A0, double MH_alpha) {
  int J = Y.n_cols;
  arma::vec alpha_star(J);
  arma::vec log_prop_star(J);
  arma::vec log_prop_old(J);
  
  for (int j = 0; j < J; j++) {
    alpha_star[j] = R::rnorm(alpha_old[j], MH_alpha);
    log_prop_star[j] = R::dnorm(alpha_star[j], alpha_old[j], MH_alpha, true);
    log_prop_old[j] = R::dnorm(alpha_old[j], alpha_star[j], MH_alpha, true);
  }
  
  arma::rowvec loglik_star = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_star, beta_old, theta_old))), true);
  arma::rowvec loglik_old = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_old))), true);
  arma::rowvec log_dnorm_star =  dnorm(as<NumericVector>(wrap(alpha_star)), a0, A0, true);
  arma::rowvec log_dnorm_old = dnorm(as<NumericVector>(wrap(alpha_old)), a0, A0, true);
  
  arma::rowvec log_cc_star = loglik_star + log_dnorm_star;
  arma::rowvec log_cc_old = loglik_old + log_dnorm_old;
  arma::vec log_post_dens = log_cc_star.t() + log_prop_old - log_cc_old.t() - log_prop_star;
  NumericVector log_ap = pmin(as<NumericVector>(wrap(log_post_dens)), 0);
  
  NumericVector log_u = log(runif(J, 0, 1));
  NumericVector sample = ifelse(log_u < log_ap, as<NumericVector>(wrap(alpha_star)), as<NumericVector>(wrap(alpha_old)));
  
  return(sample);
}