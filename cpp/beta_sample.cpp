// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loglik.h"
using namespace arma;
using namespace Rcpp;

// SAMPLING BETA
NumericVector beta_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double b0, double B0, double MH_beta) {
  int J = Y.n_cols;
  arma::vec beta_star(J);
  arma::vec log_prop_star(J);
  arma::vec log_prop_old(J);
  
  for (int j = 0; j < J; j++) {
    beta_star[j] = R::rnorm(beta_old[j], MH_beta);
    log_prop_star[j] = R::dnorm(beta_star[j], beta_old[j], MH_beta, true);
    log_prop_old[j] = R::dnorm(beta_old[j], beta_star[j], MH_beta, true);
  }
  
  arma::rowvec loglik_star = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_star, theta_old))), true);
  arma::rowvec loglik_old = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_old))), true);
  arma::rowvec log_dnorm_star =  dnorm(as<NumericVector>(wrap(beta_star)), b0, B0, true);
  arma::rowvec log_dnorm_old = dnorm(as<NumericVector>(wrap(beta_old)), b0, B0, true);
  
  arma::rowvec log_cc_star = loglik_star + log_dnorm_star;
  arma::rowvec log_cc_old = loglik_old + log_dnorm_old;
  arma::vec log_post_dens = log_cc_star.t() + log_prop_old - log_cc_old.t() - log_prop_star;
  NumericVector log_ap = pmin(as<NumericVector>(wrap(log_post_dens)), 0);
  
  NumericVector log_u = log(runif(J, 0, 1));
  NumericVector sample = ifelse(log_u < log_ap, as<NumericVector>(wrap(beta_star)), as<NumericVector>(wrap(beta_old)));
  
  return(sample);
}