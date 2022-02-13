// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loglik.h"
using namespace arma;
using namespace Rcpp;

// SAMPLING ALPHA
NumericVector alpha_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double a0, double A0, double MH_alpha) {
  // NOTE:
  // _star -> candidate
  // _old -> previous 
  
  int J = Y.n_cols; //# of alpha
  arma::vec alpha_star(J); //candidate sample for alpha
  arma::vec log_prop_star(J); //log proposal density for alpha_star
  arma::vec log_prop_old(J); //log proposal density for alpha_old
  
  //Sample theta_star and log proposal density.
  for (int j = 0; j < J; j++) {
    alpha_star[j] = R::rnorm(alpha_old[j], MH_alpha);
    log_prop_star[j] = R::dnorm(alpha_star[j], alpha_old[j], MH_alpha, true);
    log_prop_old[j] = R::dnorm(alpha_old[j], alpha_star[j], MH_alpha, true);
  }
  
  //log-likelihood
  arma::rowvec loglik_star = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_star, beta_old, theta_old))), true);
  arma::rowvec loglik_old = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_old))), true);
  
  //log prior density
  arma::rowvec log_dnorm_star =  dnorm(as<NumericVector>(wrap(alpha_star)), a0, A0, true);
  arma::rowvec log_dnorm_old = dnorm(as<NumericVector>(wrap(alpha_old)), a0, A0, true);
  
  //log posterior density
  arma::rowvec log_pd_star = loglik_star + log_dnorm_star;
  arma::rowvec log_pd_old = loglik_old + log_dnorm_old;
  
  //log acceptance probability
  arma::vec log_densfrac = log_pd_star.t() + log_prop_old - log_pd_old.t() - log_prop_star;
  NumericVector log_ap = pmin(as<NumericVector>(wrap(log_densfrac)), 0);
  NumericVector log_u = log(runif(J, 0, 1));
  
  //save samples
  NumericVector sample = ifelse(log_u < log_ap, as<NumericVector>(wrap(alpha_star)), as<NumericVector>(wrap(alpha_old)));
  
  return(sample);
}
