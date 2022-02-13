// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loglik.h"
using namespace arma;
using namespace Rcpp;

// SAMPLING THETA
NumericVector theta_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double MH_theta) {
  
  // NOTE:
  // _star -> candidate
  // _old -> previous 
  
  int I = Y.n_rows; //# of theta
  arma::vec theta_star(I); //candidate sample for theta
  arma::vec log_prop_star(I); //log proposal density for theta_star
  arma::vec log_prop_old(I); //log proposal density for theta_old
  
  //Sample theta_star and log proposal density.
  for (int i = 0; i < I; i++) {
    theta_star[i] = R::rnorm(theta_old[i], MH_theta);
    log_prop_star[i] = R::dnorm(theta_star[i], theta_old[i], MH_theta, true);
    log_prop_old[i] = R::dnorm(theta_old[i], theta_star[i], MH_theta, true);
  }
  
  //log-likelihood
  arma::vec loglik_star = rowSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_star))), true);
  arma::vec loglik_old = rowSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_old))), true);
  
  //log prior density
  arma::vec log_dnorm_star =  dnorm(as<NumericVector>(wrap(theta_star)), 0, 1, true);
  arma::vec log_dnorm_old = dnorm(as<NumericVector>(wrap(theta_old)), 0, 1, true);
  
  //log posterior density
  arma::vec log_pd_star = loglik_star + log_dnorm_star;
  arma::vec log_pd_old = loglik_old + log_dnorm_old;
  
  //log acceptance probability
  arma::vec log_densfrac = log_pd_star + log_prop_old - log_pd_old - log_prop_star;
  NumericVector log_ap = pmin(as<NumericVector>(wrap(log_densfrac)), 0);
  NumericVector log_u = log(runif(I, 0, 1));
  
  //save samples
  NumericVector sample = ifelse(log_u < log_ap, as<NumericVector>(wrap(theta_star)), as<NumericVector>(wrap(theta_old)));
  
  return(sample);
}
