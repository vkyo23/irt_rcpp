// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "loglik.h"
using namespace arma;
using namespace Rcpp;

// SAMPLING BETA
NumericVector beta_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double b0, double B0, double MH_beta) {
  
  // NOTE:
  // _star -> candidate
  // _old -> previous 
  
  int J = Y.n_cols; //# of beta
  arma::vec beta_star(J); //candidate sample for beta
  arma::vec log_prop_star(J); //log proposal density for beta_star
  arma::vec log_prop_old(J); //log proposal density for beta_old
  
  //Sample theta_star and log proposal density.
  for (int j = 0; j < J; j++) {
    beta_star[j] = R::rnorm(beta_old[j], MH_beta);
    log_prop_star[j] = R::dnorm(beta_star[j], beta_old[j], MH_beta, true);
    log_prop_old[j] = R::dnorm(beta_old[j], beta_star[j], MH_beta, true);
  }
  
  //log-likelihood
  arma::rowvec loglik_star = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_star, theta_old))), true);
  arma::rowvec loglik_old = colSums(as<NumericMatrix>(wrap(loglik(Y, alpha_old, beta_old, theta_old))), true);
  
  //log prior density
  arma::rowvec log_dnorm_star =  dnorm(as<NumericVector>(wrap(beta_star)), b0, B0, true);
  arma::rowvec log_dnorm_old = dnorm(as<NumericVector>(wrap(beta_old)), b0, B0, true);
  
  //log posterior density
  arma::rowvec log_pd_star = loglik_star + log_dnorm_star;
  arma::rowvec log_pd_old = loglik_old + log_dnorm_old;
  
  //log acceptance probability
  arma::vec log_densfrac = log_pd_star.t() + log_prop_old - log_pd_old.t() - log_prop_star;
  NumericVector log_ap = pmin(as<NumericVector>(wrap(log_densfrac)), 0);
  NumericVector log_u = log(runif(J, 0, 1));
  
  //save samples
  NumericVector sample = ifelse(log_u < log_ap, as<NumericVector>(wrap(beta_star)), as<NumericVector>(wrap(beta_old)));
  
  return(sample);
}
