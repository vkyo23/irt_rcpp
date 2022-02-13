// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;

// Log-likelihood function
arma::mat loglik(arma::mat Y, arma::vec alpha, 
                 arma::vec beta, arma::vec theta) {
  //calculate beta_j * theta_i
  arma::mat temp = beta * theta.t(); 
  
  //beta_j * theta_i - alpha_j
  arma::mat temp2 = temp.each_col() - alpha; 
  
  //exp(beta_j * theta_i - alpha_j)
  arma::mat exp_ = arma::exp(temp2.t()); 
  
  //inverse logit
  arma::mat p = exp_ / (1 + exp_);
  
  //calculate log-lik
  arma::mat log_lik = Y % arma::log(p) + (1 - Y) % arma::log(1 - p);
  
  return(log_lik);
}
