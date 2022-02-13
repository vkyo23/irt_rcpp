// [[Rcpp::depends(RcppArmadillo)]]
#ifndef THETA_SAMPLE_H
#define THETA_SAMPLE_H
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

NumericVector theta_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double MH_theta);

#endif
