// [[Rcpp::depends(RcppArmadillo)]]
#ifndef BETA_SAMPLE_H
#define BETA_SAMPLE_H
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

NumericVector beta_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                          arma::vec theta_old, double b0, double B0, double MH_beta);

#endif