// [[Rcpp::depends(RcppArmadillo)]]
#ifndef ALPHA_SAMPLE_H
#define ALPHA_SAMPLE_H
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

NumericVector alpha_sample(arma::mat Y, arma::vec alpha_old, arma::vec beta_old,
                           arma::vec theta_old, double a0, double A0, double MH_alpha);

#endif