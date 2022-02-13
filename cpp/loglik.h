// [[Rcpp::depends(RcppArmadillo)]]
#ifndef LOGLIK_H
#define LOGLIK_H
#include <RcppArmadillo.h>
using namespace arma;

arma::mat loglik(arma::mat Y, arma::vec alpha, 
                 arma::vec beta, arma::vec theta);

#endif