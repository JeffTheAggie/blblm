// [[Rcpp::depends(RcppArmadillo]]
#include <RcppArmadillo.h>
using namespace Rcpp;
//' @title An Rcpp function that constructs a linear regression model
//' @name fasttau
//'
//' @param X numeric vector
//' @param y numeric vector
//'
//' @return numeric scalar
//' @export
//' @useDynLib blblm
//' @importFrom Rcpp sourceCpp
//'
// [[Rcpp::export]]
List fastLm_impl(const arma::mat& X, const arma::colvec& y){
  int n = X.n_rows, k = X.n_cols;
  arma::colvec coef = arma::solve(X, y);
  arma::colvec res = y - X*coef;
  double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - k);
  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));
  return List::create(Named("coefficients") = coef,
                      Named("Standard Error") = std_err,
                      Named("df.residual") = n - k);
  }