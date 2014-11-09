
#include <RcppEigen.h>

using namespace Rcpp;

// large_svd
Eigen::VectorXd large_svd(std::string matrix_file);
RcppExport SEXP RcppGraphChi_large_svd(SEXP matrix_fileSEXP) {
BEGIN_RCPP
    SEXP __sexp_result;
    {
        Rcpp::RNGScope __rngScope;
        Rcpp::traits::input_parameter< std::string >::type matrix_file(matrix_fileSEXP );
        Eigen::VectorXd __result = large_svd(matrix_file);
        PROTECT(__sexp_result = Rcpp::wrap(__result));
    }
    UNPROTECT(1);
    return __sexp_result;
END_RCPP
}
