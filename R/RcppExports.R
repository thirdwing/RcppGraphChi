
large_svd <- function(matrix_file) {
    .Call('RcppGraphChi_large_svd', PACKAGE = 'RcppGraphChi', matrix_file)
}

