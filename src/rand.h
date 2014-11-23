#include <Rcpp.h>

namespace Rand
{

// mimic the behaviour of a C version rand()
inline int rand(void)
{
    Rcpp::RNGScope scp;
    double res = R::unif_rand() * RAND_MAX;
    return (int)res;
}

} // namespace Rand
