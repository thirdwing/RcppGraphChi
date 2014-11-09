
#include "toolkits/collaborative_filtering/common.hpp"
#include "toolkits/collaborative_filtering/types.hpp"
#include "toolkits/collaborative_filtering/eigen_wrapper.hpp"
#include "toolkits/collaborative_filtering/timer.hpp"
using namespace std;

#define GRAPHCHI_DISABLE_COMPRESSION
int nshards;
int nconv = 0;
/* Metrics object for keeping track of performance counters
     and other information. Currently required. */
metrics m("svd-inmemory-factors");
vec singular_values;

struct vertex_data
{
    vec pvec;
    double value;
    double A_ii;
    vertex_data()
    {
        value = 0;
        A_ii = 1;
    }

    void set_val(int field_type, double value)
    {
        pvec[field_type] = value;
    }
    double get_val(int field_type)
    {
        return pvec[field_type];
    }
    //double get_output(int field_type){ return pred_x; }
}; // end of vertex_data


/**
 * Type definitions. Remember to create suitable graph shards using the
 * Sharder-program.
 */
typedef vertex_data VertexDataType;
typedef float EdgeDataType;

graphchi_engine<VertexDataType, EdgeDataType> * pengine = NULL;
std::vector<vertex_data> latent_factors_inmem;

#include "toolkits/collaborative_filtering/io.hpp"
#include "toolkits/collaborative_filtering/rmse.hpp"
#include "toolkits/collaborative_filtering/rmse_engine.hpp"

/** compute a missing value based on SVD algorithm */
float svd_predict(const vertex_data& user,
                  const vertex_data& movie,
                  const float rating,
                  double & prediction,
                  void * extra = NULL)
{

    Eigen::DiagonalMatrix<double, Eigen::Dynamic> diagonal_matrix(nconv);
    diagonal_matrix.diagonal() = singular_values;

    prediction = user.pvec.head(nconv).transpose() * diagonal_matrix * movie.pvec.head(nconv);
    //truncate prediction to allowed values
    prediction = std::min((double)prediction, maxval);
    prediction = std::max((double)prediction, minval);
    //return the squared error
    float err = rating - prediction;
    assert(!std::isnan(err));
    return err*err;

}

/**
 *
 *  Implementation of the Lanczos algorithm, as given in:
 *  http://en.wikipedia.org/wiki/Lanczos_algorithm
 *
 *  Code written by Danny Bickson, CMU, June 2011
 * */

//LANCZOS VARIABLES
int max_iter = 10;
int actual_vector_len;
int nv = 0;
int nsv = 0;
double tol = 1e-8;
bool finished = false;
int ortho_repeats = 3;
bool save_vectors = false;
std::string format = "matrixmarket";
int nodes = 0;

int data_size = max_iter;

#include "toolkits/collaborative_filtering/math.hpp"

void init_lanczos(bipartite_graph_descriptor & info)
{
    srand48(time(NULL));
    latent_factors_inmem.resize(info.total());
    data_size = nsv + nv+1 + max_iter;
    if (info.is_square())
        data_size *= 2;
    actual_vector_len = data_size;
    #pragma omp parallel for
    for (int i=0; i< info.total(); i++)
    {
        latent_factors_inmem[i].pvec = zeros(actual_vector_len);
    }
    logstream(LOG_INFO)<<"Allocated a total of: " << ((double)actual_vector_len * info.total() * sizeof(double)/ 1e6) << " MB for storing vectors." << std::endl;
}

void output_svd_result(std::string filename)
{
    MMOutputter_mat<vertex_data> user_mat(filename + "_U.mm", 0, M , "This file contains SVD output matrix U. In each row nconv factors of a single user node.", latent_factors_inmem, nconv);
    MMOutputter_mat<vertex_data> item_mat(filename + "_V.mm", M  ,M+N, "This file contains SVD  output matrix V. In each row nconv factors of a single item node.", latent_factors_inmem, nconv);
    logstream(LOG_INFO) << "SVD output files (in matrix market format): " << filename << "_U.mm" <<
                        ", " << filename + "_V.mm " << std::endl;
}

vec lanczos( bipartite_graph_descriptor & info, timer & mytimer, vec & errest,
             const std::string & vecfile)
{

    int its = 1;
    DistMat A(info);
    DistSlicedMat U(info.is_square() ? data_size/2 : 0, info.is_square() ? data_size : data_size, true, info, "U");
    DistSlicedMat V(0, data_size, false, info, "V");
    vec alpha, beta, b;
    vec sigma = zeros(data_size);
    errest = zeros(nv);
    DistVec v_0(info, 0, false, "v_0");
    if (vecfile.size() == 0)
        v_0 = randu(size(A,2));

    DistDouble vnorm = norm(v_0);
    v_0=v_0/vnorm;

    while(nconv < nsv && its < max_iter)
    {

        std::cout<<"Starting iteration: " << its << " at time: " << mytimer.current_time() << std::endl;
        int k = nconv;
        int n = nv;

        alpha = zeros(n);
        beta = zeros(n);

        U[k] = V[k]*A._transpose();

        orthogonalize_vs_all(U, k, alpha(0));

        for (int i=k+1; i<n; i++)
        {
            std::cout <<"Starting step: " << i << " at time: " << mytimer.current_time() <<  std::endl;

            V[i]=U[i-1]*A;
            orthogonalize_vs_all(V, i, beta(i-k-1));
            U[i] = V[i]*A._transpose();
            orthogonalize_vs_all(U, i, alpha(i-k));
        }

        V[n]= U[n-1]*A;
        orthogonalize_vs_all(V, n, beta(n-k-1));

        //compute svd of bidiagonal matrix

        n = nv - nconv;
        alpha.conservativeResize(n);
        beta.conservativeResize(n);

        mat T=diag(alpha);
        for (int i=0; i<n-1; i++)
            set_val(T, i, i+1, beta(i));

        mat a,PT;
        svd(T, a, PT, b);

        alpha=b.transpose();

        for (int t=0; t< n-1; t++)
            beta(t) = 0;

        //estiamte the error

        int kk = 0;
        for (int i=nconv; i < nv; i++)
        {
            int j = i-nconv;

            sigma(i) = alpha(j);

            errest(i) = abs(a(n-1,j)*beta(n-1));

            if (alpha(j) >  tol)
            {
                errest(i) = errest(i) / alpha(j);

            }
            if (errest(i) < tol)
            {
                kk = kk+1;

            }

            if (nconv +kk >= nsv)
            {
                finished = true;
            }
        }//end for

        vec v;
        if (!finished)
        {

            vec swork=get_col(PT,kk);

            v = zeros(size(A,1));
            for (int ttt=nconv; ttt < nconv+n; ttt++)
            {
                v = v+swork(ttt-nconv)*(V[ttt].to_vec());
            }

        }

        //compute the ritz eigenvectors of the converged singular triplets
        if (kk > 0)
        {

            mat tmp= V.get_cols(nconv,nconv+n)*PT;
            V.set_cols(nconv, nconv+kk, get_cols(tmp, 0, kk));

            tmp= U.get_cols(nconv, nconv+n)*a;
            U.set_cols(nconv, nconv+kk,get_cols(tmp,0,kk));

        }

        nconv=nconv+kk;
        if (finished)
            break;

        V[nconv]=v;

        its++;


    } // end(while)

    printf(" Number of computed signular values %d",nconv);
    printf("\n");
    DistVec normret(info, nconv, false, "normret");
    DistVec normret_tranpose(info, nconv, true, "normret_tranpose");

    for (int i=0; i < std::min(nsv,nconv); i++)
    {
        normret = V[i]*A._transpose() -U[i]*sigma(i);
        double n1 = norm(normret).toDouble();

        normret_tranpose = U[i]*A -V[i]*sigma(i);
        double n2 = norm(normret_tranpose).toDouble();

        double err=sqrt(n1*n1+n2*n2);

        if (sigma(i)>tol)
        {
            err = err/sigma(i);
        }

        printf("Singular value %d \t%13.6g\tError estimate: %13.6g\n", i, sigma(i),err);
    }

    return sigma;
}

#include <Rcpp.h>

// [[Rcpp::export]]
void large_svd()
{

    graphchi_init();

    std::string vecfile = "";
    debug         = 0;
    ortho_repeats = 3;
    nv = 5;
    nsv = 3;
    tol = 1e-1;
    save_vectors = 0;
    max_iter = 5;

    global_logger().set_log_level(LOG_WARNING);

    if (nv < nsv)
    {
        std::cout <<"Please set the number of vectors --nv=XX, to be at least the number of support vectors --nsv=XX or larger" << std::endl;
        exit(1);
    }

    training = "smallnetflix_mm";
    std::cout << "Load matrix " << training << std::endl;

    /* Preprocess data if needed, or discover preprocess files */
    if (tokens_per_row == 3 || tokens_per_row == 2)
        nshards = convert_matrixmarket<EdgeDataType>(training,0,0,tokens_per_row);

    else logstream(LOG_FATAL)<<"--tokens_per_row=XX should be either 2 or 3 input columns" << std::endl;


    info.rows = M;
    info.cols = N;
    info.nonzeros = L;
    assert(info.rows > 0 && info.cols > 0 && info.nonzeros > 0);

    timer mytimer;
    mytimer.start();
    init_lanczos(info);
    init_math(info, ortho_repeats);

    //read initial vector from file (optional)
    if (vecfile.size() > 0)
    {
        std::cout << "Load inital vector from file" << vecfile << std::endl;
        load_matrix_market_vector(vecfile, 0, true, false);
    }

    graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m);
    set_engine_flags(engine);
    pengine = &engine;

    vec errest;
    singular_values = lanczos(info, mytimer, errest, vecfile);
    singular_values.conservativeResize(nconv);
    std::cout << "Lanczos finished " << mytimer.current_time() << std::endl;

    write_output_vector(training + ".singular_values", singular_values,false, "%GraphLab SVD Solver library. This file contains the singular values.");


}


