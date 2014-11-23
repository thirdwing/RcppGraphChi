
#include "toolkits/collaborative_filtering/common.hpp"
#include "toolkits/collaborative_filtering/types.hpp"
#include "toolkits/collaborative_filtering/eigen_wrapper.hpp"
#include "toolkits/collaborative_filtering/timer.hpp"

#include <RcppEigen.h>

using namespace std;

#define GRAPHCHI_DISABLE_COMPRESSION

/* Metrics object for keeping track of performance counters
     and other information. Currently required. */
metrics m("svd-inmemory-factors");

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
#include "toolkits/collaborative_filtering/math.hpp"

void init_lanczos(bipartite_graph_descriptor & info, int & data_size, int & nsv, int & nv, int & max_iter, int & actual_vector_len)
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
    
}


vec lanczos( bipartite_graph_descriptor & info, timer & mytimer, vec & errest,
             const std::string & vecfile, int & nconv, int & data_size, int & nv, 
             int & nsv, int & max_iter, double & tol, bool &finished)
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
        Rcpp::Rcout<<"Starting iteration: " << its << " at time: " << mytimer.current_time() << std::endl;
        
        int k = nconv;
        int n = nv;

        alpha = zeros(n);
        beta = zeros(n);

        U[k] = V[k]*A._transpose();

        orthogonalize_vs_all(U, k, alpha(0));

        for (int i=k+1; i<n; i++)
        {
            Rcpp::Rcout <<"Starting step: " << i << " at time: " << mytimer.current_time() <<  std::endl;

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

	Rcpp::Rcout << "Number of computed signular values: " << nconv << std::endl;
	
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
		
    }

    return sigma;
}



// [[Rcpp::export]]
Eigen::VectorXd large_svd(std::string matrix_file)
{
    int nshards;
    int nconv = 0;

    vec singular_values;

	//LANCZOS VARIABLES
    int max_iter = 5;
    int actual_vector_len;
    int nv = 5;
    int nsv = 3;
    double tol = 1e-1;
    bool finished = false;
    int ortho_repeats = 3;
    bool save_vectors = false;
    std::string format = "matrixmarket";
    int nodes = 0;
    int data_size = max_iter;


    graphchi_init();

    std::string vecfile = "";
    debug = 0;

    if (nv < nsv)
    {
        Rcpp::stop("Please set the number of vectors to be at least the number of support vectors!\n");
        
    }

    training = matrix_file;

    /* Preprocess data if needed, or discover preprocess files */
    if (tokens_per_row == 3 || tokens_per_row == 2)
        nshards = convert_matrixmarket<EdgeDataType>(training,0,0,tokens_per_row);
    else
		Rcpp::stop("--tokens_per_row should be either 2 or 3 input columns\n");

    info.rows = M;
    info.cols = N;
    info.nonzeros = L;

    timer mytimer;
    mytimer.start();
    init_lanczos(info, data_size, nsv, nv, max_iter, actual_vector_len);
    init_math(info, ortho_repeats);

    //read initial vector from file (optional)
    if (vecfile.size() > 0)
    {
        Rcpp::Rcout << "Load inital vector from file" << vecfile << std::endl;
        load_matrix_market_vector(vecfile, 0, true, false);
    }

    graphchi_engine<VertexDataType, EdgeDataType> engine(training, nshards, false, m);
    set_engine_flags(engine);
    pengine = &engine;

    vec errest;
    singular_values = lanczos(info, mytimer, errest, vecfile, nconv, data_size, nv, nsv, max_iter, tol, finished);
    singular_values.conservativeResize(nconv);
    Rcpp::Rcout << "Lanczos finished, time used: " << mytimer.current_time() << std::endl;

	return singular_values;

}
