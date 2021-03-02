/*
 *
 */
//#undef NDEBUG
#ifndef SGMRFMIX_H
#define SGMRFMIX_H

#endif //SGMRFMIX_H

#include <chrono>
#include <armadillo>
#include <cassert>
#include "sgaussmix.h"
#include "gmrfmix.h"

using namespace arma;


void sGMRFmix(const Mat<double> &X, // (N, M)
              int &K,
              double rho,
              rowvec &m0,   // (M x 1)
              // Return arguments
              Cube<double> &A, // (M, M, K)
              Mat<double> &m, // (K, M)
              Mat<double> &g_mat, // (N , K)
              rowvec &result_pi,
              bool do_kmeans=false,
              double pi_threshold = 0.01,
              double lambda0 = 1.0,
              int max_iter=500,
              double tol = 1e-1,
              bool verbose=false,
              int random_seed = 69){

    // For reproducibility
    std::srand (random_seed);
    arma_rng::set_seed(random_seed);
    std::chrono::steady_clock sc;

    int N = X.n_rows,
        M = X.n_cols;

    // Preprocess data (Not required)
    if(verbose){
        std::cout<<termcolor::blue<<"================= sGMRFmix Model ================="<<endl;
    }

    rowvec pi(K, fill::zeros);
    Cube<double> Ak (M, M,K, fill::zeros);
    Mat<double> _m(K, M, fill::zeros);

    if(verbose){
        std::cout<<termcolor::blue<<"Running sparse Gaussian Mixture Model."<<endl;
    }

    auto start = sc.now();     // start timer
    sparseGaussMix(X, K, rho, m0, pi, Ak, _m, do_kmeans, lambda0, max_iter, tol, verbose);

    if(verbose){
        std::cout<<termcolor::blue<<"Completed sparse Gaussian Mixture Model."<<endl;
    }


    //============================================================================ //
    uvec inds = find(pi >= pi_threshold);
    int new_K = inds.n_elem;

    // Reset the pi based on optimal number of mixtures
    result_pi.resize(new_K);
    result_pi.fill(0.0);
    A.resize(M, M, new_K);
    m.resize(new_K, M);

    double pi_sum = 0.0;
    for(int k=0; k < new_K; ++k){

        result_pi(k) = pi(inds(k));
        A.slice(k) = Ak.slice(inds(k));
        m.row(k) = _m.row(inds(k));

        pi_sum += result_pi(k);

    }
    for(auto &p : result_pi){
        p /= pi_sum;
    }

    // Placeholders for GMRF results
    g_mat.resize(N, new_K);
    Mat<double> log_theta_mat(M, new_K, fill::zeros);

    Cube<double> U(N, M, new_K, fill::zeros),
                 W(N, M, new_K, fill::zeros);

    if(verbose){
        std::cout<<termcolor::blue<<"Running sparse GMRF Model."<<endl;
    }
    GMRFmix(X, result_pi, A, m, log_theta_mat, U, W, g_mat, max_iter, tol, verbose);

    if(verbose){
        std::cout<<termcolor::blue<<"Completed sparse GMRF Model."<<endl;
    }

    // ============================================================================ //
    // Compute Mode
    Mat<double> Sigma(M, M),
                loglik_mat(N, new_K);

    for(int k=0; k < new_K; ++k){
        Sigma = inv_sympd(A.slice(k));
        loglik_mat.col(k) = dmvnorm(X, m.row(k), Sigma, true);
    }

    ucolvec mode = index_max(loglik_mat, 1); // index of Max value in each row

    // Return Results
    K = new_K;

    auto end = sc.now();
    auto time_span = static_cast<std::chrono::duration<double>>(end - start);
    std::cout <<termcolor::blue<<"Operation took: " << time_span.count() << " seconds"<<endl;
    if(verbose){
        std::cout<<termcolor::blue<<"=================================================="<<endl;
    }

}