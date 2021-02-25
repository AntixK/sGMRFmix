/*
 *  Term Reference:
 *   Below is a helpful list of variables used in the function
 *   along with their meaning from the original paper. This is
 *   mainly for future reference and code maintenance.

        X: Data matrix
        K: Number of mixtures
        rho: Regularization parameter for the Gauss-Laplace prior
        m0:
        do_kmeans: Flag specifying if
        lambda0:
        max_iter:
        tol:
        verbose:

        N: Number of data samples
        M: Dimension of each data sample
        Nk:
        pi:
        m:
        Lambda:
        invLambda:
        Q:
        Sigma:
        ln_r:
        r:
        x_bar:
        Ak:
 */

#ifndef SGAUSSMIX_H
#define SGAUSSMIX_H

#endif //SGAUSSMIX_H

#include "dmvnorm.h"
#include "glasso.h"
#include <cmath>
#include "../lib/termcolor.h"
//#include "../lib/progressbar.hpp"

using namespace arma;

// Function declarations
void to_symmetric(Mat<double> &X);
double compute_L1_norm(const Mat<double> &X);
double compute_loglik(const Mat<double> &X,
                      const Mat<double> &m,
                      const rowvec &m0,
                      const Mat<double> &r,
                      const rowvec &pi,
                      const Cube<double> &Lambda,
                      const Cube<double> &invLambda,
                      Mat<double> &loglik_mat,
                      double rho,
                      double lambda0,
                      int N,
                      int K);

void sparseGaussMix(const mat &X, // (N x M)
                    int K,
                    double rho,
                    rowvec &m0, // (1 x M)
                    // Return Arguments
                    rowvec &pi, // (1 x K)
                    Cube<double> &Ak, // (M, M, K)
                    Mat<double> &m, //(K x M)
                    //Defaults
                    bool do_kmeans = false,
                    double lambda0 = 1.0,
                    int max_iter = 500,
                    double tol = 1e-1,
                    bool verbose=false){

    int N = X.n_rows, M = X.n_cols;

    pi.fill(1.0/K);
    rowvec lambda = pi * N;

    // Check if num clusters is divisible by N
    // For ease of computation
    assert(N % K == 0 && "N must be divisible by K");
    int n_sampl = N / K;
    Cube<double> split_X(n_sampl, M, K, fill::zeros);

    // Initialize the Mean and Covariance of the GMM
    if (do_kmeans){

          std::cerr<<termcolor::red<<"Error: K-Means is currently not supported. Set the flag to false."<<endl;
    }

    if (!do_kmeans) {
        int i = 0;
        for (int k = 0; k < K; ++k) {

            split_X.slice(k) = X.rows(i, i + n_sampl - 1);
            i += n_sampl;
        }
    }

    Cube<double> Sigma(M, M,K, fill::zeros);
    for(int k = 0; k <K; ++k){
        m.row(k) = mean(split_X.slice(k), 0); // ColMeans
        Sigma.slice(k) = cov(split_X.slice(k));
    }


    Cube<double> Lambda(M, M,  K,fill::zeros),
                 invLambda(M, M, K, fill::zeros),
                 Q(M, M, K, fill::zeros);

    Mat<double> ln_r(N, K),
                r(N, K),
                x_bar(K, M);
    Mat<double> tmp(N, M, fill::zeros);
    rowvec tmp_row(M, fill::zeros);
    vec max_ln_r(N, fill::zeros);
    Mat<double> loglik_mat(N, K, fill::zeros);

    // Do Graphical LASSO
    for(int k=0; k < K; ++k){
        Lambda.slice(k) = GLasso(Sigma.slice(k), 0.1, verbose);            // Precision Matrix
//        invLambda.slice(k) = inv(trimatu(chol(Lambda.slice(k)))); // Covariance Matrix
        invLambda.slice(k) = inv(Lambda.slice(k)); // Covariance Matrix

    }



    double loglik = -datum::inf,
           prev_loglik,
           dloglik = 0.0;
//    X.print("X=");
//    m.print("m=");
//    Lambda.print("Lambda=");
    bool has_converged = false;
//    progressbar bar(max_iter);

    int num_iter = 0;
//    while(num_iter < max_iter){
    for(num_iter = 0; num_iter < max_iter; ++num_iter){
//        num_iter++;
//        bar.update();
//    for(int num_iter : tqdm::range(max_iter)){
        // Equation (29) in section 4.2
        for (int k = 0; k < K; ++k) {

            ln_r.col(k) = log(pi(k)) + dmvnorm(X,
                                               m.row(k),
                                               invLambda.slice(k), true) - M / (2 * lambda(k));
        }
        assert(!ln_r.has_nan() && "ln_r has NANs");



        // Equation (30) in section 4.2 (Softmax)
        // We use log-sum-exp method to prevent numerical overflows
        max_ln_r = max(ln_r, 1); // Max value of each row (N x 1)
        for (int k=0; k < K; ++k) {
            r.col(k) = exp(ln_r.col(k) - max_ln_r);
        }
        assert(!r.has_nan() && "Unnormalized r has NANs");

        // Normalize r
        vec denom = sum(r, 1); // Sum each row
        r.each_col() /= denom;

        // Add small noise for  numerical stability
        r.for_each( [](mat::elem_type &val) {val += 1e-31; });

        assert(!r.has_nan() && "r has NANs");


        //Equation (31) in section 4.2
        rowvec Nk = sum(r, 0); // Sum the columns
        pi = Nk/N;
        assert(!Nk.has_nan() && "Nk has NANs");
        assert(!pi.has_nan() && "pi has NANs");


        // Equation (32) in section 4.2
        for(int k=0; k < K; ++k){
            x_bar.row(k) = sum(X.each_col() % r.col(k) , 0) /Nk(k); // Column sum
        }

        assert(!x_bar.has_nan() && "x_bar has NANs");
//        std::cout<<termcolor::green<<"Computing Sigma"<<endl;

        // Equation (33) in section 4.2
        for(int k = 0; k < K; ++k){
            tmp = (X.each_row() - x_bar.row(k)); // NxM
            tmp = tmp.each_col() % sqrt(r.col(k));
            Sigma.slice(k) = (tmp.t() * tmp) / Nk(k); // MxM

        }
        assert(!Sigma.has_nan() && "Sigma has NANs");


        // Equation (34) in section 4.2
        lambda = lambda0 + Nk;
        for (int k=0; k < K; ++k){
            m.row(k) = (lambda0 * m0 + Nk(k)*x_bar.row(k))/lambda(k);
        }

//        std::cout<<termcolor::green<<"Computing Q"<<endl;

        // Equation (35) in section 4.2
        for(int k=0; k < K; ++k){
            tmp_row = (x_bar.row(k) - m0); // 1xM
            Q.slice(k) = Sigma.slice(k) + (lambda0 / lambda(k)) * (tmp_row.t() * tmp_row); // MxM

//            assert(Q.slice(k).is_sympd() && "Q is not SPD.");

        }
        assert(!Q.has_nan() && "Q has NANs");

        // Equation (36) in section 4.2
        // Notice that the equation (36) is simply the
        // L1 penalized likelihood of the Graphical LASSO objective. So,
        // we can directly employ the Graphical LASSO for each k.
        for(int k=0; k < K; ++k){
//            std::cout<<termcolor::green<<"Computing Lambda "<<k<<endl;

            Lambda.slice(k) = GLasso(Q.slice(k), rho/Nk(k));          // Precision Matrix

            if (rcond(Lambda.slice(k)) < 0.01){  // Handle the case for singular matrix
                std::cout<<termcolor::red<<"Warning! Lambda at slice "<<k<<" is ill-conditioned!"<<endl;
                invLambda.slice(k) = pinv(Lambda.slice(k)); //Q.slice(k);
            } else {
//                std::cout<<termcolor::green<<"Inverting lambda"<<Lambda.slice(k).size()<<endl;

                invLambda.slice(k) =inv(Lambda.slice(k)); // Covariance Matrix
//                std::cout<<approx_equal(invLambda.slice(k), Q.slice(k), "absdiff", 0.0002);

            }
            if(!invLambda.slice(k).is_symmetric()){
//                std::cout<<termcolor::green<<"Computing invlambda to symmetric"<<endl;
                to_symmetric(invLambda.slice(k));
            }
        }
        assert(!Lambda.has_nan() && "Lambda has NANs");
        assert(!invLambda.has_nan() && "invLambda has NANs");

        prev_loglik = loglik;
//        std::cout<<termcolor::green<<"Computing LogLik"<<endl;
        loglik = compute_loglik(X, m, m0, r, pi, Lambda, invLambda, loglik_mat, rho, lambda0, N, K);

        // Check for convergence
        dloglik = std::abs(loglik - prev_loglik);

        if (dloglik < tol && is_finite(dloglik)){
            has_converged = true;

            if(verbose){
                std::cout<<termcolor::green <<"VB method for sparse Gaussian Mixtures has converged within "
                    <<tol<<" tolerance."<<endl;
            }
            break;
        }
    }

    if (verbose && !has_converged){
        std::cerr<<termcolor::red <<"Warning: VB method of sparse Gaussian Mixtures has not converged after "
            <<max_iter<<" iterations with error "<<dloglik <<". \nCheck the hyper-parameters or increase the maximum iterations."<<endl;
    }

    // Next to Equation(37) in section 4.2
    for(int k=0; k < K; ++k){
        Ak.slice(k) = lambda(k)/(1 + lambda(k)) * Lambda.slice(k);
    }
}

double compute_loglik(const Mat<double> &X,
                      const Mat<double> &m,
                      const rowvec &m0,
                      const Mat<double> &r,
                      const rowvec &pi,
                      const Cube<double> &Lambda,
                      const Cube<double> &invLambda,
                      Mat<double> &loglik_mat,
                      double rho,
                      double lambda0,
                      int N,
                      int K){

    ucolvec max_inds = index_max(r, 1); // Find the index of max values in each row (N x 1)

    // Equation (24) in section 4.1
    double gauss_laplace_prior_lik = 0.0,
           cat_prior_lik = 0.0,
           model_lik = 0.0;

    for(int k=0; k < K; ++k){
        gauss_laplace_prior_lik += -rho * compute_L1_norm(Lambda.slice(k)) / 2.0;
        gauss_laplace_prior_lik += accu(dmvnorm(m.row(k), m0, invLambda.slice(k) / lambda0, true));
    }

    // Equation (25) in section 4.1
    for(auto& ind : max_inds){
        cat_prior_lik += log(pi(ind));
    }

//     Equation (23) in section 4.1
    for(int k=0; k < K; ++k){
        loglik_mat.col(k) = dmvnorm(X, m.row(k), invLambda.slice(k), true);
    }
//    assert(!invLambda.has_nan() && "invLambda has NANs.");
    assert(!loglik_mat.has_nan() && "LogLik Matrix has NANs.");

    for(int i=0; i < N; ++i){
        model_lik += loglik_mat(i, max_inds(i));
    }

    // Equation (26) in section 4.1
    return gauss_laplace_prior_lik +
           cat_prior_lik +
           model_lik;

}

double compute_L1_norm(const Mat<double> &X){
    // Next to Equation (25) in section 4.1
    return accu(abs(X));
}

void to_symmetric(Mat<double> &X){
    mat L = trimatu(X).t();
    X = trimatu(X) + L;
    X.diag() -= L.diag();
}
