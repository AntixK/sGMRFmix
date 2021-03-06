/*
 * 
 */

#ifndef GMRFMIX_H
#define GMRFMIX_H

#endif //GMRFMIX_H
#include <boost/math/special_functions/digamma.hpp>
#include "../lib/termcolor.h"
#include <cmath>


void GMRFmix( const Mat<double> &X, // (N, M)
              rowvec &pi, // (1 x K)
              const Cube<double> &A, // (M, M, K)
              const Mat<double> &m, // (K, M),
             //Return Arguments
             Mat<double> &theta_mat, // (M x K)
             Cube<double> &U, // (N, M, K)
             Cube<double> &W, // (N, M, K)
             Mat<double> &g_mat, //(N, K)
             int max_iter=500,
             double tol = 1e-1,
             bool verbose = false){

    int N = X.n_rows, M = X.n_cols;
    int K = pi.size();


    // Compute the Mean & Variance of each K slice
    // Equation (4), (5) in section 3
    for(int k=0; k < K; ++k){
        W.slice(k).each_row() = 1/A.slice(k).diag().as_row();
    }

    Cube<double> tmp(N, M, K, fill::zeros);
    for (int  k=0; k < K; ++k) {
        tmp.slice(k) = (X.each_row() - m.row(k)) * A.slice(k);

        for (int i = 0; i < M; ++i) {
            U.slice(k).col(i) = X.col(i) - (tmp.slice(k).col(i) / A(i, i, k));
        }
    }

    Mat<double> g_unnorm(N, K, fill::zeros);

    double loglik = 0.0,
           prev_loglik =0.0,
           delta_loglik;

    double a_bar, alpha_bar;
    rowvec a(K, fill::zeros),
           Nk(K, fill::zeros),
           alpha(K, fill::zeros),
           theta(K, fill::zeros);

    // Initialize alpha parameter for the
    // Dirichlet prior
    alpha.fill(1.0);

    for (int i=0; i< M; ++i){
        loglik = 0.0;
        prev_loglik = 0.0;

        // Equation (31) in section 4.2
        for(int k=0; k < K; ++k){
            Nk(k) = pi(k) * N;
        }

        int num_iter = 0;
        bool has_converged = false;
        while(num_iter < max_iter){
            num_iter++;
            // Iterate over all K
            a = alpha + Nk;


            // Equation (16) in section 3.2 (We take log for numerical stability)
            a_bar = sum(a);


            for(int k = 0; k < K; ++k) {
                theta(k) = exp(boost::math::digamma(a(k)) -
                               boost::math::digamma(a_bar));  // Log theta
            }

            assert(!theta.has_nan() && "GMRF Theta has NANs.");
            // Compute the Gating function (Posterior Distribution)
            // Equation (17) in section 3.2
            for(int k =0; k < K; ++k) {
                g_unnorm.col(k) = theta(k) * normpdf(X.col(i),
                                                             U.slice(k).col(i),
                                                             W.slice(k).col(i));
                g_mat.col(k) = g_unnorm.col(k);
            }
            vec denom = sum(g_mat, 1); // Sum each row
            g_mat.each_col() /= (denom + 1e-8);

            assert(g_mat.is_finite() && "GMRF GMat has NANs.");

            // Equation (18)
            Nk = sum(g_mat, 0); // Sum each column

            // Compute the log likelihood
            // Equation (10) in section 3.2
            prev_loglik = loglik;
            loglik = accu(trunc_log(max(g_unnorm, 1))); // Row-wise max
//            std::cout<<loglik<<endl;

            alpha_bar = accu(alpha);
            loglik -= lgamma(alpha_bar);

            loglik += accu(lgamma(alpha) + (alpha - 1.0) % log(theta));
            assert(is_finite(loglik) && "GMRF LogLik resulted in NAN.");

            delta_loglik = abs(loglik - prev_loglik);
            if(delta_loglik < tol && is_finite(delta_loglik)){

                if(verbose){
                    std::cout<<termcolor::green<<"VB Method for dim="<< i
                             << " has converged within "<<tol<<" tolerance."<<endl;
                }
                theta_mat.row(i) = theta;
                has_converged = true;
                break;
            }
        }

        if(!has_converged && verbose){
            std::cout<<termcolor::red<<"Warning: VB method for dim="<< i << " has not converged after "
                                     <<max_iter<<" iterations with error "<<delta_loglik <<".\nCheck the hyper-parameters or increase the maximum iterations."<<endl;
        }

    }
}

