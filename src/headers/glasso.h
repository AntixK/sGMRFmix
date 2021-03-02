#ifndef GLASSO_H
#define GLASSO_H

#endif //SGRMRFMIX_GLASSO_H

/*
 * Fast Graphical Lasso Implementation
 *
 * Reference
 * [1] https://apps.cs.utexas.edu/apps/sites/default/files/tech_reports/TR-2105.pdf
 */

#include "../lib/termcolor.h"

using namespace arma;

SpMat<double> GLasso (const Mat<double> &S,
                      double alpha,
                      bool verbose = false,
                      double threshold = 1e-12,
                      int max_iter = 1000){
    int N = S.n_cols;

    Mat<double> L(N, N);
    L.fill(alpha);

    double eps = 1e-16;
    double lasso_thresh;
    double convergence_thresh = accu(abs(S));
    convergence_thresh -= accu(abs(S.diag()));

    if (convergence_thresh <= 1e-12){ // Approximately 0 => S is a diagonal matrix!
        SpMat<double> X(size(S));
        vec d = S.diag() + L.diag();
        X.diag() = 1.0/(d + eps);
        return X;
    }

    convergence_thresh *= threshold/(N-1);
    lasso_thresh = convergence_thresh / N;

    if (lasso_thresh < 2 * eps){
        lasso_thresh = 2*eps;
    }
//    std::cout<<"lasso_thresh:"<<lasso_thresh<<endl;

    // Cold Start
    SpMat<double> X(size(S));
    Mat<double> W(S);

    vec d = S.diag() + L.diag();
    W.diag() = d;

    int num_iter = 0;
    vec vj(N, fill::zeros);
    while(num_iter < max_iter) {

        double dw = 0.0;
        for (int j = 0; j < N; ++j) {
            vj.fill(0.0);

            for(int  k =0; k < N; ++k){
                if(X(k,j) != 0.0){
                    vj += W.col(k)*X(k,j);
                }
            }
            int inner_iter = 0;
            double dlx;
            do {
                dlx = 0.0;
                for (int i = 0; i < N; ++i) {
                    double c = 0.0;
                    if (i != j) {
                        double a = S(i, j) - vj(i) + d(i) * X(i, j);
                        double b = std::abs(a) - L(i,j);

                        if (b > 0.0){
                            c = copysign(b,a)/d(i); // Return sign(a) * b
                        } else{
                            c = 0.0;
                        }

                        double delta = c - X(i,j);

                        if (delta != 0.0){
                            X(i,j) = c;
                            vj += W.col(i) * delta;
//                            cout<<delta<<"  "<<dlx;
                            dlx = std::max(dlx, std::abs(delta));
                        }
                    }
                }
                inner_iter ++;
//                if(inner_iter > 500){
////                    std::cout<<dlx<<endl;
//                    std::cout<<inner_iter<<endl;
//
//                }
            } while (dlx >= lasso_thresh && inner_iter < max_iter / 10);

            vj(j) = d(j);
            dw = std::max(dw, accu(abs(vj - W.col(j))));

            W.col(j) = vj;
            W.row(j) = vj.t();
        }
        num_iter ++;

        // Check for Convergence
        if (dw <= convergence_thresh){
          if(verbose){
                std::cout<<termcolor::green <<"Fast GLASSO method has converged within "
                    <<convergence_thresh<<" tolerance."<<endl;
            }
            break;
        }
    }

    for (int i = 0; i < N; ++i){
        double a = 1/(d(i) - accu(W.col(i) % X.col(i)));
        X.col(i) *= -a;
        X(i,i) = a;
    }

    X = 0.5*(X + X.t());

    return X;
}
