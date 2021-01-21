/*

 Reference: https://gallery.rcpp.org/articles/dmvnorm_arma/

 dvmnorm (Similar to the dmvnorm in R), evaluates the probability density of a multivariate normal distribution.

 Recall that the log likelihood of a multivariate normal distribution is given by

    log_prob = 0.5*[log (|\Sigma|) + (x - \mu)^T \Sigma^{-1} (x ^ \mu) + D log(2 \pi)]

 To compute \Sigma^{-1} and log |\Sigma|, we use cholesky decomposition into upper-triangular matrix.
 The resulting method is quite fast and generally numerically stable. Although it is not the most stable version.

*/

#ifndef DMVNORM_H
#define DMVNORM_H

#endif //DMVNORM_H


using namespace arma;

static double const log2pi = log(2.0 * M_PI);

vec dmvnorm(const Mat<double> &x,
            const rowvec &mu,
            const Mat<double> &sigma,
            bool return_log = false){

    int N = x.n_rows, D = x.n_cols;
    vec log_prob(N);

    Mat<double> const rooti = inv(trimatu(chol(sigma)));
    double rootisum = accu(log(rooti.diag()));
    rootisum += -(double)D/2.0 * log2pi;

    rowvec z;

    for(int i =0; i < N; ++i){
        z = (x.row(i) - mu) * rooti; // Can be made faster by computing inplace
        log_prob(i) = rootisum - 0.5*dot(z,z);
    }

    if (return_log){
        return log_prob;
    }

    return exp(log_prob);
}

