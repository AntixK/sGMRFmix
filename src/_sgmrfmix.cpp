#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <armadillo>

// sGMRFmix headers
#include "headers/sGMRFmix.h"
#include "headers/compute_anomaly.h"

/* ---------------------------
 *       C++ Code
 * --------------------------*/



/* ---------------------------
 *     Python Interface
 * --------------------------*/

namespace py = pybind11;

// Wrap the sGMRFmix C++ code with Numpy array IO
py::dict py_sGMRFmix(
        py::array_t<double, py::array::f_style> train_data,
        int init_K,
        double rho,
        py::array_t<double, py::array::f_style> m0 ,
        bool do_kmeans = false,
        double pi_threshold = 0.01,
        double lambda0 = 1.0,
        int max_iter = 500,
        double tol = 1e-1,
        bool verbose = false,
        int random_seed = 69){

    if (train_data.ndim() != 2) throw std::runtime_error("Train data must be a 2D Numpy array");

    if (init_K > train_data.shape()[0] ) throw std::runtime_error("init_K must be <= # of rows in train data");

    int N = train_data.shape()[0];
    int M = train_data.shape()[1];

    arma::cube A;
    arma::mat m,g_mat;

    // Convert pybind wrappers around STL to armadillo objects
    auto r = train_data.mutable_unchecked<2>(); // 2D array
    arma::mat X = arma::mat(
            r.mutable_data(0, 0),
            N,
            M,
            /*copy_aux_mem*/ false,
            /*strict*/ true);

    auto q = m0.mutable_unchecked<1>(); // 1D array
    arma::rowvec _m0 = arma::rowvec(
                    q.mutable_data(0),
                    M,
                    false,
                    true);

    sGMRFmix(X, init_K, rho, _m0, A, m,  g_mat, false, pi_threshold,lambda0, max_iter, tol, verbose, random_seed);

    // Return K, A, m, g_mat
    auto results = py::dict();

    // Dict only accepts const chat*
    const char r1 = 'K', r2= 'A', r3 = 'm', r4='g';
    results[&r1] = init_K;
    results[&r2] = A;
    results[&r3] = m;
    results[&r4] = g_mat;

    return results;
}

py::dict py_compute_anomaly(
        py::array_t<double, py::array::f_style> test_data,
        py::array_t<double, py::array::f_style> A,
        py::array_t<double, py::array::f_style> m,
        py::array_t<double, py::array::f_style> g_mat,
        bool verbose = false
        ){

    if (A.ndim() != 3) throw std::runtime_error("A must be a 3D Numpy array");

    arma::mat anomaly_score;


    int N = test_data.shape()[0];
    int M = test_data.shape()[1];
    int K = m.shape()[0];

    // Convert pybind wrappers around STL to armadillo objects
    auto r1 = test_data.mutable_unchecked<2>(); // 2D array
    arma::mat X = arma::mat(
            r1.mutable_data(0, 0),
            N,
            M,
            /*copy_aux_mem*/ false,
            /*strict*/ true);

    auto r2 = m.mutable_unchecked<2>();  // 2D array
    arma::rowvec _m = arma::mat(
            r2.mutable_data(0, 0),
            K,
            M,
            false,
            true);

    auto r3 = g_mat.mutable_unchecked<2>(); // 2D array
    arma::rowvec _g_mat = arma::mat(
            r3.mutable_data(0, 0),
            N,
            K,
            false,
            true);

    auto r4 = A.mutable_unchecked<3>(); // 3D array
    arma::cube _A = arma::cube(
            r4.mutable_data(0, 0, 0),
            M,
            M,
            K,
            false,
            true);

//    arma::cube _A = arma::conv_to<cube>::from(A);

    compute_anomaly_score(X, _A, _m, _g_mat, anomaly_score, verbose);
    auto results = py::dict();

    const char l = 'a';
    results[&l] = anomaly_score;

    return results;
}


// wrap as Python module
PYBIND11_MODULE(_sgmrfmix, module) {
module.doc() = "A wrapper module around the sGMRFmix C++ implementation";

module.def("sgmrfmix_fit", &py_sGMRFmix, "Fits a sGMRFmix model for the given data ");
module.def("compute_anomaly_score", &py_compute_anomaly, "Computes the anomaly score based on the learnt sGMRFmix model");
}