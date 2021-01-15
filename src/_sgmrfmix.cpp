#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <armadillo>

// sGMRFmix headers
#include "headers/sGMRFmix.h"
#include "headers/compute_anomaly.h"

namespace py = pybind11;

/* ---------------------------
 *       C++ helpers
 * --------------------------*/
arma::rowvec arr_to_rowvec(py::array_t<double, py::array::f_style> &arr){
    int N = arr.shape()[0];
    auto r = arr.mutable_unchecked<1>(); // 2D array
    arma::rowvec vec = arma::rowvec(
            r.mutable_data(0),
            N,
            /*copy_aux_mem*/ false,
            /*strict*/ true);
    return vec;
}

arma::mat arr_to_mat(py::array_t<double, py::array::f_style> &arr){
    int N = arr.shape()[0];
    int M = arr.shape()[1];

    auto r = arr.mutable_unchecked<2>(); // 2D array
    arma::mat mat = arma::mat(
            r.mutable_data(0, 0),
            N,
            M,
            /*copy_aux_mem*/ false,
            /*strict*/ true);
    return mat;
}

arma::cube arr_to_cube(py::array_t<double, py::array::f_style> &arr){
    int N = arr.shape()[0];
    int M = arr.shape()[1];
    int K = arr.shape()[2];

    auto r = arr.mutable_unchecked<3>(); // 3D array
    arma::cube cube = arma::cube(
            r.mutable_data(0, 0, 0),
            N,
            M,
            K,
            /*copy_aux_mem*/ false,
            /*strict*/ true);
    return cube;
}

py::array_t<double> mat_to_arr(arma::mat &mat){
    /* Strides (in bytes) for each index */
    // Reference (https://github.com/pybind/pybind11/issues/2529)
    py::ssize_t strides[2];
    strides[0] = sizeof(double) * mat.n_cols;
    strides[1] = sizeof(double);
    auto arr = py::array(py::buffer_info(
            mat.memptr(),                              /* data as contiguous array  */
            sizeof(double),                          /* size of one scalar        */
            py::format_descriptor<double>::format(), /* data type                 */
            2,                                      /* number of dimensions      */
            {mat.n_rows, mat.n_cols},       /* shape of the matrix       */
            strides /* strides for each axis     */));

    return arr;
}

py::array_t<double> cube_to_arr(arma::cube &cube){
    /* Strides (in bytes) for each index */
    // Reference (https://github.com/pybind/pybind11/issues/2529)
    py::ssize_t strides[3];
    strides[0] = sizeof(double) * cube.n_cols;
    strides[1] = sizeof(double);
    strides[2] = sizeof(double) * cube.n_slices;
    auto arr = py::array(py::buffer_info(
            cube.memptr(),                              /* data as contiguous array  */
            sizeof(double),                          /* size of one scalar        */
            py::format_descriptor<double>::format(), /* data type                 */
            3,                                      /* number of dimensions      */
            {cube.n_rows, cube.n_cols, cube.n_slices},       /* shape of the matrix       */
            strides /* strides for each axis     */));

    return arr;
}

/* ---------------------------
 *     Python Interface
 * --------------------------*/


// Wrap the sGMRFmix C++ code with Numpy array IO
py::tuple py_sGMRFmix(
        py::array_t<double, py::array::f_style> &train_data,
        int init_K,
        double rho,
        py::array_t<double, py::array::f_style> &m0 ,
        bool do_kmeans = false,
        double pi_threshold = 0.01,
        double lambda0 = 1.0,
        int max_iter = 500,
        double tol = 1e-1,
        bool verbose = false,
        int random_seed = 69){
    if (train_data.ndim() != 2) throw std::runtime_error("Train data must be a 2D Numpy array");

    if (init_K > train_data.shape()[0] ) throw std::runtime_error("init_K must be <= # of rows in train data");

    arma::cube A;
    arma::mat m,g_mat;

    // Convert pybind numpy objects to armadillo objects
    arma::mat X = arr_to_mat(train_data);
    arma::rowvec _m0 = arr_to_rowvec(m0);

    sGMRFmix(X, init_K, rho, _m0, A, m,  g_mat, false, pi_threshold,lambda0, max_iter, tol, verbose, random_seed);

    // Convert armadillo objects back to numpy objects
    auto m_ = mat_to_arr(m);
    auto g_mat_ = mat_to_arr(g_mat);
    auto A_ = cube_to_arr(A);

    return py::make_tuple(A_, m_, g_mat_);
}

py::tuple py_compute_anomaly(
        py::array_t<double, py::array::f_style> &test_data,
        py::array_t<double, py::array::f_style> &A,
        py::array_t<double, py::array::f_style> &m,
        py::array_t<double, py::array::f_style> &g_mat,
        bool verbose = false
        ){

    if (A.ndim() != 3) throw std::runtime_error("A must be a 3D Numpy array");

    arma::mat anomaly_score;

    // Convert pybind wrappers around STL to armadillo objects
    arma::mat X = arr_to_mat(test_data);
    arma::mat _m = arr_to_mat(m);
    arma::mat _g_mat = arr_to_mat(g_mat);
    arma::cube A_ = arr_to_cube(A);
//
//    cout<<arma::size(X)<<endl
//        <<arma::size(_m)<<endl
//        <<arma::size(_g_mat)<<endl
//        <<arma::size(A_)<<endl;
    compute_anomaly_score(X, A_, _m, _g_mat, anomaly_score, verbose);

    auto anomaly = mat_to_arr(anomaly_score);
    return py::make_tuple(anomaly);
}


// wrap as Python module
PYBIND11_MODULE(_sgmrfmix, module) {
module.doc() = "A wrapper module around the sGMRFmix C++ implementation";

module.def("sgmrfmix_fit", &py_sGMRFmix, "Fits a sGMRFmix model for the given data ");
module.def("compute_anomaly", &py_compute_anomaly, "Computes the anomaly score based on the learnt sGMRFmix model");
}