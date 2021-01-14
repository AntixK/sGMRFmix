#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <armadillo>
#include "headers/sGMRFmix.h"


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
        std::optional<py::array_t<double, py::array::f_style>> m0 ,
        bool do_kmeans = false,
        double pi_threshold = 0.01,
        double lambda0 = 1.0,
        int max_iter = 500,
        double tol = 1e-1,
        bool verbose = false,
        int random_seed = 69){

    if (train_data.ndim() != 2) throw std::runtime_error("Train data must be a 2D Numpy array");

    if (init_K > train_data.size(0) ) throw std::runtime_error("init_K must be <= # of rows in train data");

    arma::mat A, m,g_mat;

    sGMRFmix(train_data, init_K, rho, m0, A, m,  g_mat, false, pi_threshold,lambda0, max_iter, tol, verbose, random_seed);

    // Return K, A, m, g_mat
    auto results = py::dict();

    results[0] = init_K;
    results[1] = A;
    results[2] = m;
    retults[3] = g_mat;

    return results;
}


// wrap as Python module
PYBIND11_MODULE(_sgmrfmix, module) {
module.doc() = "A wrapper module around the sGMRFmix C++ implementation";

module.def("sGMRFmix", &py_sGMRFmix, "Calculate the skeleton for a given np.Array");
}