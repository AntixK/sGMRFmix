//#include <pybind11/pybind11.h>
//#include <armadillo>
//
//
///* ---------------------------
// *       C++ Code
// * --------------------------*/
//
//
//
///* ---------------------------
// *     Python Interface
// * --------------------------*/
//
//namespace py = pybind11;
//
//
//
//// wrap as Python module
//PYBIND11_MODULE(_sgmrfmix, m) {
//m.doc() = "A wrapper module around the lockfreepc C++ implementation";
//
//m.def("skeleton", &py_skeleton, "Calculate the skeleton for a given np.Array");
//}