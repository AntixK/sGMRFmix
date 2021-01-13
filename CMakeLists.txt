cmake_minimum_required(VERSION 3.16)
project(sgrmrfmix)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

link_libraries(armadillo lapack blas boost_system)
add_executable(sgrmrfmix src/main.cpp src/sgmrfmix/glasso.h src/sgmrfmix/gmrfmix.h src/sgmrfmix/sgaussmix.h src/sgmrfmix/dmvnorm.h src/sgmrfmix/sGMRFmix.h src/lib/termcolor.h src/sgmrfmix/compute_anomaly.h src/_sgmrfmix.cpp)
