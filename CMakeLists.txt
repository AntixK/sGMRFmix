cmake_minimum_required(VERSION 3.16)
project(sgrmrfmix)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

link_libraries(armadillo lapack blas boost_system)
add_executable(sgrmrfmix src/main.cpp src/headers/glasso.h src/headers/gmrfmix.h src/headers/sgaussmix.h src/headers/dmvnorm.h src/headers/sGMRFmix.h src/lib/termcolor.h src/headers/compute_anomaly.h src/_sgmrfmix.cpp)
