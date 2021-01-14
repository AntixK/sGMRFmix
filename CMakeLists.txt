cmake_minimum_required(VERSION 3.16)
project(sgrmrfmix)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Boost REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

link_libraries(armadillo lapack blas boost_system)

add_subdirectory(pybind11)

set(SOURCE_FILES
    src/headers/compute_anomaly.h
    src/headers/dmvnorm.h
    src/headers/glasso.h
    src/headers/gmrfmix.h
    src/headers/sgaussmix.h
    src/headers/sGMRFmix.h
    src/lib/termcolor.h)

#add_subdirectory(src)
add_executable(sgrmrfmix src/test.cpp src/_sgmrfmix.cpp ${SOURCE_FILES})
#add_executable(sgrmrfmix  src/headers/glasso.h src/headers/gmrfmix.h src/headers/sgaussmix.h src/headers/dmvnorm.h src/headers/sGMRFmix.h src/lib/termcolor.h src/headers/compute_anomaly.h src/_sgmrfmix.cpp)
