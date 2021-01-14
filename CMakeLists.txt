# CMakeLists.txt -- Build system for the sGMRFmix modules
#
# Copyright (c) 2021 Anand K Subramanian <anandkrish894@gmail.com>
#
# All rights reserved. Use of this source code is governed by the
# MIT license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.16)
project(sgrmrfmix)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs 3 REQUIRED)
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

include_directories("${PROJECT_SOURCE_DIR}/")
#add_subdirectory(src)
pybind11_add_module(_sgmrfmix src/_sgmrfmix.cpp ${SOURCE_FILES})

# Only include CPP file that has the main() function
add_executable(sgrmrfmix src/test.cpp ${SOURCE_FILES})
#target_include_directories(sgrmrfmix PRIVATE ${PYTHON_INCLUDE_DIRS} src/)
target_link_libraries(_sgmrfmix PRIVATE)

#add_executable(sgrmrfmix  src/headers/glasso.h src/headers/gmrfmix.h src/headers/sgaussmix.h src/headers/dmvnorm.h src/headers/sGMRFmix.h src/lib/termcolor.h src/headers/compute_anomaly.h src/_sgmrfmix.cpp)
