cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda11/bin/nvcc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

project(lab4 CUDA)
#project(lab4 CXX)

set(CMAKE_CUDA_STANDARD 14)
include_directories(includes/)
add_executable(lab4 main.cu)

set_target_properties(
        lab4
        PROPERTIES
        CUDA_SEPARABLE_COMinPILATION ON)