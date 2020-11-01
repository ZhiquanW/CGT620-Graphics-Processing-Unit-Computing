cmake_minimum_required(VERSION 3.17)
#SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(terrain_generator CUDA C CXX  )

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

include_directories(include)
add_executable(terrain_generator main.cu test.cu include/utils.cuh include/kernels.cuh)

set_target_properties(
        terrain_generator
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)