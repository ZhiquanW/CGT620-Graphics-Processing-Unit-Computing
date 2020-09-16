cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(lab3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab3 unified.cu)

set_target_properties(
        lab3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


