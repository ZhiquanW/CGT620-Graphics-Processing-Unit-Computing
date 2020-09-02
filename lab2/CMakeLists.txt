cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(lab2 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(lab2 main.cu)

set_target_properties(
        lab2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)