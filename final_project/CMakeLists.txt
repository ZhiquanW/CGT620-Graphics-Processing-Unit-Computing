cmake_minimum_required(VERSION 3.17)
project(final_project CUDA C CXX )
set(CMAKE_CUDA_STANDARD 14)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
include_directories(include)
add_subdirectory(glfw-3.3.2)
add_subdirectory(glm)
add_executable(final_project main.cu glad.c)
find_package(OpenGL REQUIRED)

set_target_properties(
        final_project
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(final_project glfw GL)
