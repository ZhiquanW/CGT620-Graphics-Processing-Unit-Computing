cmake_minimum_required(VERSION 3.10)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(${ProjectId} LANGUAGES CUDA C CXX )
set(CMAKE_CXX_STANDARD 14)

#find_package(imgui CONFIG)

include_directories(include)
include_directories(/usr/local/cuda/include)
include_directories(/usr/local/cuda/samples/common/inc)
include_directories(/usr/local/cuda/)
aux_source_directory(src source_files)
aux_source_directory(imgui imgui_files)

add_executable(${PROJECT_NAME} ${imgui_files} ${source_files})
set_target_properties(
        ${PROJECT_NAME}
        PROPERTIES
        CUDA_SEPARABLE_COMinPILATION ON)
target_link_libraries(${PROJECT_NAME} glfw3 GL Xi X11 pthread dl assimp)

