#include "ZWEngine.h"
#include <iostream>

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

const GLchar *vs_shader_path = "../glsl/vertex_shader.glsl";
const GLchar *fs_shader_path = "../glsl/fragment_shader.glsl";


__global__ void add(int n, float *x, float *y) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
int main() {
    auto *tmp_app = new ZWEngine();

    if (!tmp_app->init_engine(SCR_WIDTH, SCR_HEIGHT)) {
        std::cout << "engine failed to initialize" << std::endl;
    } else {
        std::cout << "engine initialized successfully" << std::endl;
    }
    tmp_app->init_shader_program(vs_shader_path, fs_shader_path);
    std::cout << "engine start running" << std::endl;
    tmp_app->run();
}