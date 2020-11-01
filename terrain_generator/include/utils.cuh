//
// Created by zhiquan on 11/1/20.
//

#ifndef TERRAIN_GENERATOR_UTILS_CUH
#define TERRAIN_GENERATOR_UTILS_CUH


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "obj.h"
#include "glm/glm.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>    // std::sort

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(result));
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '"
                  << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void progress_indicator(const int pos, const int total, const int bar_width) {
    float progress = float(pos) / float(total);

    int val = (int) (progress * 100);
    if (val == (int) (float(pos + 1) / float(total) * 100)) {
        return;
    }
    int lpad = (int) (progress * float(bar_width));
    int rpad = bar_width - lpad;
    std::string pre_str;
    for (int i = 0; i < bar_width; ++i) {
        pre_str += "|";
    }
    printf("\r%3d%% [%.*s%*s] (%.d/%.d)", val, lpad, pre_str.c_str(), rpad, "", pos, total);
    fflush(stdout);
}
#endif //TERRAIN_GENERATOR_UTILS_CUH
