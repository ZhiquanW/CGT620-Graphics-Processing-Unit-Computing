//
// Created by zhiquan on 10/31/20.
//

#ifndef TERRAIN_GENERATOR_TERRAIN_MANAGER_CUH
#define TERRAIN_GENERATOR_TERRAIN_MANAGER_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "obj.h"
#include "glm/glm.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>    // std::sort

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

using namespace std;
using namespace wow;

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

class terrain_manager {

public:
    std::vector<float> h_terrain_vertices;
    uint height;
    uint width;

    terrain_manager(uint h, uint w) : height(h), width(w) {
        h_terrain_vertices.resize(w * h * 6);
    }

//    void gen_flat() {
//        for (int i = 0; i < this->width; ++i) {
//            for (int j = 0; j < this->height; ++j) {
//                this->h_terrain_vertices.push_back(i);
//                this->h_terrain_vertices.push_back(j);
//                this->h_terrain_vertices.push_back(0.0f);
//                this->h_terrain_vertices.push_back(0.4f);
//                this->h_terrain_vertices.push_back(0.6f);
//                this->h_terrain_vertices.push_back(0.5f);
//            }
//        }
//    }

    void randomize(float max_h) {
        glm::vec3 color_0(0.0f, 0.0f, 0.8f);
        glm::vec3 color_1(0.0f, 0.7, 0.0f);
        for (int i = 0; i < this->h_terrain_vertices.size(); i += 6) {
            float h = i / 6 / this->width;
            float w = i / 6 % this->width;
            this->h_terrain_vertices[i] = w;
            this->h_terrain_vertices[i + 1] = h;
            this->h_terrain_vertices[i + 2] = ((float) rand() / RAND_MAX) * max_h;
            glm::vec3 col = color_0 + (color_1 - color_0) * ((max_h - this->h_terrain_vertices[i + 2]) / max_h);
            this->h_terrain_vertices[i + 3] = col.x;
            this->h_terrain_vertices[i + 4] = col.y;
            this->h_terrain_vertices[i + 5] = col.z;

        }
    }
    __host__ __device__ void d_gen_gaps(uint num, float w, float r) {


    }
    void gap_terrain(uint num, float w, float r) {
//        std::random_device rd;
//        std::mt19937 mt(rd());
//        std::uniform_real_distribution<double> dist(0, height);
//        vector<uint> gap_dis_list(num);
//        for (int i = 0; i < num; ++i) {
//            gap_dis_list[i] = (uint) dist(mt);
//        }
//        sort(gap_dis_list.begin(), gap_dis_list.end());
//
//        const uint GRID_LEN = num;
//        std::size_t data_size = this->h_terrain_vertices.size() * sizeof(float);
//        float *d_vertices;
//        checkCudaErrors(cudaMalloc(&d_vertices, data_size));
//        checkCudaErrors(cudaMemcpy(d_vertices, h_terrain_vertices.data(), data_size, cudaMemcpyHostToDevice));
//        data_size = gap_dis_list.size() * sizeof(float);
//        float *d_gap_info;
//        checkCudaErrors(cudaMalloc(&d_gap_info, data_size));
//        checkCudaErrors(cudaMemcpy(d_gap_info, gap_dis_list.data(), data_size, cudaMemcpyHostToDevice));
//        float *d_results;
//        checkCudaErrors(cudaMalloc(&d_results, data_size));
//        dim3 grid_size(GRID_LEN);
//        dim3 block_size(1);
//        d_gen_gaps <<< grid_size, block_size>>>();

    }



    void export_obj(const string &name) {
        Obj obj;
        Vertex vertex;
        vertex.setNormal(0, 0, 0);
        for (int i = 0; i < this->width - 1; ++i) {
            for (int j = 0; j < this->height - 1; ++j) {
                uint idx_0 = j * this->width + i;
                uint idx_1 = idx_0 + this->width;
                uint idx_2 = idx_0 + 1;
                uint idx_3 = idx_0 + 1;
                uint idx_4 = idx_0 + this->width;
                uint idx_5 = idx_0 + this->width + 1;
                Vec3 v_0(this->h_terrain_vertices[idx_0 * 6], this->h_terrain_vertices[idx_0 * 6 + 1],
                         this->h_terrain_vertices[idx_0 * 6 + 2]);
                Vec3 v_1(this->h_terrain_vertices[idx_1 * 6], this->h_terrain_vertices[idx_1 * 6 + 1],
                         this->h_terrain_vertices[idx_1 * 6 + 2]);
                Vec3 v_2(this->h_terrain_vertices[idx_2 * 6], this->h_terrain_vertices[idx_2 * 6 + 1],
                         this->h_terrain_vertices[idx_2 * 6 + 2]);
                Vec3 v_3(this->h_terrain_vertices[idx_3 * 6], this->h_terrain_vertices[idx_3 * 6 + 1],
                         this->h_terrain_vertices[idx_3 * 6 + 2]);
                Vec3 v_4(this->h_terrain_vertices[idx_4 * 6], this->h_terrain_vertices[idx_4 * 6 + 1],
                         this->h_terrain_vertices[idx_4 * 6 + 2]);
                Vec3 v_5(this->h_terrain_vertices[idx_5 * 6], this->h_terrain_vertices[idx_5 * 6 + 1],
                         this->h_terrain_vertices[idx_5 * 6 + 2]);
                vertex.setPosition(v_0.x, v_0.y, v_0.z);
                obj.appendVertex(vertex);
                vertex.setPosition(v_1.x, v_1.y, v_1.z);
                obj.appendVertex(vertex);
                vertex.setPosition(v_2.x, v_2.y, v_2.z);
                obj.appendVertex(vertex);
                obj.closeFace();
                vertex.setPosition(v_3.x, v_3.y, v_3.z);
                obj.appendVertex(vertex);
                vertex.setPosition(v_4.x, v_4.y, v_4.z);
                obj.appendVertex(vertex);
                vertex.setPosition(v_5.x, v_5.y, v_5.z);
                obj.appendVertex(vertex);
                obj.closeFace();
            }
        }
        obj.output(name);
    }

private:


};


#endif //TERRAIN_GENERATOR_TERRAIN_MANAGER_CUH
