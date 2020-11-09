//
// Created by zhiquan on 10/31/20.
//

#ifndef TERRAIN_GENERATOR_TERRAIN_MANAGER_CUH
#define TERRAIN_GENERATOR_TERRAIN_MANAGER_CUH

#include "utils.cuh"
#include "kernels.cuh"
#include "PerlinNoise.cuh"
#include <cmath>
using namespace std;
using namespace wow;

#include <math.h>
class terrain_manager {

public:
    std::vector<float> h_terrain_vertices;
    uint height;
    uint width;
    const uint GRID_LEN;
    dim3 grid_size;
    dim3 block_size;

    terrain_manager(uint h, uint w) : height(h),
                                      width(w),
                                      GRID_LEN(64),
                                      grid_size(GRID_LEN,GRID_LEN),
                                      block_size((uint) ceil((float) this->height / GRID_LEN), (uint)ceil(
                                              (float) this->width / GRID_LEN)){
//        h_terrain_vertices.resize(w * h * 6);

        this->init_flat();
    }

    void init_flat() {
        for (int i = 0; i < this->width; ++i) {
            for (int j = 0; j < this->height; ++j) {
                this->h_terrain_vertices.push_back(i);
                this->h_terrain_vertices.push_back(j);
                this->h_terrain_vertices.push_back(0.0f);
                this->h_terrain_vertices.push_back(0.4f);
                this->h_terrain_vertices.push_back(0.6f);
                this->h_terrain_vertices.push_back(0.5f);
            }
        }
    }

    void uniforma_randomize(float max_h) {
        for (int i = 0; i < this->h_terrain_vertices.size(); i += 6) {
            float h = i / 6 / this->width;
            float w = i / 6 % this->width;
            this->h_terrain_vertices[i + 2] += ((float) rand() / RAND_MAX) * max_h;

        }
    }

    void perlin_randomize(){
        unsigned int seed = 237;
        PerlinNoise pn(seed);
        for (int i = 0; i < this->width; ++i) {
            for (int j = 0; j < this->height; ++j) {
                uint offset= (i * height + j) * 6;
                double x = (double)j/((double)this->width);
                double y = (double)i/((double)this->height);
                double n = pn.noise( 100*x, 100*y, 0.8);
                std::cout << offset << " " << n << std::endl;
                this->h_terrain_vertices[offset+2] += n;
            }
        }
    }


    void gap_terrain(uint num, float r,glm::vec2 range) {
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(range.x, range.y);
        vector<uint> gap_dis_list(num);
        for (int i = 0; i < num; ++i) {
            gap_dis_list[i] = (uint) dist(mt);
        }

        sort(gap_dis_list.begin(), gap_dis_list.end());
        std::size_t data_size = this->h_terrain_vertices.size() * sizeof(float);
        float *d_vertices;
        checkCudaErrors(cudaMalloc(&d_vertices, data_size));
        checkCudaErrors(cudaMemcpy(d_vertices, h_terrain_vertices.data(), data_size, cudaMemcpyHostToDevice));
        data_size = gap_dis_list.size() * sizeof(uint);
        uint *d_gap_info;
        checkCudaErrors(cudaMalloc(&d_gap_info, data_size));
        checkCudaErrors(cudaMemcpy(d_gap_info, gap_dis_list.data(), data_size, cudaMemcpyHostToDevice));
        float *h_results = new float[this->h_terrain_vertices.size() * sizeof(float)];
//        grid_size = dim3(GRID_LEN, GRID_LEN);
//        block_size = dim3((uint) ceil((float) this->height / GRID_LEN), (uint)ceil(
//                (float) this->width / GRID_LEN));
        kernel_gen_gaps <<< grid_size, block_size>>>(d_vertices, d_gap_info, this->height, this->width, num, r);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaMemcpy(h_results, d_vertices, this->h_terrain_vertices.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int i = 0; i < this->h_terrain_vertices.size(); ++i) {
            h_terrain_vertices[i] = h_results[i];
        }
        free(h_results);
        cudaFree(d_vertices);
        cudaFree(d_gap_info);
    }

    void stair_terrain(glm::vec2 f_range, glm::vec2 stair_h_range,glm::vec2 range) {
        vector<float> h_forward_info;
        vector<float> h_height_info;
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<float> forward_dist(f_range.x, f_range.y);
        std::uniform_real_distribution<float> h_dist(stair_h_range.x, stair_h_range.y);
        h_forward_info.emplace_back(range.x);
        h_height_info.emplace_back(0.0f);
        float end_lin = min((float)this->height,range.y);
        while (true) {
            float new_forward = h_forward_info.back() + forward_dist(mt);
            float new_stair_height = h_height_info.back() + h_dist(mt);
            if (new_forward > end_lin) {
                new_forward = this->height;
                h_forward_info.emplace_back(new_forward);
                h_height_info.emplace_back(new_stair_height);
                break;
            }
            h_forward_info.emplace_back(new_forward);
            h_height_info.emplace_back(new_stair_height);
        }
        std::size_t data_size = this->h_terrain_vertices.size() * sizeof(float);
        float *d_vertices;
        checkCudaErrors(cudaMalloc(&d_vertices, data_size));
        checkCudaErrors(cudaMemcpy(d_vertices, h_terrain_vertices.data(), data_size, cudaMemcpyHostToDevice));
        data_size = h_forward_info.size() * sizeof(float);
        float *d_forward_info;
        checkCudaErrors(cudaMalloc(&d_forward_info, data_size));
        checkCudaErrors(cudaMemcpy(d_forward_info, h_forward_info.data(), data_size, cudaMemcpyHostToDevice));
        data_size = h_height_info.size() * sizeof(float);
        float *d_height_info;
        checkCudaErrors(cudaMalloc(&d_height_info, data_size));
        checkCudaErrors(cudaMemcpy(d_height_info, h_height_info.data(), data_size, cudaMemcpyHostToDevice));
        kernel_gen_stairs<<<grid_size, block_size>>>(d_forward_info, d_height_info, h_forward_info.size(), d_vertices,
                                                     this->height, this->width);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        float *h_results = new float[this->h_terrain_vertices.size() * sizeof(float)];
        checkCudaErrors(cudaMemcpy(h_results, d_vertices, this->h_terrain_vertices.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int i = 0; i < this->h_terrain_vertices.size(); ++i) {
            h_terrain_vertices[i] = h_results[i];
        }
        free(h_results);
        cudaFree(d_vertices);
        cudaFree(d_forward_info);
        cudaFree(d_height_info);

    }

    void wall_terrain(uint num, glm::vec2 r,glm::vec2 range){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> dist(range.x, range.y);
        std::uniform_real_distribution<double> r_dist(r.x, r.y);
        vector<uint> gap_dis_list(num);
        vector<float> wall_r(num);
        for (int i = 0; i < num; ++i) {
            gap_dis_list[i] = (uint) dist(mt);
            wall_r[i] = r_dist(mt);
        }
        sort(gap_dis_list.begin(), gap_dis_list.end());
        std::size_t data_size = this->h_terrain_vertices.size() * sizeof(float);
        float *d_vertices;
        checkCudaErrors(cudaMalloc(&d_vertices, data_size));
        checkCudaErrors(cudaMemcpy(d_vertices, h_terrain_vertices.data(), data_size, cudaMemcpyHostToDevice));
        data_size = gap_dis_list.size() * sizeof(uint);
        uint *d_gap_info;
        checkCudaErrors(cudaMalloc(&d_gap_info, data_size));
        checkCudaErrors(cudaMemcpy(d_gap_info, gap_dis_list.data(), data_size, cudaMemcpyHostToDevice));
        float * d_wall_r;
        checkCudaErrors(cudaMalloc(&d_wall_r,data_size));
        checkCudaErrors(cudaMemcpy(d_wall_r,wall_r.data(),data_size,cudaMemcpyHostToDevice));
//        grid_size = dim3(GRID_LEN, GRID_LEN);
//        block_size = dim3((uint) ceil((float) this->height / GRID_LEN), (uint)ceil(
//                (float) this->width / GRID_LEN));
        kernel_gen_walls <<< grid_size, block_size>>>(d_vertices, d_gap_info, d_wall_r,this->height, this->width, num);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        auto *h_results = new float[this->h_terrain_vertices.size() * sizeof(float)];
        checkCudaErrors(cudaMemcpy(h_results, d_vertices, this->h_terrain_vertices.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int i = 0; i < this->h_terrain_vertices.size(); ++i) {
            h_terrain_vertices[i] = h_results[i];
        }
        free(h_results);
        cudaFree(d_vertices);
        cudaFree(d_gap_info);
    }
    void obstacle_terrain(uint num,glm::vec2 r,glm::vec2 range){
        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<double> x_dist(range.x, range.y);
        std::uniform_real_distribution<double> y_dist(0,width);
        std::uniform_real_distribution<double> r_dist(r.x, r.y);
        vector<int3> loc_list(num);
        for(auto & i : loc_list){
            i = make_int3(x_dist(mt),(uint)y_dist(mt),r_dist(mt));
        }
        int3 * d_loc_list;
        checkCudaErrors(cudaMalloc(&d_loc_list,loc_list.size() * sizeof(float3)));
        checkCudaErrors(cudaMemcpy(d_loc_list,loc_list.data(),loc_list.size()*sizeof(float3),cudaMemcpyHostToDevice));
        std::size_t data_size = this->h_terrain_vertices.size() * sizeof(float);
        float *d_vertices;
        checkCudaErrors(cudaMalloc(&d_vertices, data_size));
        checkCudaErrors(cudaMemcpy(d_vertices, h_terrain_vertices.data(), data_size, cudaMemcpyHostToDevice));
        kernel_gen_obstacles<<<grid_size,block_size>>>(d_vertices,d_loc_list,loc_list.size(),height,width);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        auto *h_results = new float[this->h_terrain_vertices.size() * sizeof(float)];
        checkCudaErrors(cudaMemcpy(h_results, d_vertices, this->h_terrain_vertices.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
        for (int i = 0; i < this->h_terrain_vertices.size(); ++i) {
            h_terrain_vertices[i] = h_results[i];
        }
        free(h_results);
        cudaFree(d_vertices);
        cudaFree(d_loc_list);
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
