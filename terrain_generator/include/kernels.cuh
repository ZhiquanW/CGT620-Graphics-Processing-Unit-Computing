//
// Created by zhiquan on 11/1/20.
//

#ifndef TERRAIN_GENERATOR_KERNELS_CUH
#define TERRAIN_GENERATOR_KERNELS_CUH

__global__ void kernel_gen_gaps(float *d_vertices, const uint *d_gap_info, uint height, uint width, uint n, float r) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_id >= height || col_id >= width) {
        return;
    }
    unsigned int v_offset = 6 * (row_id * width + col_id);
    float tmp_line = 0;
    for (int i = 0; i < n; ++i) {
        tmp_line = (float) d_gap_info[i];
        if (tmp_line - r <= (float) row_id && (float) row_id <= tmp_line + r) {
            d_vertices[v_offset + 2] = -100.0f;
        }
    }
}

__global__ void
kernel_gen_stairs(const float *d_forward_info, const float *d_height_indo, const uint size, float *d_vertices,
                  const uint height, const uint width) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_id >= height || col_id >= width) {
        return;
    }
    unsigned int v_offset = 6 * (row_id * width + col_id);
    for (int i = 0; i < size - 1; ++i) {
        if (d_forward_info[i] < (float) row_id && (float) row_id < d_forward_info[i + 1]) {
            d_vertices[v_offset + 2] = d_height_indo[i];
        }

    }
}

__global__ void
kernel_gen_walls(float *d_vertices, const uint *d_gap_info, const float *d_wall_r, uint height, uint width, uint n) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_id >= height || col_id >= width) {
        return;
    }
    unsigned int v_offset = 6 * (row_id * width + col_id);
    float tmp_line = 0;
    float wall_range = 0;
    for (int i = 0; i < n; ++i) {
        tmp_line = (float) d_gap_info[i];
        wall_range = d_wall_r[i];
        if (tmp_line - wall_range <= (float) row_id && (float) row_id <= tmp_line + wall_range) {
            d_vertices[v_offset + 2] = 1;
        }
    }
}

__global__ void kernel_gen_obstacles(float *d_vertices, int3 *d_loc_list, uint num, uint height, uint width) {
    unsigned int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_id >= height || col_id >= width) {
        return;
    }
    unsigned int v_offset = 6 * (row_id * width + col_id);
    for (int i = 0; i < num; ++i) {
        int2 loc = make_int2(d_loc_list[i].x, d_loc_list[i].y);
        int r = d_loc_list[i].z;
        if (row_id == (uint) loc.x && col_id == (uint) loc.y) {
            uint start_x = row_id - r;
            uint start_y = col_id - r;
            for (int x = start_x; x <= start_x + 2 * r;++ x){
                for(int  y = start_y ; y <= start_y + 2 *r ;++ y){
                    uint tmp_offset = 6 * (x * width + y);
                    d_vertices[tmp_offset+2] = 3;
                }
            }
        }
    }

}

#endif //TERRAIN_GENERATOR_KERNELS_CUH
