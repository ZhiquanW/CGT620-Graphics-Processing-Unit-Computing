
#include "terrain_manager.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__ void d_gen_gaps() {


}

int main() {
    uint mesh_width = 500;
    uint mesh_height = 500;
    terrain_manager tm(mesh_height, mesh_width);
//    tm.gap_terrain(20,3,2);
    //    tm.randomize(0.4f);
//    tm.export_obj("test_3");
    d_gen_gaps<<< 1 , 1 >>>();
    return 0;
}
