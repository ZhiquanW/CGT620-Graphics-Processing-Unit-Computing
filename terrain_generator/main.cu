//
#include "terrain_manager.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

# include <cassert>
# include <iostream>
# include <fstream>
# include <sstream>

int main() {
    uint mesh_width = 2;
    uint mesh_height = 2;
    terrain_manager tm(mesh_height, mesh_width);
    tm.stair_terrain(glm::vec2(1,3),glm::vec2(0.2,1),glm::vec2(0,100));
//    tm.export_obj("test_stair");
    tm.gap_terrain(10, 3,glm::vec2(100,400));
//    tm.export_obj("test_gap");
    tm.wall_terrain(30,glm::vec2(1,2),glm::vec2(200,300));
    tm.obstacle_terrain(1000,glm::vec2(1,3),glm::vec2(100,200));
//    tm.uniforma_randomize(0.2);
    tm.perlin_randomize();
    tm.new_export("test_mixed_perlin");
    return 0;
}



//#include <stdio.h>
//
//class A{
//
//    int data;
//public:
//    A() { data = 0;}
//    __host__ __device__
//    void increment()  { data++;}
//    __host__ __device__
//    void print_data() { printf("data = %d\n", data);}
//};
//
//__global__ void test(A a){
//
//    a.increment();
//    a.print_data();
//}
//
//int main(){
//
//    A h_a;
//    h_a.increment();
//    h_a.print_data();
//    test<<<1,1>>>(h_a);
//    cudaDeviceSynchronize();
//}