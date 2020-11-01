//
#include "terrain_manager.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int main() {
    uint mesh_width = 500;
    uint mesh_height = 500;
    terrain_manager tm(mesh_height, mesh_width);
//    tm.stair_terrain(glm::vec2(1,3),glm::vec2(0.2,1));
//    tm.export_obj("test_stair");
//    tm.gap_terrain(50, 3);
//    tm.export_obj("test_gap");
//    tm.wall_terrain(30,glm::vec2(1,2));
//    tm.export_obj("test_walls");
    tm.obstacle_terrain(100,glm::vec2(1,3));
    tm.export_obj("test");
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