#include "ZWEngine.h"
#include <iostream>
#include "obj.h"

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

Obj create_tri(Vec3 startPoint, float len) {
    Obj obj;

    Vertex vertex;
    vertex.setNormal(0, 0, 0);

    vertex.setPosition(startPoint.x, startPoint.y, 0);
    vertex.setTexCoord(0, 0);
    obj.appendVertex(vertex);

    vertex.setPosition(startPoint.x + len, startPoint.y, 0);
    vertex.setTexCoord(1, 0);
    obj.appendVertex(vertex);

    vertex.setPosition(startPoint.x + len, startPoint.y + len, 0);
    vertex.setTexCoord(1, 1);
    obj.appendVertex(vertex);

    vertex.setPosition(startPoint.x, startPoint.y + len, 0);
    vertex.setTexCoord(0, 1);
    obj.appendVertex(vertex);

    obj.closeFace();

    return obj;
}

int main() {
//    const int terrain_width = 100;
//    const int terrain_height = 100;
//    const float terrain_density = 100;
//    std::vector<std::vector<int>> twoDimVector(3, std::vector<int>(2, 0));
//
//    std::vector<std::vector<float3>> terrain_vertex(0);
//    for (int i = 0; i < 10; ++i) {
//        std::cout << terrain_vertex[i].x << std::endl;
//    }

//    Obj obj = createQuad(Vec3(0,0,0), 100);
//    obj.enableTextureCoordinates();
//    obj.enableNormal();

//    obj.output("quad");
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