//
// @Author: Zhiquan Wang 
// @Date: 2/16/20.
// @Email: zhiquan.wzq@gmail.com
// Copyright (c) 2020 Zhiquan Wang. All rights reserved.
//
#include "ZWEngine.h"
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc

#include <glm/glm.hpp>
#include "tiny_obj_loader.h"

//#define STB_IMAGE_IMPLEMENTATION
//
//#include "stb_image.h"
#define enable_GPU false
static ZWEngine *self;
GLuint mesh_width = 500;
GLuint mesh_height = 500;
dim3 block(16, 16, 1);
dim3 grid(ceil((float) mesh_width / block.x), ceil((float) mesh_height / block.y), 1);
std::vector<GLfloat> terrain_vertices;
std::vector<GLuint> terrain_indices;
GLuint max_iter = 15000;
GLuint iter = 0;
float cam_dis = 1000.0f;
GLfloat max_height = 1.0f;
GLfloat timer = 0;
bool random_signal = false;
__constant__ float d_fault_info[4];

__global__ void fault_cut_kernel(float3 *ptr, unsigned int width, unsigned int height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    float3 color_0 = make_float3(0.0f, 0.0f, 0.8f);
    float3 color_1 = make_float3(0.0f, 0.7, 0.0f);
    float2 r_pos = make_float2(d_fault_info[0], d_fault_info[1]);
    float2 r_dir = make_float2(d_fault_info[2], d_fault_info[3]);
    // write output vertex
    unsigned int offset = 2 * (x * width + y);
    if (offset + 1 < width * height * 2) {
        float3 pos = ptr[offset];
        float2 pos_2d = make_float2(pos.x, pos.y);
        float cur_dir = (float) (dot(r_dir, pos_2d - r_pos) < 0) * 2.0f - 1.0f;
        ptr[offset] -= make_float3(0.0f, 0.0f, cur_dir * 0.1f);
        ptr[offset + 1] = color_0 + (color_1 - color_0) * (ptr[offset].z * 0.05f + 0.5f);
    }
}

void export_terrain() {
    Obj obj;

    Vertex vertex;
    vertex.setNormal(0, 0, 0);
    for (int i = 0; i < mesh_width - 1; ++i) {
        for (int j = 0; j < mesh_height - 1; ++j) {
            GLuint idx_0 = j * mesh_width + i;
            GLuint idx_1 = idx_0 + mesh_width;
            GLuint idx_2 = idx_0 + 1;
            GLuint idx_3 = idx_0 + 1;
            GLuint idx_4 = idx_0 + mesh_width;
            GLuint idx_5 = idx_0 + mesh_width + 1;
            Vec3 v_0(terrain_vertices[idx_0 * 6], terrain_vertices[idx_0 * 6 + 1], terrain_vertices[idx_0 * 6 + 2]);
            Vec3 v_1(terrain_vertices[idx_1 * 6], terrain_vertices[idx_1 * 6 + 1], terrain_vertices[idx_1 * 6 + 2]);
            Vec3 v_2(terrain_vertices[idx_2 * 6], terrain_vertices[idx_2 * 6 + 1], terrain_vertices[idx_2 * 6 + 2]);
            Vec3 v_3(terrain_vertices[idx_3 * 6], terrain_vertices[idx_3 * 6 + 1], terrain_vertices[idx_3 * 6 + 2]);
            Vec3 v_4(terrain_vertices[idx_4 * 6], terrain_vertices[idx_4 * 6 + 1], terrain_vertices[idx_4 * 6 + 2]);
            Vec3 v_5(terrain_vertices[idx_5 * 6], terrain_vertices[idx_5 * 6 + 1], terrain_vertices[idx_5 * 6 + 2]);
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
    obj.output("test");
}

// Opengl functions
void ZWEngine::set_render_info() {
    glEnable(GL_DEPTH_TEST);
    srand(time(0));
//    std::string inputfile = "test.obj";
//    tinyobj::attrib_t attrib;
//    std::vector<tinyobj::shape_t> shapes;
//    std::vector<tinyobj::material_t> materials;
//
//    std::string warn;
//    std::string err;
//
//    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());
//    if (!warn.empty()) {
//        std::cout << warn << std::endl;
//    }
//    if (!err.empty()) {
//        std::cerr << err << std::endl;
//    }
//    if (!ret) {
//        exit(1);
//    }
//    std::cout <<"shape "  << shapes.size() << std::endl;
//    // Loop over shapes
//    for (size_t s = 0; s < shapes.size(); s++) {
//        // Loop over faces(polygon)
//        size_t index_offset = 0;
//        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
//            int fv = shapes[s].mesh.num_face_vertices[f];
//            // Loop over vertices in the face.
//            for (size_t v = 0; v < fv; v++) {
//                // access to vertex
//                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
//                tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
//                tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
//                tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];
//                std::cout<<" faces " << vx <<  " " << vy << " " << vz << std::endl;
//                tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
//                tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
//                tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];
//                tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
//                tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];
//                // Optional: vertex colors
//                // tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
//                // tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
//                // tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
//            }
//            index_offset += fv;
//
//            // per-face material
//            shapes[s].mesh.material_ids[f];
//        }
//    }

    // Set Render
    self = this;
    Camera main_cam;
    main_camera.set_pos(glm::vec3(0, 0, cam_dis));
    this->attach_camera(main_camera);
    glfwSetFramebufferSizeCallback(this->window, framebuffer_size_callback);
    shader_program->use_shader_program();

    float left_boundary = -(float) mesh_width / 2.0f;
    float low_boundary = -(float) mesh_height / 2.0f;
    for (int i = 0; i < mesh_height; ++i) {
        for (int j = 0; j < mesh_height; ++j) {
            float h = low_boundary + (float) i;
            float w = left_boundary + (float) j;
            terrain_vertices.push_back(w);
            terrain_vertices.push_back(h);
            terrain_vertices.push_back(0.0f);
            terrain_vertices.push_back(0.4f);
            terrain_vertices.push_back(0.6f);
            terrain_vertices.push_back(0.5f);
        }
    }
    for (int i = 0; i < mesh_width - 1; ++i) {
        for (int j = 0; j < mesh_height - 1; ++j) {
            GLuint idx = j * mesh_width + i;
            terrain_indices.push_back(idx);
            terrain_indices.push_back(idx + mesh_width);
            terrain_indices.push_back(idx + 1);
            terrain_indices.push_back(idx + 1);
            terrain_indices.push_back(idx + mesh_width);
            terrain_indices.push_back(idx + mesh_width + 1);
        }
    }

    VertexArrayObject vao(true);
    VertexBufferObject vbo(terrain_vertices, GL_STATIC_DRAW);
    ElementBufferObject ebo(terrain_indices, GL_STATIC_DRAW);
    //pos
    bind_vertex_attribute(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void *) nullptr);
    //col
    bind_vertex_attribute(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void *) (3 * sizeof(GLfloat)));
    vao.attach_vbo(vbo.id());
    vao.attach_ebo(ebo.id());
    this->add_vao("tmp_vao", vao);
    // register this buffer object with CUDA
    checkCudaErrors(
            cudaGraphicsGLRegisterBuffer(&this->cuda_vbo_resource, vbo.id(), cudaGraphicsMapFlagsWriteDiscard));
//    Texture tex_0(0);
//    tex_0.load_image("../resources/test0.jpeg");
//    this->add_texture(tex_0);
//    Texture tex_1(1);
//    tex_1.load_image("../resources/test_image.jpg");
//    this->add_texture(tex_1);
}

void ZWEngine::render_ui() {
    // feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Create a window called "Hello, world!" and append into it.
    ImGui::Begin("Hello, world!");
    if (this->uniform_failed_id != -1) {
        std::string tmp = "uniform variable ";
        tmp += std::to_string(uniform_failed_id);
        tmp += " declare failed";
        ImGui::Text("%s", tmp.c_str());
    }

    ImGui::SliderFloat("obj angle y: ", &obj_angle_y, -180.0f, 180.0f);
    ImGui::SliderFloat("obj angle x: ", &obj_angle_x, -180.0f, 180.0f);
    ImGui::SliderFloat("camera dis:", &cam_dis, 0, 1200.0f);
    ImGui::SliderFloat2("camera angle:", &this->main_camera.get_pitch_yaw()[0], -180, 180);
    ImGui::SliderFloat("terrain maximum height", &max_height, 0.0f, 10.0f);
    ImGui::Text("%s", std::to_string(timer).c_str());
    if (ImGui::Button("random")) {
        random_signal = true;
    }
    if (ImGui::Button("export")) {
        export_terrain();
    }
    ImGui::End();
    ImGui::Render();
}

void ZWEngine::render_world() {
    if (iter < max_iter) {
        float h_fault_info[4];
        h_fault_info[0] = rand() % mesh_width - mesh_width / 2.0f;
        h_fault_info[1] = rand() % mesh_height - mesh_height / 2.0f;
        float2 rand_pos = make_float2(h_fault_info[0], h_fault_info[1]);
        float2 rand_dir = normalize(
                make_float2(rand() % mesh_width - mesh_width / 2.0f, rand() % mesh_height - mesh_height / 2.0f));
        h_fault_info[2] = rand_dir.x;
        h_fault_info[3] = rand_dir.y;
        if (enable_GPU) {
            GLuint fault_info_len = 4 * sizeof(float);
            cudaMemcpyToSymbol(d_fault_info, h_fault_info, fault_info_len); //copy values
            // map OpenGL buffer object for writing from CUDA
            float3 *dptr;
            checkCudaErrors(cudaGraphicsMapResources(1, &this->cuda_vbo_resource, nullptr));
            size_t num_bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &dptr, &num_bytes,
                                                                 this->cuda_vbo_resource));
            fault_cut_kernel<<<grid, block>>>(dptr, mesh_width, mesh_height);
            checkCudaErrors(cudaGraphicsUnmapResources(1, &this->cuda_vbo_resource, nullptr));
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, this->vao_map["tmp_vao"].vbo_list[0]);

            if (random_signal) {
                float *ptr = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
                for (int i = 0; i < terrain_vertices.size(); i += 6) {
                    float3 pos = make_float3(terrain_vertices[i], terrain_vertices[i + 1],
                                             terrain_vertices[i + 2]);

                    terrain_vertices[i] = pos.x;
                    terrain_vertices[i + 1] = pos.y;
                    terrain_vertices[i + 2] = ((GLfloat) rand() / RAND_MAX) * max_height;
                    float3 color_0 = make_float3(0.0f, 0.0f, 0.8f);
                    float3 color_1 = make_float3(0.0f, 0.7, 0.0f);
                    float3 col = color_0 + color_1;//(color_1 - color_0) * ((max_height - terrain_vertices[i + 2]) / max_height);
                    terrain_vertices[i + 3] = col.x;
                    terrain_vertices[i + 4] = col.y;
                    terrain_vertices[i + 5] = col.z;

                }
                if (ptr) {
                    std::copy(terrain_vertices.begin(), terrain_vertices.end(), ptr);
                    glUnmapBuffer(GL_ARRAY_BUFFER);
                }
                random_signal = false;
            }
        }


        iter++;
        if (iter == max_iter) {
            timer = this->last_time;
        }
    }


    // clear buffers
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    main_camera.set_pos(glm::vec3(0, 100, cam_dis));
    glm::mat4 model = glm::rotate(glm::radians(this->obj_angle_y), glm::vec3(0.0f, 1.0f, 0.0f))
                      * glm::rotate(glm::radians(this->obj_angle_x), glm::vec3(1.0f, 0.0f, 0.0f));
    if (!shader_program->set_uniform_mat4fv(2, model)) {
        this->uniform_failed_id = 2;
    }
    glm::mat4 view = this->main_camera.get_view_mat();
    if (!shader_program->set_uniform_mat4fv(3, view)) {
        this->uniform_failed_id = 3;
    }
    glm::mat4 proj = this->main_camera.get_projection_mat();
    if (!shader_program->set_uniform_mat4fv(4, proj)) {
        this->uniform_failed_id = 4;
    }
//    this->activate_texture();
    this->activate_vao("tmp_vao");

    glDrawElements(GL_TRIANGLES, terrain_indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


}

void ZWEngine::process_input() {
    // check 'ESC' is pressed
    if (glfwGetKey(this->window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(this->window, true);
    }
}


void framebuffer_size_callback(GLFWwindow *window, int w, int h) {
    glViewport(0, 0, w, h);
    self->get_camera().set_aspect((GLfloat) w / (GLfloat) h);
}

bool first_in = true;
glm::vec2 pre_mouse_pos;
// callback function


void ZWEngine::keycode_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        self->main_camera.process_keyboard(FORWARD, self->delta_time);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        self->main_camera.process_keyboard(BACKWARD, self->delta_time);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        self->main_camera.process_keyboard(LEFT, self->delta_time);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        self->main_camera.process_keyboard(RIGHT, self->delta_time);
}

void ZWEngine::mouse_callback(GLFWwindow *window, double xpos, double ypos) {
    if (first_in) {
        pre_mouse_pos = glm::vec2(xpos, ypos);
        first_in = false;
    }
    glm::vec2 offset(xpos - pre_mouse_pos.x, pre_mouse_pos.y - ypos);
    pre_mouse_pos = glm::vec2(xpos, ypos);
    self->main_camera.process_mouse_movement(offset);
}

void ZWEngine::scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    self->main_camera.process_mouse_scroll(yoffset);
}



