//
// @Author: Zhiquan Wang 
// @Date: 2/16/20.
// @Email: zhiquan.wzq@gmail.com
// Copyright (c) 2020 Zhiquan Wang. All rights reserved.
//
#include "ZWEngine.h"

//#define STB_IMAGE_IMPLEMENTATION
//
//#include "stb_image.h"
static ZWEngine *self;
GLuint mesh_width =100;
GLuint mesh_height = 100;
dim3 block(16, 16, 1);
dim3 grid(ceil((float) mesh_width / block.x), ceil((float) mesh_height / block.y), 1);
std::vector<GLfloat> terrain_vertices;
std::vector<GLuint> terrain_indices;
GLuint max_iter = 1000;
GLuint iter = 0;
float cam_dis = 100.0f;
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
        float cur_dir = (float)(dot(r_dir, pos_2d - r_pos) < 0) * 2.0f - 1.0f;
        ptr[offset] -= make_float3(0.0f, 0.0f,cur_dir* 0.2f);
        ptr[offset + 1] = color_0 + (color_1 - color_0) * (ptr[offset].z * 0.05f + 0.5f);
    }
}

// Opengl functions
void ZWEngine::set_render_info() {
    glEnable(GL_DEPTH_TEST);
    srand(time(0));

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
    ImGui::SliderFloat("camera dis:",&cam_dis,0,400.0f);
    ImGui::SliderFloat2("camera angle:", &this->main_camera.get_pitch_yaw()[0], -180, 180);
    ImGui::End();
    ImGui::Render();
}

void ZWEngine::render_world() {
    if (iter < max_iter) {
        float h_fault_info[4];
        h_fault_info[0] = rand() % mesh_width - mesh_width / 2.0f;
        h_fault_info[1] = rand() % mesh_height - mesh_height / 2.0f;
        float2 rand_dir = normalize(
                make_float2(rand() % mesh_width - mesh_width / 2.0f, rand() % mesh_height - mesh_height / 2.0f));
        h_fault_info[2] = rand_dir.x;
        h_fault_info[3] = rand_dir.y;
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
        iter++;
    }


    // clear buffers
    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    main_camera.set_pos(glm::vec3(0, 0, cam_dis));
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



