#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "svpng.inc"
#include <cstring>
#include <algorithm>    // std::min
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION

#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define IMAGE_NUM 1906
#define IMAGE_SIZE 1920*1080*3
#define WIDTH 1920
#define HEIGHT 1080
#define CHANNEL 3
#define MAX_IMAGE_IN_DEVICE 300
#define BLUR_RADIUS 100
#define GRID_LEN 64
#define MODE 1
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
    if (val == (int)( float(pos+1) / float(total)*100)){
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

__global__ void motion_blur_kernel(const unsigned int n, const unsigned char *d_images, unsigned char *d_results,
                                   const unsigned int start, const unsigned int proc_num) {
    unsigned int cuda_start_idx;
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int pixel_offset = CHANNEL * (row_idx * WIDTH + col_idx);
    for (unsigned int i = 0; i < proc_num; ++i) {
        cuda_start_idx = (start + i) % MAX_IMAGE_IN_DEVICE;

        unsigned int last_image_idx = cuda_start_idx + n;
        unsigned int target_image_idx = cuda_start_idx * IMAGE_SIZE;
        unsigned int target_image_offset = target_image_idx + pixel_offset;
        float r = 0;
        float g = 0;
        float b = 0;
        int cuda_copy_idx;
        float start_weight, delta_h;
        if (MODE == 0) {
            start_weight = 1.0f / (float)n;
            delta_h = 0.0f;
        } else if (MODE == 1) {
            start_weight = 2.0f / (float)n;
            delta_h = - 2.0f / (float)n / (float)n;
        }
        for (unsigned int j = cuda_start_idx; j < last_image_idx; ++j) {
            float pos = (float)j - (float)cuda_start_idx;
            cuda_copy_idx = j % MAX_IMAGE_IN_DEVICE;
            unsigned int blur_image_offset = cuda_copy_idx * IMAGE_SIZE;
            unsigned int offset = pixel_offset + blur_image_offset;
            float weight = start_weight +  pos* delta_h;
            if (MODE==2){
                weight = 1.0f/sqrt(2.0f*3.141592654f) * exp(-(1.0f/2.0f)*(4*pos/n*pos/n));
            }
            if (offset < MAX_IMAGE_IN_DEVICE * IMAGE_SIZE - 2) {
                r += weight * (float) d_images[offset];
                r = min(r,255.0f);
                g += weight * (float) d_images[offset + 1];
                g = min(g,255.0f);
                b += weight * (float) d_images[offset + 2];
                b = min(b,255.0f);
            }

        }
        d_results[target_image_offset] = (unsigned char) min((unsigned char) r, 255);
        d_results[target_image_offset + 1] = (unsigned char) min((unsigned char) g, 255);
        d_results[target_image_offset + 2] = (unsigned char) min((unsigned char) b, 255);
    }
}

int main() {


    std::cout << "image num: " << IMAGE_NUM << " in total" << std::endl;
    std::cout << "blur radius " << BLUR_RADIUS << std::endl;
    int epoch = ceil(IMAGE_NUM / MAX_IMAGE_IN_DEVICE);
    auto *h_images = new unsigned char[MAX_IMAGE_IN_DEVICE * IMAGE_SIZE];
    auto *h_results = new unsigned char[MAX_IMAGE_IN_DEVICE * IMAGE_SIZE];
    unsigned char *d_images;
    unsigned char *d_results;
    checkCudaErrors(cudaMalloc(&d_images, IMAGE_SIZE * MAX_IMAGE_IN_DEVICE * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_results, IMAGE_SIZE * MAX_IMAGE_IN_DEVICE * sizeof(unsigned char)));
    checkCudaErrors(cudaMemset(d_results, 90, IMAGE_SIZE * MAX_IMAGE_IN_DEVICE * sizeof(unsigned char)));
    int start_idx = 0;
    int end_idx;
    int occupied_num = 0;
    int image_counter = 0;
    int output_counter = 0;
    int proc_start_idx = 0;
    int proc_num = -BLUR_RADIUS;
    float running_time = 0;
    int e = 0;
    while (image_counter < IMAGE_NUM) {
        printf("==========#############################==========\n");
        printf("epoch: %d\n", e);
        e++;
        // read images to host
        std::cout << "start loading ...\n";
        std::cout << "remained image num: " << IMAGE_NUM - image_counter << std::endl;
        std::cout << "occupied space: " << occupied_num << std::endl;
        unsigned int load_num = std::min(IMAGE_NUM - image_counter, MAX_IMAGE_IN_DEVICE - occupied_num);
        end_idx = (start_idx + load_num) % MAX_IMAGE_IN_DEVICE;

        printf("start at %d\n", start_idx);
        printf("end at %d \n", end_idx);
        #pragma omp parallel
        #pragma omp for
        for (int i = 0; i < load_num; ++i) {
            image_counter++;
            if (image_counter > IMAGE_NUM) {
                break;
            }
            int i_height, i_width, channels;
            std::string file_name =
                    "../resources/images/out" + std::to_string(image_counter) + ".png";
            unsigned char *h_in_image = stbi_load(file_name.c_str(), &i_width, &i_height, &channels, 0);
            if (h_in_image == nullptr && stbi_failure_reason()) {
                std::cout << "stbi_load error msg: ";
                std::cout << stbi_failure_reason() << std::endl;
                std::cout << "reading image failed." << std::endl;
                std::cout << h_in_image << std::endl;
                exit(-1);
            }
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                h_images[start_idx * IMAGE_SIZE + j] = h_in_image[j];
            }
            occupied_num++;
            start_idx++;
            end_idx++;
            start_idx %= MAX_IMAGE_IN_DEVICE;
            end_idx %= MAX_IMAGE_IN_DEVICE;
            progress_indicator(i + 1, load_num, 50);
        }
        proc_num = occupied_num - BLUR_RADIUS;
        std::cout << std::endl;
        std::cout << load_num << " images loaded\n";
        printf("start processing %d images\n", proc_num);
        // Choose which GPU to run on, change this on a multi-GPU system.
        checkCudaErrors(cudaSetDevice(0));
        // copy images from host to device
        checkCudaErrors(cudaMemcpy(d_images, h_images, MAX_IMAGE_IN_DEVICE * IMAGE_SIZE * sizeof(unsigned char),
                                   cudaMemcpyHostToDevice));
        printf("%d occupied before process\n", occupied_num);
        dim3 grid_size(GRID_LEN, GRID_LEN);
        dim3 block_size(ceil((float) HEIGHT / GRID_LEN), ceil((float) WIDTH / GRID_LEN));
        cudaEvent_t startT, stopT;
        float time;
        cudaEventCreate(&startT);
        cudaEventCreate(&stopT);
        cudaEventRecord(startT, 0);
        motion_blur_kernel<<<grid_size, block_size>>>(BLUR_RADIUS, d_images, d_results, proc_start_idx, proc_num);
        cudaEventRecord(stopT, 0);
        cudaEventSynchronize(stopT);
        cudaEventElapsedTime(&time, startT, stopT);
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);
        running_time += time;
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        occupied_num -= proc_num;
        printf("%d occupied after process\n", occupied_num);
        std::cout << ">> " << proc_num << " images processed\n";
        checkCudaErrors(
                cudaMemcpy(h_results, d_results, MAX_IMAGE_IN_DEVICE * IMAGE_SIZE * sizeof(unsigned char),
                           cudaMemcpyDeviceToHost));
        std::cout << "data copied\n";

        printf("release space: %d\n", proc_num);
//        auto tmp_image = new unsigned char[IMAGE_SIZE];
        int result;
        int copy_start_idx;
        #pragma omp parallel
        #pragma omp for
        for (int i = proc_start_idx; i < proc_start_idx + proc_num; ++i) {
            copy_start_idx = i % MAX_IMAGE_IN_DEVICE;
//            memcpy(tmp_image, h_results + copy_start_idx * IMAGE_SIZE, IMAGE_SIZE * sizeof(unsigned char));
            std::string file_name = "/home/zhiquan/Pictures/lab4/result" + std::to_string(++output_counter) + ".png";
            result = stbi_write_png(file_name.c_str(), WIDTH, HEIGHT, CHANNEL, h_results+copy_start_idx * IMAGE_SIZE, 0);
            if (!result) {
                std::cout << "Something went wrong during writing. Invalid path?" << std::endl;
                return 0;
            }
            progress_indicator(output_counter, IMAGE_NUM - BLUR_RADIUS, 50);
        }
//        free(tmp_image);
        proc_start_idx += proc_num;
        proc_start_idx %= MAX_IMAGE_IN_DEVICE;
        printf("\n%d blur image created\n", proc_num);
    }
    printf("running time: %f", running_time);
    free(h_results);
    free(h_images);
    cudaFree(d_results);
    cudaFree(d_images);
    return 0;
}
