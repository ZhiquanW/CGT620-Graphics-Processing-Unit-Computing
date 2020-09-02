#include <cstdio>
__global__ void my_kernel() {
        printf("Hello from block %i of %i and thread %i \n ", blockIdx.x, blockDim.x, threadIdx.x);

}

int main() {
    my_kernel <<<16, 16 >>> ();
    cudaError_t cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cuda_err));
    return 0;
}

