#include <iostream>
__global__ void add(int n, float *x, float *y) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
