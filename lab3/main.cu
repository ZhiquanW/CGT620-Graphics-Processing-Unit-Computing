#include <iostream>
#include <math.h> 
// #define UNIFIED
#ifdef UNIFIED
// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main() {
  int N = pow(2, 20);
  float *x, *y;
  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  int blockSize = 32;
  int numBlocks = 128;
  cudaEvent_t startT, stopT;
  float time;
  cudaEventCreate(&startT);
  cudaEventCreate(&stopT);
  cudaEventRecord(startT, 0);
  add<<<numBlocks, blockSize>>>(N, x, y);
  cudaEventRecord(stopT, 0);
  cudaEventSynchronize(stopT);
  cudaEventElapsedTime(&time, startT, stopT);
  cudaEventDestroy(startT);
  cudaEventDestroy(stopT);
  std::cout << "cuda function :" << time << " ms" << std::endl;
  // Run kernel on 1M elements on the GPU

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}

#else
__global__ void add(int n, const float *x, float *y) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  uint stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main() {

  const int N = pow(2, 20);
  float x[N];
  float y[N];
  // initialize x and y arrays on the host
  for (uint i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  float *d_x;
  float *d_y;
  cudaError_t cudaStatus;
  // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus = cudaMalloc((void **)&d_x, N * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }
  cudaStatus = cudaMemcpy(d_x, x, N * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }
  cudaStatus = cudaMalloc((void **)&d_y, N * sizeof(int));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
  }

  cudaStatus = cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  int blockSize = 32;
  int numBlocks = 128;
  cudaEvent_t startT, stopT;
  float time;
  cudaEventCreate(&startT);
  cudaEventCreate(&stopT);
  cudaEventRecord(startT, 0);
  add<<<numBlocks, blockSize>>>(N, d_x, d_y);
  cudaEventRecord(stopT, 0);
  cudaEventSynchronize(stopT);
  cudaEventElapsedTime(&time, startT, stopT);
  cudaEventDestroy(startT);
  cudaEventDestroy(stopT);
  std::cout << "cuda function :" << time << " ms" << std::endl;
  // Run kernel on 1M elements on the GPU

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "max error: " << maxError << std::endl;

  // Free memory
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;
}

#endif