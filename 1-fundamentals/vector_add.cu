#include <iostream>
#include <cuda_runtime.h>

// kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // thread mapping
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    // host memory
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // HOST TO DEVICE
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // launch configuration
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // kernel launch
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // DEVICE TO HOST
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // verify results
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << std::endl;
    }

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}



/*

1. Host Memory
   h_A h_B h_C

2. Device Memory
   d_A d_B d_C

3. cudaMemcpy
   Host ↔ Device

4. Kernel Launch
   <<<blocks, threads>>>

5. Thread Mapping
   idx = blockIdx.x * blockDim.x + threadIdx.x


CPU VERSION

for(int i=0;i<N;i++)
{
    C[i] = A[i] + B[i];
}

A[0]+B[0]
A[1]+B[1]
A[2]+B[2]
...
A[1023]+B[1023]


*/

