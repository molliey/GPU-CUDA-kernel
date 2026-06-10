#include <iostream>
#include <cuda_runtime.h>

__global__ void gridStrideAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 128;

    gridStrideAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "C[100] = " << h_C[100] << std::endl;
    std::cout << "Expected = " << h_A[100] + h_B[100] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

/*

one thread process multiple elements with a stride of blockDim.x * gridDim.x

grid-stride loop

for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < N;
     idx += blockDim.x * gridDim.x)
{
    C[idx] = A[idx] + B[idx];
}


*/