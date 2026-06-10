
#include <iostream>
#include <cuda_runtime.h>

__global__ void sharedMemoryReduction(const float* input, float* partialSums, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx < N) {
        shared[tid] = input[globalIdx];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = shared[0];
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_input = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* h_partialSums = new float[blocks];

    float *d_input, *d_partialSums;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partialSums, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    sharedMemoryReduction<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partialSums, N);

    cudaMemcpy(h_partialSums, d_partialSums, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;

    for (int i = 0; i < blocks; i++) {
        finalSum += h_partialSums[i];
    }

    std::cout << "Shared memory reduction sum = " << finalSum << std::endl;
    std::cout << "Expected sum = " << N << std::endl;

    cudaFree(d_input);
    cudaFree(d_partialSums);

    delete[] h_input;
    delete[] h_partialSums;

    return 0;
}