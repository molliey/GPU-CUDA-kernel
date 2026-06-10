#include <iostream>
#include <cuda_runtime.h>

// ============================================================
// Occupancy Experiment
// threads = 64 / 128 / 256 / 512 / 1024
// test how different block sizes affect kernel execution time

// Result:
// 1. occupancy is not the highest better
// 2. blocksize is not the biggest the better
// 3. kernel time is affected by many factors, occupancy is just one of them
// =============================================================

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

float measureTime(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N,
    int threads)
{
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main()
{
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float* h_A = new float[N];
    float* h_B = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 预热
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    int threadOptions[] = {64, 128, 256, 512, 1024};

    for (int i = 0; i < 5; i++)
    {
        int threads = threadOptions[i];
        int blocks = (N + threads - 1) / threads;

        float time = measureTime(
            d_A,
            d_B,
            d_C,
            N,
            threads);

        std::cout
            << "Threads per block: "
            << threads
            << ", Blocks: "
            << blocks
            << ", Time: "
            << time
            << " ms"
            << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;

    return 0;
}

