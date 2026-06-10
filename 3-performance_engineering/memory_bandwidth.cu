#include <iostream>
#include <cuda_runtime.h>

// ===================================================
// Bandwidth = Total Bytes Read/Written / Kernel Time
// ===================================================

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

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up
     vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
     cudaDeviceSynchronize();

     // start recording
     cudaEventRecord(start);

     vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

     // end recording
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);

     float milliseconds = 0.0f;
     cudaEventElapsedTime(&milliseconds, start, stop);
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // vector add:
    // read A: N 个 float
    // read B: N 个 float
    // write C: N 个 float
    // totalBytes = 3 * N * sizeof(float)
    double totalBytes = 3.0 * bytes;

    // milliseconds / 1000 = seconds
    double seconds = milliseconds / 1000.0;

    // GB/s
    double bandwidth = totalBytes / seconds / 1e9;

    std::cout << "Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Effective Bandwidth: " << bandwidth << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;

    return 0;
}