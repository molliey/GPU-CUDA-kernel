#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int idx =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if(idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    const int N = 1 << 24;

    size_t bytes =
        N * sizeof(float);

    float *h_A = new float[N];
    float *h_B = new float[N];

    for(int i=0;i<N;i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(
        d_A,
        h_A,
        bytes,
        cudaMemcpyHostToDevice);

    cudaMemcpy(
        d_B,
        h_B,
        bytes,
        cudaMemcpyHostToDevice);

    int threads = 256;

    int blocks =
        (N + threads - 1)
        / threads;

    // ====================================
    // create CUDA Events
    // ====================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ====================================
    // start recording
    // ====================================

    cudaEventRecord(start);

    vectorAdd<<<blocks,threads>>>(
        d_A,
        d_B,
        d_C,
        N);

    // ====================================
    // end recording
    // ====================================

    cudaEventRecord(stop);

    // wait for the event to complete
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    std::cout
        << "Kernel Time: "
        << milliseconds
        << " ms"
        << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;

    return 0;
}