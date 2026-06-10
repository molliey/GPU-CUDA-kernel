#include <iostream>
#include <cuda_runtime.h>

#define THREADS 256

// ============================================================
// Naive Reduction
// each thread directly atomicAdd to global result
// problem: high contention on global atomicAdd, especially when N is large
// ============================================================
__global__ void naiveReduction(
    const float* input,
    float* result,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        atomicAdd(result, input[idx]);
    }
}

// ============================================================
// Optimized Reduction
//
// each block does a local reduction in shared memory first
// each block only does one atomicAdd to global result, greatly reducing contention
//
// ============================================================
__global__ void optimizedReduction(
    const float* input,
    float* result,
    int N)
{
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个 thread 先从 global memory 读取一个元素到 shared memory
    if (idx < N)
    {
        sdata[tid] = input[idx];
    }
    else
    {
        sdata[tid] = 0.0f;
    }

    __syncthreads();

    // block reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    // each block's thread 0 does one atomicAdd to global result
    if (tid == 0)
    {
        atomicAdd(result, sdata[0]);
    }
}

// ============================================================
// Kernel Timer
// ============================================================
float measureKernelTime(
    void (*kernel)(const float*, float*, int),
    const float* d_input,
    float* d_result,
    int N,
    int blocks,
    int threads)
{
    cudaMemset(d_result, 0, sizeof(float));

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    kernel<<<blocks, threads>>>(d_input, d_result, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main()
{
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float* h_input = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }

    float h_result = 0.0f;

    float* d_input;
    float* d_result;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice);

    int threads = THREADS;
    int blocks = (N + threads - 1) / threads;

    // warm up
    cudaMemset(d_result, 0, sizeof(float));
    optimizedReduction<<<blocks, threads>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    // test naive
    float naiveTime = measureKernelTime(
        naiveReduction,
        d_input,
        d_result,
        N,
        blocks,
        threads);

    cudaMemcpy(
        &h_result,
        d_result,
        sizeof(float),
        cudaMemcpyDeviceToHost);

    std::cout << "Naive Result = " << h_result << std::endl;
    std::cout << "Naive Time   = " << naiveTime << " ms" << std::endl;

    // test optimized
    float optimizedTime = measureKernelTime(
        optimizedReduction,
        d_input,
        d_result,
        N,
        blocks,
        threads);

    cudaMemcpy(
        &h_result,
        d_result,
        sizeof(float),
        cudaMemcpyDeviceToHost);

    std::cout << "Optimized Result = " << h_result << std::endl;
    std::cout << "Optimized Time   = " << optimizedTime << " ms" << std::endl;

    std::cout
        << "Speedup = "
        << naiveTime / optimizedTime
        << "x"
        << std::endl;

    cudaFree(d_input);
    cudaFree(d_result);

    delete[] h_input;

    return 0;
}