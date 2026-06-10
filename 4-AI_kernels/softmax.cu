#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// softmax(x_i) = exp(x_i) / sum(exp(x_j))

__global__ void softmaxKernel(
    const float* input,
    float* output,
    int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // ========================================================
    // Step 1: calculate max(x)
    // ========================================================
    float val = (tid < N) ? input[tid] : -1e20f;

    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }

        __syncthreads();
    }

    float maxVal = sdata[0];

    __syncthreads();

    // ========================================================
    // Step 2: calculate exp(x_i - max)
    // ========================================================
    float expVal = 0.0f;

    if (tid < N)
    {
        expVal = expf(input[tid] - maxVal);
        output[tid] = expVal;
    }

    sdata[tid] = expVal;
    __syncthreads();

    // ========================================================
    // Step 3: reduction calculate sum(exp)
    // ========================================================
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    float sumExp = sdata[0];

    // ========================================================
    // Step 4: normalize
    // ========================================================
    if (tid < N)
    {
        output[tid] = output[tid] / sumExp;
    }
}

int main()
{
    const int N = 8;
    size_t bytes = N * sizeof(float);

    float h_input[N] =
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    float h_output[N];

    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(
        d_input,
        h_input,
        bytes,
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = 1;

    size_t sharedMemBytes = threads * sizeof(float);

    softmaxKernel<<<blocks, threads, sharedMemBytes>>>(
        d_input,
        d_output,
        N);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost);

    std::cout << "Softmax Output\n";

    float sum = 0.0f;

    for (int i = 0; i < N; i++)
    {
        sum += h_output[i];

        std::cout
            << h_input[i]
            << " -> "
            << h_output[i]
            << std::endl;
    }

    std::cout << "Sum = " << sum << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}