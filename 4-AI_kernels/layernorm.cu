#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define THREADS 256

// =========================================================
// Reduction + Element-wise + Numerical Stability
// =========================================================

__global__ void layerNormKernel(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                int rows,
                                int cols,
                                float eps) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;

    // ========================================================
    // Step 1: reduction calculate sum(x)
    // ========================================================
    float x = (tid < N) ? input[tid] : 0.0f;

    sdata[tid] = x;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    float mean = sdata[0] / N;

    __syncthreads();

    // ========================================================
    // Step 2: reduction calculate variance
    // ========================================================
    float diff = 0.0f;

    if (tid < N)
    {
        diff = input[tid] - mean;
    }

    sdata[tid] = diff * diff;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();
    }

    float variance = sdata[0] / N;

    __syncthreads();

    // ========================================================
    // Step 3: element-wise normalize
    // ========================================================
    if (tid < N)
    {
        output[tid] =
            gamma[tid] *
            (input[tid] - mean) /
            sqrtf(variance + eps)
            + beta[tid];
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

    float h_gamma[N];
    float h_beta[N];
    float h_output[N];

    for (int i = 0; i < N; i++)
    {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    float* d_input;
    float* d_gamma;
    float* d_beta;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_gamma, bytes);
    cudaMalloc(&d_beta, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, bytes, cudaMemcpyHostToDevice);

    layerNormKernel<<<1, THREADS>>>(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        N,
        1e-5f);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    std::cout << "LayerNorm Output\n";

    for (int i = 0; i < N; i++)
    {
        std::cout
            << h_input[i]
            << " -> "
            << h_output[i]
            << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);

    return 0;
}