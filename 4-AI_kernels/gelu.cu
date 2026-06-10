#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
// ============================================================

__global__ void geluKernel(
    const float* input,
    float* output,
    int N)
{
    // calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = input[idx];

        // sqrt(2/pi) ≈ 0.7978845608
        float c = 0.7978845608f;

        // x^3
        float x3 = x * x * x;

        // tanh argument
        float inner = c * (x + 0.044715f * x3);

        // GELU formula
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

int main()
{
    const int N = 10;
    size_t bytes = N * sizeof(float);

    float h_input[N] =
    {
        -3.0f,
        -2.0f,
        -1.0f,
         0.0f,
         1.0f,
         2.0f,
         3.0f,
         4.0f,
         5.0f,
         6.0f
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
    int blocks = (N + threads - 1) / threads;

    geluKernel<<<blocks, threads>>>(
        d_input,
        d_output,
        N);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output,
        d_output,
        bytes,
        cudaMemcpyDeviceToHost);

    std::cout << "GELU Output\n";

    for (int i = 0; i < N; i++)
    {
        std::cout
            << h_input[i]
            << " -> "
            << h_output[i]
            << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}