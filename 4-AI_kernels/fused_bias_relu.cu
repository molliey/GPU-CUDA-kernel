#include <iostream>
#include <cuda_runtime.h>

// =========================================================
// Fused Bias + ReLU Kernel
// output[i] = max(0, input[i] + bias[i % D])
// fused kernel to reduce memory bandwidth and improve performance
// =========================================================

__global__ void fusedBiasReluKernel(
    const float* input,
    const float* bias,
    float* output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        float x = input[idx] + bias[idx];

        output[idx] = (x > 0.0f) ? x : 0.0f;
    }
}

int main()
{
    const int N = 10;
    size_t bytes = N * sizeof(float);

    float h_input[N] =
    {
        -5.0f, -2.0f, -1.0f, 0.0f, 1.0f,
         2.0f,  3.0f,  4.0f, 5.0f, 6.0f
    };

    float h_bias[N] =
    {
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        -3.0f,-3.0f,-3.0f,-3.0f,-3.0f
    };

    float h_output[N];

    float* d_input;
    float* d_bias;
    float* d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_bias, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    fusedBiasReluKernel<<<blocks, threads>>>(
        d_input,
        d_bias,
        d_output,
        N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Fused Bias + ReLU Output\n";

    for (int i = 0; i < N; i++)
    {
        std::cout
            << "max(0, "
            << h_input[i]
            << " + "
            << h_bias[i]
            << ") = "
            << h_output[i]
            << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);

    return 0;
}