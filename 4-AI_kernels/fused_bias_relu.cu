#include <iostream>
#include <cuda_runtime.h>

__global__ void fusedBiasReLU(const float* input, const float* bias, float* output, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * D) {
        int col = idx % D;
        float x = input[idx] + bias[col];
        output[idx] = x > 0.0f ? x : 0.0f;
    }
}

int main() {
    int N = 4;
    int D = 8;
    int size = N * D;
    size_t bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_bias = new float[D];
    float* h_output = new float[size];

    for (int i = 0; i < size; i++) h_input[i] = i - 10.0f;
    for (int i = 0; i < D; i++) h_bias[i] = 1.0f;

    float *d_input, *d_bias, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_bias, D * sizeof(float));
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, D * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    fusedBiasReLU<<<blocks, threads>>>(d_input, d_bias, d_output, N, D);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_bias;
    delete[] h_output;

    return 0;
}