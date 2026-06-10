#include <iostream>
#include <cuda_runtime.h>

__global__ void scalarMultiply(const float* input, float* output, float scalar, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx] * scalar;
    }
}

int main() {
    const int N = 1024;
    const float scalar = 3.0f;
    size_t bytes = N * sizeof(float);

    float *h_input = new float[N], *h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    scalarMultiply<<<blocks, threads>>>(d_input, d_output, scalar, N);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << h_input[i] << " * " << scalar << " = " << h_output[i] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}