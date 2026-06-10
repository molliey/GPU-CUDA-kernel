#include <iostream>
#include <cuda_runtime.h>

__global__ void atomicReduction(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        atomicAdd(output, input[idx]);
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_input = new float[N];
    float h_output = 0.0f;

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    atomicReduction<<<blocks, threads>>>(d_input, d_output, N);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Atomic reduction sum = " << h_output << std::endl;
    std::cout << "Expected sum = " << N << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;

    return 0;
}