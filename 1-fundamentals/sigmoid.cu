#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void sigmoid(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_input = new float[N], *h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = (i - 512) / 100.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    sigmoid<<<blocks, threads>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int i = 508; i < 518; i++) {
        std::cout << "sigmoid(" << h_input[i] << ") = " << h_output[i] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}


/*

ReLU:
output[i] = max(0, input[i])

Sigmoid:
output[i] = 1 / (1 + expf(-input[i]))


*/