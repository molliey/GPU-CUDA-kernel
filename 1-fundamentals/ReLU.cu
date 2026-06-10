#include <iostream>
#include <cuda_runtime.h>

__global__ void relu(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_input = new float[N], *h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = i - 512;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    relu<<<blocks, threads>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int i = 508; i < 518; i++) {
        std::cout << "ReLU(" << h_input[i] << ") = " << h_output[i] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}

/*

ReLU = Rectified Linear Unit

CPU

for(int i=0;i<N;i++)
{
    output[i] = std::max(0.0f, input[i]);
}

Thread0 -> input[0]

Thread1 -> input[1]

Thread2 -> input[2]

...


*/