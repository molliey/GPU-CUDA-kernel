#include <iostream>
#include <cuda_runtime.h>

__global__ void simpleReduction(const float* input, float* output, int N) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        shared[tid] = input[idx];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * sizeof(float);

    float *h_input = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float *h_partial = new float[blocks];

    float *d_input, *d_partial;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    simpleReduction<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_partial, N);

    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;

    for (int i = 0; i < blocks; i++) {
        sum += h_partial[i];
    }

    std::cout << "Reduction sum = " << sum << std::endl;
    std::cout << "Expected sum  = " << N << std::endl;

    cudaFree(d_input);
    cudaFree(d_partial);

    delete[] h_input;
    delete[] h_partial;

    return 0;
}

/*

CPU

float sum = 0;

for (int i = 0; i < N; i++)
{
    sum += input[i];
}

Thread0 负责 input[0]
Thread1 负责 input[1]
Thread2 负责 input[2]
...


data race condition


*/