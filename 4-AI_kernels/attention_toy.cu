#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define SEQ_LEN 4
#define D_MODEL 4

__global__ void attentionToyKernel(const float* Q,
                                   const float* K,
                                   const float* V,
                                   float* output) {
    int row = blockIdx.x;

    if (row < SEQ_LEN) {
        float scores[SEQ_LEN];

        for (int j = 0; j < SEQ_LEN; j++) {
            float dot = 0.0f;

            for (int d = 0; d < D_MODEL; d++) {
                dot += Q[row * D_MODEL + d] * K[j * D_MODEL + d];
            }

            scores[j] = dot / sqrtf((float)D_MODEL);
        }

        float maxScore = scores[0];

        for (int j = 1; j < SEQ_LEN; j++) {
            if (scores[j] > maxScore) maxScore = scores[j];
        }

        float sum = 0.0f;

        for (int j = 0; j < SEQ_LEN; j++) {
            scores[j] = expf(scores[j] - maxScore);
            sum += scores[j];
        }

        for (int j = 0; j < SEQ_LEN; j++) {
            scores[j] /= sum;
        }

        for (int d = 0; d < D_MODEL; d++) {
            float value = 0.0f;

            for (int j = 0; j < SEQ_LEN; j++) {
                value += scores[j] * V[j * D_MODEL + d];
            }

            output[row * D_MODEL + d] = value;
        }
    }
}

int main() {
    int size = SEQ_LEN * D_MODEL;
    size_t bytes = size * sizeof(float);

    float* h_Q = new float[size];
    float* h_K = new float[size];
    float* h_V = new float[size];
    float* h_output = new float[size];

    for (int i = 0; i < size; i++) {
        h_Q[i] = 1.0f;
        h_K[i] = 1.0f;
        h_V[i] = static_cast<float>(i);
    }

    float *d_Q, *d_K, *d_V, *d_output;

    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);

    attentionToyKernel<<<SEQ_LEN, 1>>>(d_Q, d_K, d_V, d_output);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int r = 0; r < SEQ_LEN; r++) {
        std::cout << "Output row " << r << ": ";
        for (int d = 0; d < D_MODEL; d++) {
            std::cout << h_output[r * D_MODEL + d] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_output;

    return 0;
}