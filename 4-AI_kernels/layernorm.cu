#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void layerNormKernel(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                int rows,
                                int cols,
                                float eps) {
    int row = blockIdx.x;

    if (row < rows) {
        float mean = 0.0f;

        for (int i = 0; i < cols; i++) {
            mean += input[row * cols + i];
        }

        mean /= cols;

        float variance = 0.0f;

        for (int i = 0; i < cols; i++) {
            float diff = input[row * cols + i] - mean;
            variance += diff * diff;
        }

        variance /= cols;

        for (int i = 0; i < cols; i++) {
            float normalized = (input[row * cols + i] - mean) / sqrtf(variance + eps);
            output[row * cols + i] = normalized * gamma[i] + beta[i];
        }
    }
}

int main() {
    int rows = 4;
    int cols = 8;
    int size = rows * cols;
    float eps = 1e-5f;

    size_t bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_gamma = new float[cols];
    float* h_beta = new float[cols];
    float* h_output = new float[size];

    for (int i = 0; i < size; i++) h_input[i] = static_cast<float>(i % cols);
    for (int i = 0; i < cols; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    float *d_input, *d_gamma, *d_beta, *d_output;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_beta, cols * sizeof(float));
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, cols * sizeof(float), cudaMemcpyHostToDevice);

    layerNormKernel<<<rows, 1>>>(d_input, d_gamma, d_beta, d_output, rows, cols, eps);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; r++) {
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < cols; c++) {
            std::cout << h_output[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_gamma;
    delete[] h_beta;
    delete[] h_output;

    return 0;
}