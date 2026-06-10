#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define MAX_D 1024

__global__ void softmaxKernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;

    if (row < rows) {
        float maxVal = input[row * cols];

        for (int i = 1; i < cols; i++) {
            float val = input[row * cols + i];
            if (val > maxVal) maxVal = val;
        }

        float sum = 0.0f;

        for (int i = 0; i < cols; i++) {
            float e = expf(input[row * cols + i] - maxVal);
            output[row * cols + i] = e;
            sum += e;
        }

        for (int i = 0; i < cols; i++) {
            output[row * cols + i] /= sum;
        }
    }
}

int main() {
    int rows = 4;
    int cols = 8;
    int size = rows * cols;
    size_t bytes = size * sizeof(float);

    float* h_input = new float[size];
    float* h_output = new float[size];

    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(i % cols);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    softmaxKernel<<<rows, 1>>>(d_input, d_output, rows, cols);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        std::cout << "Row " << r << ": ";
        for (int c = 0; c < cols; c++) {
            std::cout << h_output[r * cols + c] << " ";
            sum += h_output[r * cols + c];
        }
        std::cout << " | sum = " << sum << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}