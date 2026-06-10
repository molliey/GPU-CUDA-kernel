#include <iostream>
#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void matrixTranspose(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    int transposedX = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposedY = blockIdx.x * TILE_DIM + threadIdx.y;

    if (transposedX < height && transposedY < width) {
        output[transposedY * height + transposedX] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int width = 512;
    const int height = 512;
    const int N = width * height;

    size_t bytes = N * sizeof(float);

    float* h_input = new float[N];
    float* h_output = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((width + TILE_DIM - 1) / TILE_DIM,
                (height + TILE_DIM - 1) / TILE_DIM);

    matrixTranspose<<<blocks, threads>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Input[1][2] = " << h_input[1 * width + 2] << std::endl;
    std::cout << "Output[2][1] = " << h_output[2 * height + 1] << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;

    return 0;
}

/*

Naive Matrix Transpose

__global__ void matrixTranspose(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if(row < rows && col < cols)
    {
        output[col * rows + row]
            =
        input[row * cols + col];
    }
}


Shared Memory Tile
↓
Bank Conflict
↓
Memory Coalescing
↓
Memory Bandwidth




*/