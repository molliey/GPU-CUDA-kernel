#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void tiledMatrixMultiplication(const float* A,
                                          const float* B,
                                          float* C,
                                          int M,
                                          int N,
                                          int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < N && col < K) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * K + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

int main() {
    const int M = 128;
    const int N = 256;
    const int K = 64;

    size_t bytesA = M * N * sizeof(float);
    size_t bytesB = N * K * sizeof(float);
    size_t bytesC = M * K * sizeof(float);

    float* h_A = new float[M * N];
    float* h_B = new float[N * K];
    float* h_C = new float[M * K];

    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < N * K; i++) {
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    tiledMatrixMultiplication<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "Expected = " << N * 2.0f << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}