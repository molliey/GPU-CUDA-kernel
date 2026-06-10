#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// =======================================
// Tiled Matrix Multiplication Kernel
// C = A * B 
// A: M x N
// B: N x K
// C: M x K
// Each thread block computes a TILE_SIZE x TILE_SIZE tile of C
// =======================================

__global__ void tiledMatrixMultiplication(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // Shared Memory
    // tileA stores a TILE_SIZE x TILE_SIZE tile of A
    // tileB stores a TILE_SIZE x TILE_SIZE tile of B
    // shared memory belongs to one block
    // speed is much faster than global memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // tx: thread's x coordinate within the block
    // ty: thread's y coordinate within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // one block computes a TILE_SIZE x TILE_SIZE tile of C
    // one thread computes one element of C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // result for C[row][col]
    float sum = 0.0f;

    // loop over tiles of A and B
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++)
    {
        // load tile of A into shared memory
        // load tile of B into shared memory
        // handle boundary conditions
        int tiledColA = t * TILE_SIZE + tx;
        int tiledRowB = t * TILE_SIZE + ty;

        if (row < M && tiledColA < N)
            tileA[ty][tx] = A[row * N + tiledColA];
        else
            tileA[ty][tx] = 0.0f;

        if (tiledRowB < N && col < K)
            tileB[ty][tx] = B[tiledRowB * K + col];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K)
    {
        C[row * K + col] = sum;
    }
}

int main() {
    // Matrix dimensions
    const int M = 128;
    const int N = 256;
    const int K = 64;

    // size of matrices in bytes
    size_t bytesA = M * N * sizeof(float);
    size_t bytesB = N * K * sizeof(float);
    size_t bytesC = M * K * sizeof(float);

    // host memory allocation
    float* h_A = new float[M * N];
    float* h_B = new float[N * K];
    float* h_C = new float[M * K];

    // initialize A and B with some values
    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.0f;
    }

    // initialize B with some values
    for (int i = 0; i < N * K; i++) {
        h_B[i] = 2.0f;
    }

    // initialize C with zeros
    for (int i = 0; i < M * K; i++) {
        h_C[i] = 0.0f;
    }
    
    // device memory allocation
    float *d_A, *d_B, *d_C;

    // copy data from host to device
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    // copy A and B to device
    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // launch configuration
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((K + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    // launch kernel
    tiledMatrixMultiplication<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);

    // copy result from device to host
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);

    // verify results
    std::cout << "C[0][0] = " << h_C[0] << std::endl;
    std::cout << "Expected = " << N * 2.0f << std::endl;

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

/*








*/