#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {

        int tiled_row_A = row;
        int tiled_col_A = t * TILE_SIZE + threadIdx.x;
        if (tiled_row_A < M && tiled_col_A < K)
            tile_A[threadIdx.y][threadIdx.x] = A[tiled_row_A * K + tiled_col_A];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;


        int tiled_row_B = t * TILE_SIZE + threadIdx.y;
        int tiled_col_B = col;
        if (tiled_row_B < K && tiled_col_B < N)
            tile_B[threadIdx.y][threadIdx.x] = B[tiled_row_B * N + tiled_col_B];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // 计算局部乘积并累加
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}