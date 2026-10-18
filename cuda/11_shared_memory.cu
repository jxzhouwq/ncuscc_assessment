#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrix_mul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // 1. 声明共享内存，用于存放 A 和 B 的子块（Tile）
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // 2. 计算当前线程对应的全局行和列
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 累加器，用于计算 C[row][col]
    float sum = 0.0f;

    // 3. 循环遍历 K 维度，按 TILE_SIZE 切块加载
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 计算当前要加载的全局坐标
        int tiled_row_A = row;
        int tiled_col_A = t * TILE_SIZE + threadIdx.x;
        int tiled_row_B = t * TILE_SIZE + threadIdx.y;
        int tiled_col_B = col;

        // 4. 加载 A 的子块到共享内存（提示4：处理越界，越界填0）
        if (tiled_row_A < M && tiled_col_A < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tiled_row_A * K + tiled_col_A];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 5. 加载 B 的子块到共享内存（处理越界）
        if (tiled_row_B < K && tiled_col_B < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[tiled_row_B * N + tiled_col_B];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 6. 等待所有线程完成加载（提示3：线程同步）
        __syncthreads();

        // 7. 计算当前 tile 的局部乘积并累加
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        // 8. 等待所有线程完成计算，再进行下一次循环（防止下一次加载覆盖还没算完的数据）
        __syncthreads();
    }

    // 9. 将结果写回全局内存 C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host 端入口函数（提示1：自行构建 host 函数）
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    // 1. 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 2. 将数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. 配置 Grid 和 Block
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // 4. 启动 Kernel
    matrix_mul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    // 5. 等待 Kernel 执行完毕
    cudaDeviceSynchronize();

    // 6. 将结果从 Device 拷贝回 Host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. 释放 Device 内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}