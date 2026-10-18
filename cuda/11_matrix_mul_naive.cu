#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// 初版矩阵乘法：还没考虑性能问题，直接读写全局内存
// 问题在于：C 的每个元素要循环 K 次，每次都从全局内存各读一个 A 和 B 的值，
// 全局显存延迟高，而且同一个 Tile 内的线程重复读取了大量相同的数据
// 后来意识到这是带宽瓶颈，才有了 11_shared_memory.cu 那个共享内存分块的版本
__global__ void matrix_mul_naive_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // 计算当前线程负责的输出行列坐标
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // 直接从全局内存逐个读取并累加，没有任何分块
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host 端入口函数
extern "C" void solve(const float* A, const float* B, float* C, int M, int K, int N) {
    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 数据从 Host 拷贝到 Device
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 配置 Grid 和 Block 并启动 Kernel
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrix_mul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // 结果拷贝回 Host 并释放内存
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
