#include <cuda_runtime.h>

#define KERNEL_SIZE 3
#define HALO (KERNEL_SIZE / 2)

// 初版二维卷积（对应报告图表二里的 Baseline）：
// 没有用共享内存，每个输出像素直接从全局内存读 3x3 一共 9 个邻居，
// 相邻线程会大量重复读取同一块输入数据，全局访存次数约是共享内存版的 9 倍
// 后来改成 14_conv2d.cu 的共享内存 + Halo 处理版本
__global__ void conv2d_naive_kernel(const float* input, const float* kernel, float* output, int width, int height) {
    // 计算当前线程负责的输出像素坐标
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_row < height && out_col < width) {
        float sum = 0.0f;
        // 直接从全局内存读取输入，图像边界外按 0 处理（零填充）
        for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
            for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                int in_r = out_row + ky - HALO;
                int in_c = out_col + kx - HALO;
                if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                    sum += input[in_r * width + in_c] * kernel[ky * KERNEL_SIZE + kx];
                }
            }
        }
        output[out_row * width + out_col] = sum;
    }
}

// Host 端入口函数
extern "C" void solve(const float* input, const float* kernel, float* output, int width, int height) {
    float *d_in, *d_k, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(float));
    cudaMalloc(&d_k, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_out, width * height * sizeof(float));

    cudaMemcpy(d_in, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    conv2d_naive_kernel<<<grid, block>>>(d_in, d_k, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_k); cudaFree(d_out);
}
