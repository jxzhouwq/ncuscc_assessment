#include <cuda_runtime.h>

// 整数灰度公式（提示4）：Gray = (77*R + 150*G + 29*B) >> 8
__global__ void rgba_to_gray_kernel(const uchar4* rgba, unsigned char* gray, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uchar4 pixel = rgba[idx];

        gray[idx] = (unsigned char)((77 * pixel.x + 150 * pixel.y + 29 * pixel.z) >> 8);
    }
}

extern "C" void solve(const uchar4* rgba, unsigned char* gray, int num_pixels) {
    uchar4 *d_rgba;
    unsigned char *d_gray;
    cudaMalloc(&d_rgba, num_pixels * sizeof(uchar4));
    cudaMalloc(&d_gray, num_pixels * sizeof(unsigned char));

    cudaMemcpy(d_rgba, rgba, num_pixels * sizeof(uchar4), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (num_pixels + block - 1) / block;
    rgba_to_gray_kernel<<<grid, block>>>(d_rgba, d_gray, num_pixels);
    cudaDeviceSynchronize();

    cudaMemcpy(gray, d_gray, num_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_rgba); cudaFree(d_gray);
}