#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERNEL_SIZE 3
#define HALO (KERNEL_SIZE / 2)
#define IN_TILE (TILE_SIZE + 2 * HALO)

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int width, int height) {

    __shared__ float tile[IN_TILE][IN_TILE];


    int out_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    for (int i = threadIdx.y * TILE_SIZE + threadIdx.x; i < IN_TILE * IN_TILE; i += TILE_SIZE * TILE_SIZE) {
        int r = i / IN_TILE;
        int c = i % IN_TILE;
        int in_r = blockIdx.y * TILE_SIZE - HALO + r;
        int in_c = blockIdx.x * TILE_SIZE - HALO + c;
        if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width)
            tile[r][c] = input[in_r * width + in_c];
        else
            tile[r][c] = 0.0f;
    }

    __syncthreads();

    if (out_row < height && out_col < width) {
        float sum = 0.0f;
        for (int ky = 0; ky < KERNEL_SIZE; ++ky)
            for (int kx = 0; kx < KERNEL_SIZE; ++kx)
                sum += tile[threadIdx.y + ky][threadIdx.x + kx] * kernel[ky * KERNEL_SIZE + kx];
        output[out_row * width + out_col] = sum;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int width, int height) {
    float *d_in, *d_k, *d_out;
    cudaMalloc(&d_in, width * height * sizeof(float));
    cudaMalloc(&d_k, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&d_out, width * height * sizeof(float));

    cudaMemcpy(d_in, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    conv2d_kernel<<<grid, block>>>(d_in, d_k, d_out, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_k); cudaFree(d_out);
}