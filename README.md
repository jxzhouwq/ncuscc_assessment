# ncuscc_assessment

NCUSC 超算队迎新考核仓库。主要内容是Linux 基础关卡与 CUDA 并行计算进阶，最终进阶方向选择了 **6-2 二维卷积优化**。

- 环境：Windows 11 (宿主机) / Ubuntu 22.04 (WSL2 & 容器)，CUDA Toolkit 12.x，`nvcc` + `gcc` 编译
- 报告：见 [report/考核报告.md](report/考核报告.md) 与 [report/考核报告.pdf](report/考核报告.pdf)

## 目录结构

```text
ncuscc_assessment/
├── README.md               # 项目说明
├── cuda/                   # CUDA 进阶阶段
│   ├── 08_hello_cuda.cu    # Hello CUDA
│   ├── 09_vector_add.cu    # 向量加法
│   ├── 10_matrix_add.cu    # 矩阵加法
│   ├── 11_matrix_mul_naive.cu  # 11题初版：直接读全局内存
│   ├── 11_shared_memory.cu     # 11题优化版：共享内存分块
│   ├── 13_gemm.cu          # GEMM 优化
│   ├── 14_conv2d_naive.cu  # 14题初版（Baseline）
│   ├── 14_conv2d.cu        # 二维卷积优化（最终方向）
│   └── 15_grayscale.cu     # 图像灰度转换
└── report/                 # 报告目录
    ├── 考核报告.md         # 学习报告（Markdown 源文件）
    ├── 考核报告.pdf        # 学习报告 PDF
    ├── 感悟报告.md         # 感悟报告（Markdown 源文件）
    ├── 感悟报告.pdf        # 感悟报告 PDF
    └── images/             # 报告中的图表与截图
```

`*_naive.cu` 是对应题目的初版写法（直接读写全局内存、没做性能优化），保留下来与优化版对比。

## 编译与运行

```bash
cd cuda
nvcc -O2 08_hello_cuda.cu -o hello && ./hello
nvcc -O2 14_conv2d.cu -o conv2d && ./conv2d
```

## 进阶方向

- 二维线程映射：`blockIdx.x/y` + `threadIdx.x/y`，行优先存储按 `idx = row * N + col` 扁平化索引
- 边界保护：`row < M && col < N` 严格越界检查
- 共享内存分块（Tiling）：`__shared__` + `__syncthreads()` 减少全局显存访问
- Halo 处理：共享内存 Tile 尺寸为 `TILE_SIZE + 2*HALO`，保证 3x3 卷积边界像素计算正确