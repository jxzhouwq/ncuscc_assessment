#include <cstdio>

extern "C" __global__ void hello_kernel() {

    int idx = threadIdx.x;

    printf("hello cuda %d\n", idx);
}