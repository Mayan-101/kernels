#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template<const int BK>
__global__ void matmul_1D(int M, int K, int N, float alpha, float *A, float *B, float beta, float *C)
{
    uint const x = threadIdx.x / BK+ BK* blockIdx.y;
    uint const y = threadIdx.x % BK+ BK* blockIdx.x;
    if (x < M && y < N)
    {
        float dot_prod = 0.f;
        for (int i = 0; i < K; i++)
        {
            dot_prod += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha*dot_prod + beta*C[x * N + y];
    }
}