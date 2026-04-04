#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template <const uint BK, const uint BM, const uint BN, const uint TM, const uint TN>
__global__ void matmul_tiled(int M, int K, int N, float alpha, float *A, float *B, float beta, float *C)
{
    const uint thread_pool = (BM / TM) * (BN / TN);

    uint const a_row = threadIdx.x / BK;
    uint const a_col = threadIdx.x % BK;
    uint const a_tile_stride = thread_pool / BK;

    uint const b_row = threadIdx.x / BN;
    uint const b_col = threadIdx.x % BN;
    uint const b_tile_stride = thread_pool / BN;

    uint const by = blockIdx.y;
    uint const bx = blockIdx.x;

    uint const tx = (threadIdx.x / (BN / TN)) * TM;
    uint const ty = (threadIdx.x % (BN / TN)) * TN;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];

    float temp[TM * TN] = {0.f};
    float a_temp[TM];
    float b_temp[TN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];
#pragma unroll
    for (int k = 0; k < K; k += BK)
    {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride)
        {
            sh_A[a_row + i][a_col] = A[(a_row + i) * K + a_col];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride)
        {
            sh_B[b_row + i][b_col] = B[(b_row + i) * N + b_col];
        }
        A += BK;
        B += BK * N;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; i++)
        {
            #pragma unroll
            for (int y = 0; y < TM; y++)
            {
                a_temp[y] = sh_A[ty + y][i];
            }
            #pragma unroll
            for (int z = 0; z < TN; z++)
            {
                b_temp[z] = sh_B[i][tx + z];
            }
            #pragma unroll
            for (int y = 0; y < TM; y++)
            {
                #pragma unroll
                for (int z = 0; z < TN; z++)
                {
                    temp[y*TN + z] += a_temp[y]*b_temp[z];
                }
            }
        }
        __syncthreads();
    }
       #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
            C[(ty + j) * N + tx + l] = alpha * temp[j*TN + l] + beta * C[(ty + j) * N + tx + l];
    }
}
