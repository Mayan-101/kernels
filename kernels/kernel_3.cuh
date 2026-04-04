#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

template <const uint BK, const uint BM, const uint BN>
__global__ void matmul_tiled(int M, int K, int N, float alpha, float *A, float *B, float beta, float *C)
{
    const uint thread_pool = (BM)*(BN);

    uint const a_row = threadIdx.x / BK;
    uint const a_col = threadIdx.x % BK;
    uint const a_tile_stride = thread_pool / BK;

    uint const b_row = threadIdx.x / BN;
    uint const b_col = threadIdx.x % BN;
    uint const b_tile_stride = thread_pool / BN;

    uint const by = blockIdx.y;
    uint const bx = blockIdx.x;
    
    float dot_prod = 0.f;

    __shared__ float sh_A[BM][BK];
    __shared__ float sh_B[BK][BN];
    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N + bx*BN];

    #pragma unroll
    for (int k = 0; k < K; k += BK){
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride){
            sh_A[a_row + i][a_col] = A[(a_row + i)*K + a_col];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride){
            sh_B[b_row + i][b_col] = B[(b_row + i)*N + b_col];
        }
        A += BK;
        B += BK*N;
        __syncthreads();
        

        #pragma unroll
        for (int i = 0; i < BK; i++)
        {
            dot_prod += sh_A[b_row][i] * sh_B[i][b_col];
        }
        __syncthreads();
    }
    C[b_row*N + b_col] = alpha*dot_prod + beta*C[b_row*N + b_col];
    
}