#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static __device__ __forceinline__
    float4
    safe_ldB(const float *__restrict__ B,
             int row, int col, int K, int N)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= K)
        return v;
    const float *p = B + (size_t)row * N + col;
    if (col + 3 < N)
        return reinterpret_cast<const float4 *>(p)[0];
    if (col < N)
        v.x = p[0];
    if (col + 1 < N)
        v.y = p[1];
    if (col + 2 < N)
        v.z = p[2];
    return v;
}
static __device__ __forceinline__
    float4
    safe_ldA(const float *__restrict__ A,
             int row, int col, int M, int K)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= M)
        return v;
    const float *p = A + (size_t)row * K + col;
    if (col + 3 < K)
        return reinterpret_cast<const float4 *>(p)[0];
    if (col < K)
        v.x = p[0];
    if (col + 1 < K)
        v.y = p[1];
    if (col + 2 < K)
        v.z = p[2];
    return v;
}
template <const uint BK, const uint BM, const uint BN, const uint TM, const uint TN, const uint extra_cols>
__global__ void matmul_tiled(int M, int K, int N, float alpha, float *A, float *B, float beta, float *C)
{
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    __shared__ float sh_AT[BK * BM];
    __shared__ float sh_B[BK * (BN + extra_cols)];

    float threadResults[TM * TN] = {};
    float regM[TM] = {};
    float regN[TN] = {};

    constexpr int stride_A = ((BM / TM) * (BN / TN)) / (BK / 4);
    constexpr int stride_B = ((BM / TM) * (BN / TN)) / (BN / 4);

    const int phases = (K + BK - 1) / BK;
     
    for (int phase = 0; phase < phases; ++phase)
    {
         
        for (int offset = 0; offset < BM; offset += stride_A)
        {
            const int gRow = blockIdx.x * BM + innerRowA + offset;
            const int gCol = phase * BK + innerColA * 4;
            float4 tmp = safe_ldA(A, gRow, gCol, M, K);

            sh_AT[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            sh_AT[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            sh_AT[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            sh_AT[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
         
        for (int offset = 0; offset < BK; offset += stride_B)
        {
            const int gRow = phase * BK + innerRowB + offset;
            const int gCol = blockIdx.y * BN + innerColB * 4;
            float4 tmp = safe_ldB(B, gRow, gCol, K, N);
            sh_B[(innerRowB + offset) * (BN + extra_cols) + innerColB * 4 + 0] = tmp.x;
            sh_B[(innerRowB + offset) * (BN + extra_cols) + innerColB * 4 + 1] = tmp.y;
            sh_B[(innerRowB + offset) * (BN + extra_cols) + innerColB * 4 + 2] = tmp.z;
            sh_B[(innerRowB + offset) * (BN + extra_cols) + innerColB * 4 + 3] = tmp.w;
        }
        __syncthreads();
        
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            reinterpret_cast<float4 *>(regM)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM])[0];
            reinterpret_cast<float4 *>(regM + 4)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM + 4])[0];

            reinterpret_cast<float4 *>(regN)[0] =
                reinterpret_cast<float4 *>(&sh_B[dotIdx * (BN + extra_cols) + threadCol * TN])[0];
            reinterpret_cast<float4 *>(regN + 4)[0] =
                reinterpret_cast<float4 *>(&sh_B[dotIdx * (BN + extra_cols) + threadCol * TN + 4])[0];

            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
        const int cRow = blockIdx.x * BM + threadRow * TM + resIdxM;
        const int cCol = blockIdx.y * BN + threadCol * TN + resIdxN;

        float4 tmp = reinterpret_cast<float4 *>(&C[cRow * N + cCol])[0];
        tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
        tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
        tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
        tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
        reinterpret_cast<float4 *>(&C[cRow * N + cCol])[0] = tmp;
    }
}
}