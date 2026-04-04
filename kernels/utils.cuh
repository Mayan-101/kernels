#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static __device__ __forceinline__
float4 safe_ldA(const float *__restrict__ A, int row, int col, int M, int K)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= M) return v;
    const float *p = A + (size_t)row * K + col;
    if (col + 3 < K && ((uintptr_t)p & 15) == 0)   // ← alignment guard
        return reinterpret_cast<const float4 *>(p)[0];
    if (col     < K) v.x = p[0];
    if (col + 1 < K) v.y = p[1];
    if (col + 2 < K) v.z = p[2];
    if (col + 3 < K) v.w = p[3];
    return v;
}

static __device__ __forceinline__
float4 safe_ldB(const float *__restrict__ B, int row, int col, int K, int N)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= K) return v;
    const float *p = B + (size_t)row * N + col;
    if (col + 3 < N && ((uintptr_t)p & 15) == 0)   // ← alignment guard
        return reinterpret_cast<const float4 *>(p)[0];
    if (col     < N) v.x = p[0];
    if (col + 1 < N) v.y = p[1];
    if (col + 2 < N) v.z = p[2];
    if (col + 3 < N) v.w = p[3];
    return v;
}

static __device__ __forceinline__
float4 safe_ldC(const float *__restrict__ C, int row, int col, int M, int N)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= M) return v;
    const float *p = C + (size_t)row * N + col;
    if (col + 3 < N && ((uintptr_t)p & 15) == 0)   // ← alignment guard
        return reinterpret_cast<const float4 *>(p)[0];
    if (col     < N) v.x = p[0];
    if (col + 1 < N) v.y = p[1];
    if (col + 2 < N) v.z = p[2];
    if (col + 3 < N) v.w = p[3];
    return v;
}

static __device__ __forceinline__
void safe_stC(float *__restrict__ C, int row, int col, int M, int N, float4 val)
{
    if (row >= M) return;
    float *p = C + (size_t)row * N + col;
    if (col + 3 < N && ((uintptr_t)p & 15) == 0) { // ← alignment guard
        reinterpret_cast<float4 *>(p)[0] = val;
        return;
    }
    if (col     < N) p[0] = val.x;
    if (col + 1 < N) p[1] = val.y;
    if (col + 2 < N) p[2] = val.z;
    if (col + 3 < N) p[3] = val.w;
}
