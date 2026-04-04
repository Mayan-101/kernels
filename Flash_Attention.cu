#include <stdio.h>
#include <cuda_runtime.h>
#define TM 8
#define BK 8
#define SBK 96
#define TN 4
#define BM 32
#define BLOCK_SIZE 64
#define BN 32
static __device__ __forceinline__
float4 safe_ldB(const float * __restrict__ B,
                int row, int col, int K, int N)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= K) return v;
    const float *p = B + (size_t)row * N + col;
    if (col + 3 < N) return reinterpret_cast<const float4 *>(p)[0];
    if (col     < N) v.x = p[0];
    if (col + 1 < N) v.y = p[1];
    if (col + 2 < N) v.z = p[2];
    return v;
}
static __device__ __forceinline__
float4 safe_ldA(const float * __restrict__ A,
                int row, int col, int M, int K)
{
    float4 v = {0.f, 0.f, 0.f, 0.f};
    if (row >= M) return v;
    const float *p = A + (size_t)row * K + col;
    if (col + 3 < K) return reinterpret_cast<const float4 *>(p)[0];
    if (col     < K) v.x = p[0];
    if (col + 1 < K) v.y = p[1];
    if (col + 2 < K) v.z = p[2];
    return v;
}
__global__ void matmul_tiled_2D_coarse_vec(int M, int K, int N,
                                           const float * __restrict__ A,
                                           const float * __restrict__ B,
                                           float       * __restrict__ C)
{
    const uint threadRow = threadIdx.x / (BN / TN);   
    const uint threadCol = threadIdx.x % (BN / TN);   

    const uint innerRowA = threadIdx.x / (BK / 4);   
    const uint innerColA = threadIdx.x % (BK / 4);   
    const uint innerRowB = threadIdx.x / (BN / 4);   
    const uint innerColB = threadIdx.x % (BN / 4);   

    __shared__ float sh_AT[BK * BM];   
    __shared__ float sh_B [BK * BN];   


    float threadResults[TM * TN] = {};
    float regM[TM] = {};
    float regN[TN] = {};

    constexpr int stride_A = BLOCK_SIZE / (BK / 4);
    constexpr int stride_B = BLOCK_SIZE / (BN / 4);

    const int phases = (K + BK - 1) / BK;

    for (int phase = 0; phase < phases; ++phase)
    {
        for (int offset = 0; offset < BM; offset += stride_A)
        {
            const int gRow = blockIdx.x * BM + innerRowA + offset;
            const int gCol = phase * BK  + innerColA * 4;
            float4 tmp     = safe_ldA(A, gRow, gCol, M, K);

            sh_AT[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            sh_AT[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            sh_AT[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            sh_AT[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int offset = 0; offset < BK; offset += stride_B)
        {
            const int gRow = phase * BK       + innerRowB + offset;
            const int gCol = blockIdx.y * BN  + innerColB * 4;
            reinterpret_cast<float4 *>(
                &sh_B[(innerRowB + offset) * BN + innerColB * 4])[0]
                    = safe_ldB(B, gRow, gCol, K, N);
        }
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            reinterpret_cast<float4 *>(regM    )[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM    ])[0];
            reinterpret_cast<float4 *>(regM + 4)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM + 4])[0];


            reinterpret_cast<float4 *>(regN)[0] =
                reinterpret_cast<float4 *>(&sh_B[dotIdx * BN + threadCol * TN])[0];

            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }


    for (int m = 0; m < TM; ++m)
    {
        const int gRow = blockIdx.x * BM + threadRow * TM + m;
        if (gRow >= M) continue;          
        for (int n = 0; n < TN; ++n)
        {
            const int gCol = blockIdx.y * BN + threadCol * TN + n;
            if (gCol >= N) continue;      
            C[(size_t)gRow * N + gCol] = threadResults[m * TN + n];
        }
    }
}


__global__ void softmax_shared(int R, int C, float *input, float temperature, float *output) {
    int ly = threadIdx.x / SBK;   
    int lx = threadIdx.x % SBK;   

    __shared__ float sf[SBK * SBK];
    __shared__ float max_val[SBK];
    __shared__ float denom[SBK];


    if (lx == 0) {
        max_val[ly] = -3.402823466e+38f;
        denom[ly]   = 0.f;
    }
    __syncthreads();

    for (int phase = 0; phase < C / SBK; phase++) {
        sf[ly * SBK + lx] = input[(phase * SBK + ly) * C + blockIdx.x * SBK + lx];
        __syncthreads();


        if (lx == 0) {
            float new_max = max_val[ly];
            for (int i = 0; i < SBK; i++)
                new_max = fmaxf(new_max, sf[ly * SBK + i]);

            float new_denom = denom[ly] * expf(max_val[ly] - new_max);
            for (int i = 0; i < SBK; i++)
                new_denom += expf(sf[ly * SBK + i] - new_max);

            max_val[ly] = new_max;
            denom[ly]   = new_denom;
        }
        __syncthreads();
    }

    for (int phase = 0; phase < C / SBK; phase++) {
        sf[ly * SBK + lx] = input[(phase * SBK + ly) * C + blockIdx.x * SBK + lx];
        __syncthreads();
        output[(phase * SBK + ly) * C + blockIdx.x * SBK + lx] =
            expf(sf[ly * SBK + lx] - max_val[ly]) / denom[ly];
        __syncthreads();
    }
}

__global__ void FlashAttention(int d_in, int d, int T,
                                            const float * __restrict__ Q,
                                            const float * __restrict__ K,
                                            float       * __restrict__ V,
                                            float *__restrict__ O)
{
    const uint threadRow = threadIdx.x / (BN / TN);   
    const uint threadCol = threadIdx.x % (BN / TN);   

    const uint innerRowA = threadIdx.x / (BK / 4);   
    const uint innerColA = threadIdx.x % (BK / 4);   
    const uint innerRowB = threadIdx.x / (BN / 4);   
    const uint innerColB = threadIdx.x % (BN / 4);   

    __shared__ float sh_AT[BK * BM];   
    __shared__ float sh_BT[BN * BK];   


    float threadResults[TM * TN] = {};
    float regM[TM] = {};
    float regN[TN] = {};

    constexpr int stride_A = BLOCK_SIZE / (BK / 4);
    constexpr int stride_B = BLOCK_SIZE / (BN / 4);

    const int phases = (d + BK - 1) / BK;

    for (int phase = 0; phase < phases; ++phase)
    {
        for (int offset = 0; offset < BM; offset += stride_A)
        {
            const int gRow = blockIdx.x * BM + innerRowA + offset;
            const int gCol = phase * BK  + innerColA * 4;
            float4 tmp     = safe_ldA(Q, gRow, gCol, M, d);

            sh_AT[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            sh_AT[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            sh_AT[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            sh_AT[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (int offset = 0; offset < BK; offset += stride_B)
        {
            const int gRow = phase * BK       + innerRowB + offset;
            const int gCol = blockIdx.y * BN  + innerColB * 4;
            float4 tmp     = safe_ldB(K, gRow, gCol, d, N);

            sh_BT[(innerColB) * BK + innerRowB * 4 + 0 + offset] = tmp.x;
            sh_BT[(innerColB) * BK + innerRowB * 4 + 1 + offset] = tmp.y;
            sh_BT[(innerColB) * BK + innerRowB * 4 + 2 + offset] = tmp.z;
            sh_BT[(innerColB) * BK + innerRowB * 4 + 3 + offset] = tmp.w;
        }
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            reinterpret_cast<float4 *>(regM    )[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM    ])[0];
            reinterpret_cast<float4 *>(regM + 4)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM + 4])[0];


            reinterpret_cast<float4 *>(regN)[0] =
                reinterpret_cast<float4 *>(&sh_BT[dotIdx * BK + threadCol * TN])[0];

            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }


}