#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ─── Auto-Tuning Macros ─────────────────────────────────────────────
#ifndef _BM
#define _BM 64
#endif
#ifndef _BN
#define _BN 64
#endif
#ifndef _BK
#define _BK 64
#endif
#ifndef _TM
#define _TM 8
#endif
#ifndef _TN
#define _TN 8
#endif

#define BM _BM
#define BN _BN
#define BK _BK
#define TM _TM
#define TN _TN

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

__global__ void matmul_tiled_2D_coarse_vec(int M, int K, int N,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float *__restrict__ C)
{
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);

    __shared__ float sh_AT[BK * BM];
    __shared__ float sh_B[BK * BN];

    float threadResults[TM * TN] = {};
    float regM[TM] = {};
    float regN[TN] = {};

    constexpr int stride_A = BM / (BK / 4);
    constexpr int stride_B = BM / (BN / 4);

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
            reinterpret_cast<float4 *>(
                &sh_B[(innerRowB + offset) * BN + innerColB * 4])[0] = safe_ldB(B, gRow, gCol, K, N);
        }
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx)
        {

#if TM >= 4
            reinterpret_cast<float4 *>(regM)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM])[0];
#endif
#if TM >= 8
            reinterpret_cast<float4 *>(regM + 4)[0] =
                reinterpret_cast<float4 *>(&sh_AT[dotIdx * BM + threadRow * TM + 4])[0];
#endif

// Safely load into regN based on compile-time TN size
#if TN >= 4
            reinterpret_cast<float4 *>(regN)[0] =
                reinterpret_cast<float4 *>(&sh_B[dotIdx * BN + threadCol * TN])[0];
#endif
#if TN >= 8
            reinterpret_cast<float4 *>(regN + 4)[0] =
                reinterpret_cast<float4 *>(&sh_B[dotIdx * BN + threadCol * TN + 4])[0];
#endif

            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    threadResults[m * TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }

    for (int m = 0; m < TM; ++m)
    {
        const int gRow = blockIdx.x * BM + threadRow * TM + m;
        if (gRow >= M)
            continue;
        for (int n = 0; n < TN; ++n)
        {
            const int gCol = blockIdx.y * BN + threadCol * TN + n;
            if (gCol >= N)
                continue;
            C[(size_t)gRow * N + gCol] = threadResults[m * TN + n];
        }
    }
}

// ─── Verification Logic ─────────────────────────────────────────────
bool verify_results(const float *C_custom, const float *C_reference, int size)
{
    const float epsilon = 1e-4;
    for (int i = 0; i < size; i++)
    {
        float diff = std::abs(C_custom[i] - C_reference[i]);
        float max_val = std::max(std::abs(C_custom[i]), std::abs(C_reference[i]));

        // Relative and absolute error check
        if (diff > epsilon && (diff / max_val) > epsilon)
        {
            std::cerr << "Verification FAILED at index " << i
                      << ": Custom=" << C_custom[i]
                      << ", cuBLAS=" << C_reference[i] << std::endl;
            return false;
        }
    }
    std::cerr << "Verification PASSED!" << std::endl;
    return true;
}

// ─── Harness & Timing Logic ─────────────────────────────────────────
int main(int argc, char **argv)
{
    int N = 2048;
    if (argc > 1)
        N = atoi(argv[1]);

    // Safety checks for vectorization requirements
    if (BK % 4 != 0 || BN % 4 != 0)
    {
        std::cerr << "Error: BK and BN must be multiples of 4 for float4." << std::endl;
        std::cout << "0" << std::endl;
        return 0;
    }

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes); // For cuBLAS baseline

    // Initialize matrices
    for (int i = 0; i < N * N; i++)
    {
        // Pseudo-randomizing slightly is better for catching logic errors than uniform 1.0fs
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_ref, bytes); // For cuBLAS baseline

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 grid((N + BM - 1) / BM, (N + BN - 1) / BN); // Added bounds safety
    dim3 block((BM / TM) * (BN / TN));

    // 1. Warm-up and Catch Launch Errors (Custom Kernel)
    matmul_tiled_2D_coarse_vec<<<grid, block>>>(N, N, N, d_A, d_B, d_C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Launch Failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "0" << std::endl;
        return 0;
    }

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Execution Failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "0" << std::endl;
        return 0;
    }

    // 2. cuBLAS Baseline & Verification
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // Note: cuBLAS is column-major. To compute C = A*B in row-major,
    // we compute C^T = B^T * A^T by passing B first, then A.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C_ref, N);

    cudaDeviceSynchronize();

    // Copy results back for comparison
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost);

    // Verify
    bool is_correct = verify_results(h_C, h_C_ref, N * N);
    if (!is_correct)
    {
        std::cout << "0" << std::endl; // Fail bash script
        return 0;
    }

    // 3. Timing Run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_tiled_2D_coarse_vec<<<grid, block>>>(N, N, N, d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print success time to stdout
    std::cout << milliseconds << std::endl;

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}