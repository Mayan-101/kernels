#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/kernel_6.cuh"

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
#ifndef _EXTRA_COLS
#define _EXTRA_COLS 0
#endif

#define BM _BM
#define BN _BN
#define BK _BK
#define TM _TM
#define TN _TN
#define EXTRA_COLS _EXTRA_COLS



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
    matmul_tiled<BK, BM, BN, TM, TN, EXTRA_COLS><<<grid, block>>>(N, N, N, 1.0f, d_A, d_B,0.0f,  d_C);

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
    matmul_tiled<BK, BM, BN, TM, TN, EXTRA_COLS><<<grid, block>>>(N, N, N, 1.0f, d_A, d_B,0.0f,  d_C);
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