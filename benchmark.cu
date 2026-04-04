#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernels/kernel_5.cuh"


//  Macros


#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA] %s  @ %s:%d\n",                            \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[cuBLAS] error %d  @ %s:%d\n",                    \
                    (int)st, __FILE__, __LINE__);                               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


//  Config


static const int   WARMUP_RUNS   = 1;
static const int   BENCH_RUNS    = 10;
static const float REL_TOL       = 1e-3f;   // max relative error vs cuBLAS
static const float ABS_TOL       = 1e-5f;   // abs floor to avoid div-by-zero

// Tile sizes — must match the template parameters you call the kernel with
static const int BM = 64;
static const int BN = 64;
static const int BK = 64;
static const int TM = 8;
static const int TN = 8;


//  Helpers


static void fill_random(float *h, size_t n)
{
    for (size_t i = 0; i < n; i++)
        h[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

// Returns elapsed milliseconds between two recorded events.
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}


//  Correctness check  (compares kernel output against cuBLAS reference)


static bool check_correctness(const float *ref, const float *out,
                               int M, int N, float rel_tol, float abs_tol)
{
    int   errors     = 0;
    float max_rel    = 0.f;

    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float denom = fmaxf(fabsf(ref[i]), abs_tol);
        float rel   = diff / denom;
        if (rel > rel_tol) {
            if (errors < 5)   // print first 5 mismatches only
                printf("  [mismatch] idx=%d  ref=%.6f  got=%.6f  rel=%.2e\n",
                       i, ref[i], out[i], rel);
            errors++;
        }
        if (rel > max_rel) max_rel = rel;
    }

    printf("  max relative error : %.2e  (%s)\n",
           max_rel, errors == 0 ? "PASS" : "FAIL");
    return errors == 0;
}


//  Benchmark one (M, K, N) problem size


static void run_benchmark(cublasHandle_t handle, int M, int K, int N,
                           float alpha, float beta)
{
    printf("  M=%-5d  K=%-5d  N=%-5d   α=%.1f  β=%.1f\n", M, K, N, alpha, beta);

    const size_t szA = (size_t)M * K;
    const size_t szB = (size_t)K * N;
    const size_t szC = (size_t)M * N;
    const double flops = 2.0 * M * K * N;   // multiply-add pairs

    // host buffers 
    float *hA = (float *)malloc(szA * sizeof(float));
    float *hB = (float *)malloc(szB * sizeof(float));
    float *hC = (float *)malloc(szC * sizeof(float));
    float *hC_ref = (float *)malloc(szC * sizeof(float));
    float *hC_out = (float *)malloc(szC * sizeof(float));

    fill_random(hA, szA);
    fill_random(hB, szB);
    fill_random(hC, szC);

    //  device buffers 
    float *dA, *dB, *dC_ref, *dC_ker;
    CUDA_CHECK(cudaMalloc(&dA,     szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB,     szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_ref, szC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_ker, szC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA,     hA, szA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,     hB, szB * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_ref, hC, szC * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_ker, hC, szC * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    
    //  cuBLAS reference
    
    {
        // warmup
        for (int r = 0; r < WARMUP_RUNS; r++)
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC_ref, N));

        CUDA_CHECK(cudaMemcpy(dC_ref, hC, szC * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int r = 0; r < BENCH_RUNS; r++)
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dB, N,
                dA, K,
                &beta,
                dC_ref, N));
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms_total = elapsed_ms(ev_start, ev_stop);
        float ms_avg   = ms_total / BENCH_RUNS;
        double tflops  = flops / (ms_avg * 1e-3) / 1e12;

        printf("  [cuBLAS]           avg = %8.3f ms   %6.2f TFLOPS\n",
               ms_avg, tflops);

        CUDA_CHECK(cudaMemcpy(hC_ref, dC_ref, szC * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    
    //  Custom kernel
    
    {
        const int THREADS = (BM/TM) * (BN/TN); 
        
        // Use ceiling division for grid to handle non-multiples of BM/BN safely
        dim3 block(THREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        // warmup
        // Moved cudaMemcpy outside the warmup loop
        CUDA_CHECK(cudaMemcpy(dC_ker, hC, szC * sizeof(float), cudaMemcpyHostToDevice));
        for (int r = 0; r < WARMUP_RUNS; r++) {
            matmul_tiled<BK, BM, BN, TM, TN><<<grid, block>>>(
                M, K, N, alpha, dA, dB, beta, dC_ker);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // reset C to original before timed run
        CUDA_CHECK(cudaMemcpy(dC_ker, hC, szC * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int r = 0; r < BENCH_RUNS; r++) {
            matmul_tiled<BK, BM, BN, TM, TN><<<grid, block>>>(
                M, K, N, alpha, dA, dB, beta, dC_ker);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float ms_total = elapsed_ms(ev_start, ev_stop);
        float ms_avg   = ms_total / BENCH_RUNS;
        double tflops  = flops / (ms_avg * 1e-3) / 1e12;

        printf("  [matmul_tiled]     avg = %8.3f ms   %6.2f TFLOPS\n",
               ms_avg, tflops);

        CUDA_CHECK(cudaMemcpy(hC_out, dC_ker, szC * sizeof(float),
                              cudaMemcpyDeviceToHost));

        check_correctness(hC_ref, hC_out, M, N, REL_TOL, ABS_TOL);
    }

    // cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_ref));
    CUDA_CHECK(cudaFree(dC_ker));
    free(hA); free(hB); free(hC); free(hC_ref); free(hC_out);
}


//  main


int main(void)
{
    srand(42);

    // device info 
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device : %s  (SM %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("SMs    : %d   |   Global mem : %.1f GB\n",
           prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1 << 30));
           
    // Adjusted the print statement to properly calculate threads per block
    printf("Tile config : BM=%d  BK=%d  BN=%d   "
           "threads/block=%d\n\n", BM, BK, BN, (BM/TM) * (BN/TN));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    
    const int sizes[][2] = {
        {2048, 2048},
        {4096, 4096},
    };
    const float alpha = 1.0f, beta = 0.0f;

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        run_benchmark(handle, sizes[i][0], sizes[i][0], sizes[i][0], alpha, beta);

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\nDone.\n");
    return 0;
}