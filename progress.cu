#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels/kernel_1.cuh"
#include "kernels/kernel_2.cuh"
#include "kernels/kernel_3.cuh"
#include "kernels/kernel_4.cuh"
#include "kernels/kernel_5.cuh"
#include "kernels/kernel_6.cuh"

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t _e = (x);                                                  \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA %s:%d — %s\n",                               \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(x)                                                        \
    do {                                                                       \
        cublasStatus_t _s = (x);                                               \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS %s:%d — status %d\n",                      \
                    __FILE__, __LINE__, (int)_s);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr uint K2_BK = 32;

static constexpr uint K3_BK = 32, K3_BM = 32, K3_BN = 32;

static constexpr uint K4_BK = 8,  K4_BM = 64,  K4_BN = 64,
                      K4_TM = 4,  K4_TN = 4;

static constexpr uint K5_BK = 16, K5_BM = 128, K5_BN = 128,
                      K5_TM = 8,  K5_TN = 8;

static constexpr uint K6_BK = 16, K6_BM = 128, K6_BN = 128,
                      K6_TM = 8,  K6_TN = 8,   K6_EC = 4;

template <typename F>
static float run_ms(F fn, int warmup, int reps)
{
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < reps; i++) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / reps;
}

static bool verify_result(float *d_out, float *d_ref, long N2,
                           float *h_out,  float *h_ref)
{
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref, d_ref, N2 * sizeof(float), cudaMemcpyDeviceToHost));

    double maxerr = 0., maxval = 0.;
    for (long i = 0; i < N2; i++) {
        double e = fabs((double)h_out[i] - (double)h_ref[i]);
        double v = fabs((double)h_ref[i]);
        if (e > maxerr) maxerr = e;
        if (v > maxval) maxval = v;
    }
    return (maxerr / (maxval + 1e-8)) < 1e-2;
}

static float gbps(int N, float ms)
{
    return 3.0 * (double)N * N * sizeof(float) / (ms * 1e-3) / 1e9;
}

int main(void)
{
    constexpr int WARMUP = 3, REPS = 10;
    const float alpha = 1.f, beta = 0.f;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    srand(42);

    const int sizes[] = {256, 512, 1024, 2048, 4096};
    const int nsizes  = (int)(sizeof sizes / sizeof *sizes);

    for (int s = 0; s < nsizes; s++) {
        const int  N  = sizes[s];
        const long N2 = (long)N * N;

        float *h_A   = (float *)malloc(N2 * sizeof(float));
        float *h_B   = (float *)malloc(N2 * sizeof(float));
        float *h_out = (float *)malloc(N2 * sizeof(float));
        float *h_ref = (float *)malloc(N2 * sizeof(float));

        for (long i = 0; i < N2; i++) {
            h_A[i] = (float)rand() / RAND_MAX - 0.5f;
            h_B[i] = (float)rand() / RAND_MAX - 0.5f;
        }

        float *d_A, *d_B, *d_C, *d_ref;
        CUDA_CHECK(cudaMalloc(&d_A,   N2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B,   N2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C,   N2 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref, N2 * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_A, h_A, N2 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, N2 * sizeof(float), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_ref, 0, N2 * sizeof(float)));
        CUBLAS_CHECK(cublasSgemm(cublas,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N,
                                 &alpha, d_B, N,
                                         d_A, N,
                                 &beta,  d_ref, N));
        CUDA_CHECK(cudaDeviceSynchronize());

        dim3 g1((N+31)/32, (N+31)/32), b1(32, 32);

        dim3 g2((N+K2_BK-1)/K2_BK, (N+K2_BK-1)/K2_BK), b2(K2_BK * K2_BK);

        dim3 g3((N+K3_BN-1)/K3_BN, (N+K3_BM-1)/K3_BM), b3(K3_BM * K3_BN);

        constexpr uint K4T = (K4_BM / K4_TM) * (K4_BN / K4_TN);
        dim3 g4((N+K4_BN-1)/K4_BN, (N+K4_BM-1)/K4_BM), b4(K4T);

        constexpr uint K5T = (K5_BM / K5_TM) * (K5_BN / K5_TN);
        dim3 g5((N+K5_BM-1)/K5_BM, (N+K5_BN-1)/K5_BN), b5(K5T);

        constexpr uint K6T = (K6_BM / K6_TM) * (K6_BN / K6_TN);
        dim3 g6((N+K6_BM-1)/K6_BM, (N+K6_BN-1)/K6_BN), b6(K6T);

        auto k1 = [&]{ mysgemm1<<<g1,b1>>>(N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto k2 = [&]{ mysgemm2<K2_BK><<<g2,b2>>>(N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto k3 = [&]{ mysgemm3<K3_BK,K3_BM,K3_BN><<<g3,b3>>>(
                           N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto k4 = [&]{ mysgemm4<K4_BK,K4_BM,K4_BN,K4_TM,K4_TN><<<g4,b4>>>(
                           N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto k5 = [&]{ mysgemm5<K5_BK,K5_BM,K5_BN,K5_TM,K5_TN><<<g5,b5>>>(
                           N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto k6 = [&]{ mysgemm6<K6_BK,K6_BM,K6_BN,K6_TM,K6_TN,K6_EC><<<g6,b6>>>(
                           N,N,N, alpha, d_A,d_B, beta, d_C); };

        auto kcub = [&]{
            cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
        };

        bool all_ok = true;
        auto go = [&](auto fn, bool do_check) -> float {
            fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            if (do_check)
                if (!verify_result(d_C, d_ref, N2, h_out, h_ref)) all_ok = false;
            return run_ms(fn, WARMUP, REPS);
        };

        printf("N=%4d", N);

        float ms;
        ms = go(k1,   true);
        printf("  naive:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(k2,   true);
        printf("  coal:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(k3,   true);
        printf("  tiled:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(k4,   true);
        printf("  tiled_1d:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(k5,   true);
        printf("  tiled_2d_vec:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(k6,   true);
        printf("  tiled_2d_vec_pad:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        ms = go(kcub, false);
        printf("  cuBLAS:%8.1fms(%5.1fGB/s)", ms, gbps(N, ms));

        printf("  verify:%s\n", all_ok ? "Ok" : "FAIL");
        fflush(stdout);

        free(h_A); free(h_B); free(h_out); free(h_ref);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_ref));
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}