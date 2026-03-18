#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

// ─── Auto-Tuning Macros ─────────────────────────────────────────────
#ifndef _BM
#define _BM 64
#endif
#ifndef _BN
#define _BN 64
#endif
#ifndef _BK
#define _BK 8
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

// ─── Generalized CUDA Kernel ─────────────────────────────────────────
__global__ void matmul_tiled_2D_coarse_vec(int N, float *A, float *B, float *C)
{
    // ── compute indices (unchanged from 2D kernel) ─────────────
    uint threadRow = threadIdx.x / (BN/TN);   // 0..7
    uint threadCol = threadIdx.x % (BN/TN);   // 0..7

    // ── load indices (float4 granularity) ──────────────────────
    uint innerRowA = threadIdx.x / (BK/4);    // 0..31
    uint innerColA = threadIdx.x % (BK/4);    // 0..1
    uint innerRowB = threadIdx.x / (BN/4);    // 0..3
    uint innerColB = threadIdx.x % (BN/4);    // 0..15

    // ── sh_A is now TRANSPOSED: [BK][BM] instead of [BM][BK] ──
    __shared__ float sh_AT[BK*BM];   // [8][64] — rows are dotIdx
    __shared__ float sh_B [BK*BN];   // [8][64] — unchanged

    float threadResults[TM * TN] = {0.f};
    float regM[TM] = {0.f};
    float regN[TN] = {0.f};

    for (int phase = 0; phase < N/BK; phase++)
    {
        // ── load sh_A with float4, transpose during load ────────
        for (int offset = 0; offset < BM; offset += 64/(BK/4)) {
            float4 tmp = reinterpret_cast<float4*>(
                &A[(blockIdx.x*BM + innerRowA + offset)*N
                   + phase*BK + innerColA*4])[0];
            // scatter into transposed layout: sh_AT[col][row]
            sh_AT[(innerColA*4 + 0)*BM+innerRowA + offset] = tmp.x;
            sh_AT[(innerColA*4 + 1)*BM+innerRowA + offset] = tmp.y;
            sh_AT[(innerColA*4 + 2)*BM+innerRowA + offset] = tmp.z;
            sh_AT[(innerColA*4 + 3)*BM+innerRowA + offset] = tmp.w;
        }

        // ── load sh_B with float4, no transpose needed ──────────
        for (int offset = 0; offset < BK; offset += 64/(BN/4)) {
            reinterpret_cast<float4*>(
                &sh_B[(innerRowB + offset)*BK+innerColB*4])[0] =
            reinterpret_cast<float4*>(
                &B[(phase*BK + innerRowB + offset)*N
                   + blockIdx.y*BN + innerColB*4])[0];
        }
        __syncthreads();

        // ── compute: vectorized SMEM→register loads ─────────────
        for (int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            // sh_AT[dotIdx][threadRow*TM .. +7] is contiguous → 2×float4
            reinterpret_cast<float4*>(regM    )[0] =
                reinterpret_cast<float4*>(&sh_AT[dotIdx*BK+threadRow*TM    ])[0];
            reinterpret_cast<float4*>(regM + 4)[0] =
                reinterpret_cast<float4*>(&sh_AT[dotIdx*BK+threadRow*TM + 4])[0];

            // sh_B[dotIdx][threadCol*TN .. +7] is contiguous → 2×float4
            reinterpret_cast<float4*>(regN    )[0] =
                reinterpret_cast<float4*>(&sh_B[dotIdx*BK+threadCol*TN    ])[0];
            reinterpret_cast<float4*>(regN + 4)[0] =
                reinterpret_cast<float4*>(&sh_B[dotIdx*BK+threadCol*TN + 4])[0];

            // outer product — pure register arithmetic
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    threadResults[m*TN + n] += regM[m] * regN[n];
        }
        __syncthreads();
    }

    // ── write TM×TN results ─────────────────────────────────────
   for (int m = 0; m < TM; m++)
    {
        // threadCol*TN is 8-aligned, so TN=8 elements = 2 float4s
        reinterpret_cast<float4 *>(
            &C[(blockIdx.x * BM + threadRow * TM + m) * N + blockIdx.y * BN + threadCol * TN])[0] =
            reinterpret_cast<float4 *>(&threadResults[m * TN])[0];
        reinterpret_cast<float4 *>(
            &C[(blockIdx.x * BM + threadRow * TM + m) * N + blockIdx.y * BN + threadCol * TN + 4])[0] =
            reinterpret_cast<float4 *>(&threadResults[m * TN + 4])[0];
    }
}


// ─── Harness & Timing Logic ─────────────────────────────────────────
int main(int argc, char** argv) 
{
    int N = 2048; 
    if (argc > 1) N = atoi(argv[1]);

    // Safety checks for vectorization requirements
    if (BK % 4 != 0 || BN % 4 != 0) {
        std::cerr << "Error: BK and BN must be multiples of 4 for float4." << std::endl;
        std::cout << "0" << std::endl; 
        return 0;
    }

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; 
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 grid(N / BM, N / BN); 
    dim3 block((BM / TM) * (BN / TN));

    // 1. Warm-up and Catch Launch Errors
    matmul_tiled_2D_coarse_vec<<<grid, block>>>(N, d_A, d_B, d_C);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Print error to stderr so we can see it, but it doesn't break the bash script
        std::cerr << "Launch Failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "0" << std::endl; // Bash script sees '0' and marks as FAILED
        return 0;
    }

    // Catch execution errors (like illegal memory access)
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Execution Failed: " << cudaGetErrorString(err) << std::endl;
        std::cout << "0" << std::endl;
        return 0;
    }

    // 2. Timing Run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matmul_tiled_2D_coarse_vec<<<grid, block>>>(N, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print success time to stdout
    std::cout << milliseconds << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}