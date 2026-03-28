#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define TM 8
#define BK 16
#define TN 8
#define BM 32
#define BN 64
#define BLOCK_SIZE 32
#define BLOCK_SIZE_2D 128

__global__ void matmul(int Rin, int C, int Rout, float *A, float *B, float *Out)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int column = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < Rin && column < Rout)
    {
        float dot_prod = 0.f;
        for (int i = 0; i < C; i++)
        {
            dot_prod += A[row * C + i] * B[i * Rout + column];
        }
        Out[row * Rout + column] = dot_prod;
    }
}

__global__ void matmul_1D(int Rin, int C, int Rout, float *A, float *B, float *Out)
{
    uint const x = threadIdx.x / BLOCK_SIZE + BLOCK_SIZE * blockIdx.y;
    uint const y = threadIdx.x % BLOCK_SIZE + BLOCK_SIZE * blockIdx.x;
    if (x < Rin && y < Rout)
    {
        float dot_prod = 0.f;
        for (int i = 0; i < C; i++)
        {
            dot_prod += A[x * C + i] * B[i * Rout + y];
        }
        Out[x * Rout + y] = dot_prod;
    }
}

__global__ void matmul_tiled(int N, float *A, float *B, float *C)
{
    uint ly = threadIdx.x / BLOCK_SIZE;
    uint lx = threadIdx.x % BLOCK_SIZE;

    uint const row = threadIdx.x / BLOCK_SIZE + BLOCK_SIZE * blockIdx.y;
    uint const column = threadIdx.x % BLOCK_SIZE + BLOCK_SIZE * blockIdx.x;

    __shared__ float sh_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sh_B[BLOCK_SIZE][BLOCK_SIZE];
    float dot_prod = 0.f;
    for (int phase = 0; phase < N / BLOCK_SIZE; phase++)
    {

        sh_A[ly][lx] = A[lx + phase * BLOCK_SIZE + N * row];
        sh_B[ly][lx] = B[N * (phase * BLOCK_SIZE + ly) + column];
        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; i++)
            dot_prod += sh_A[ly][i] * sh_B[i][lx];
        __syncthreads();
    }
    C[row * N + column] = dot_prod;
}

__global__ void matmul_tiled_1D_coarse(int N, float *A, float *B, float *C)
{
    uint ly = threadIdx.x / BK;
    uint lx = threadIdx.x % BK;
    uint by = threadIdx.x / BN;
    uint bx = threadIdx.x % BN;

    __shared__ float sh_A[BM*BK];
    __shared__ float sh_B[BK*BN];
    float threadResults[TM] = {0.f};

    for (int phase = 0; phase < N / BK; phase++)
    {
        sh_A[ly*BK + lx] = A[(blockIdx.x * BM + ly) * N + phase * BK + lx];
        sh_B[by*BN + bx] = B[(phase * BK + by) * N + blockIdx.y * BN + bx];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++)
        {
            float Btmp = sh_B[dotIdx*BN+bx];
            for (int resIdx = 0; resIdx < TM; resIdx++)
                threadResults[resIdx] += sh_A[(by * TM + resIdx)*BK +dotIdx] * Btmp;
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++)
        C[(blockIdx.x * BM + by * TM + resIdx) * N + blockIdx.y * BN + bx] = threadResults[resIdx];
}


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

    constexpr int stride_A = BLOCK_SIZE_2D / (BK / 4);   
    constexpr int stride_B = BLOCK_SIZE_2D / (BN / 4);

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

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

int main()
{
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int NSIZE = 5;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int s = 0; s < NSIZE; s++)
    {
        int N = sizes[s];
        size_t bytes = N * N * sizeof(float);

        float *h_A = new float[N * N];
        float *h_B = new float[N * N];
        float *h_C = new float[N * N];
        for (int i = 0; i < N * N; i++)
        {
            h_A[i] = 1.f;
            h_B[i] = 1.f;
        }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);

        cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

        //Warm-up for cuBLAS
        float alpha = 1.f, beta = 0.f;
        cudaEventRecord(start);
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N, 
                    d_A, N,
                    &beta,
                    d_C, N);
        cudaDeviceSynchronize();
        //Warm-up for CUDA
        dim3 block2dw(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid2dw((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul<<<grid2dw, block2dw>>>(N, N, N, d_A, d_B, d_C);
        cudaDeviceSynchronize();


        dim3 block2d(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid2d((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaEventRecord(start);
        matmul<<<grid2d, block2d>>>(N, N, N, d_A, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_naive;
        cudaEventElapsedTime(&ms_naive, start, stop);

        dim3 block1d(BLOCK_SIZE * BLOCK_SIZE);
        dim3 grid2d1((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaEventRecord(start);
        matmul_1D<<<grid2d1, block1d>>>(N, N, N, d_A, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_coal;
        cudaEventElapsedTime(&ms_coal, start, stop);

        dim3 grid_t1((N / BN), (N / BM));
        cudaEventRecord(start);
        matmul_tiled_2D_coarse_vec<<<grid_t1, (BN/TN) *(BM / TM)>>>(N, N, N, d_A, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_2D;
        cudaEventElapsedTime(&ms_2D, start, stop);

        dim3 grid_((N/BN), (N / BM));
        cudaEventRecord(start);
        matmul_tiled_1D_coarse<<<grid_, (BN) *(BM / TM)>>>(N, d_A, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_1D;
        cudaEventElapsedTime(&ms_1D, start, stop);

        dim3 grid_t((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        cudaEventRecord(start);
        matmul_tiled<<<grid_t, BLOCK_SIZE * BLOCK_SIZE>>>(N, d_A, d_B, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_tiled;
        cudaEventElapsedTime(&ms_tiled, start, stop);
        cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N, 
                    d_A, N,
                    &beta,
                    d_C, N);
        cudaEventRecord(stop);
        float ms_cublas;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_cublas, start, stop);

        bool ok = true;
        for (int i = 0; i < N * N && ok; i++)
            if (h_C[i] != (float)N)
            {
                ok = false;
            }

        float gb = 3.f * bytes / 1e9f;
        printf("N=%4d  naive:%6.1fms(%5.1fGB/s)  coal:%6.1fms(%5.1fGB/s)"
               "  tiled:%6.1fms(%5.1fGB/s)  tiled_1d:%6.1fms(%5.1fGB/s)   tiled_2d_vec:%6.1fms(%5.1fGB/s)   cuBLAS:%6.1fms(%5.1fGB/s) verify:%s\n",
               N,
               ms_naive, gb / (ms_naive / 1000.f),
               ms_coal, gb / (ms_coal / 1000.f),
               ms_tiled, gb / (ms_tiled / 1000.f),
               ms_1D, gb / (ms_1D / 1000.f),
               ms_2D, gb / (ms_2D / 1000.f),
               ms_cublas, gb / (ms_cublas / 1000.f),
               ok ? "OK" : "FAIL");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    return 0;
}