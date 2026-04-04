#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>               
#define BK 32
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)



__global__ void softmax_naive(int R, int C, float *input, float temperature, float *output) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < R) {
        float max_val = input[row * C];
        for (int i = 1; i < C; i++)
            max_val = fmaxf(max_val, input[row * C + i]);
        float denom = 0.f;
        for (int i = 0; i < C; i++)
            denom += expf(input[row * C + i] - max_val);
        for (int i = 0; i < C; i++)
            output[row * C + i] = expf(input[row * C + i] - max_val) / denom;
    }
}

__global__ void softmax_shared(int R, int C, float *input, float temperature, float *output) {
    int ly = threadIdx.x / BK;
    int lx = threadIdx.x % BK;

    __shared__ float sf[BK * BK];
    __shared__ float max_val[BK];
    __shared__ float denom[BK];

    if (lx == 0) {
        max_val[ly] = -3.402823466e+38f;
        denom[ly]   = 0.f;
    }
    __syncthreads();

    for (int phase = 0; phase < C / BK; phase++) {
        sf[ly * BK + lx] = input[(blockIdx.x * BK + ly) * C + phase * BK  + lx];
        __syncthreads();

        if (lx == 0) {
            float new_max = max_val[ly];
            for (int i = 0; i < BK; i++)
                new_max = fmaxf(new_max, sf[ly * BK + i]);

            float new_denom = denom[ly] * expf(max_val[ly] - new_max);
            for (int i = 0; i < BK; i++)
                new_denom += expf(sf[ly * BK + i] - new_max);

            max_val[ly] = new_max;
            denom[ly]   = new_denom;
        }
        __syncthreads();
    }

    for (int phase = 0; phase < C / BK; phase++) {
        output[(blockIdx.x * BK + ly) * C + phase * BK + lx] =
            expf(input[(blockIdx.x * BK + ly) * C + phase * BK + lx] - max_val[ly]) / denom[ly];
    }
}



static float measure(void (*fn)(int, int, float*, float, float*),
                     int R, int C, float *d_in, float *d_out,
                     int block, int warmup, int reps)
{
    int grid = (R + BK - 1) / BK;
    for (int i = 0; i < warmup; i++)
        fn<<<grid, BK*BK>>>(R, C, d_in, 1.f, d_out);   // note: original used BK*BK here too, kept for consistency
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < reps; i++)
        fn<<<grid, block>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / reps;
}

static float measure_shared(int R, int C, float *d_in, float *d_out,
                             int warmup, int reps)
{
    int block = BK * BK;
    dim3 grid((R + BK - 1) / BK);
    softmax_shared<<<grid, block>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < reps; i++)
        softmax_shared<<<grid, block>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / reps;
}


static void verify_kernels(int R, int C, float *d_in, float *d_out, int block_naive)
{
    long N = (long)R * C;
    float *h_naive = (float*)malloc(N * sizeof(float));
    float *h_shared = (float*)malloc(N * sizeof(float));

    int grid = (R + BK - 1) / BK;
    int block_shared = BK * BK;

    // Run naive kernel → save result
    softmax_naive<<<grid, block_naive>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_naive, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Run shared kernel (overwrites d_out) → save result
    softmax_shared<<<grid, block_shared>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_shared, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare with reasonable floating-point tolerance
    float max_err = 0.0f;
    for (long i = 0; i < N; i++) {
        float diff = fabsf(h_naive[i] - h_shared[i]);
        if (diff > max_err) max_err = diff;
    }

    const char *status = (max_err < 1e-5f) ? "PASS" : "FAIL";
    printf("  Verification %4d×%-4d : %s (max error %.2e)\n", R, C, status, max_err);

    free(h_naive);
    free(h_shared);
}



int main(void) {
    const int WARMUP = 1, REPS = 1, BLOCK = 256;

    int shapes[][2] = { {4096, 4096}, {2048, 8192}, {512, 1024} , {4096, 16192}};
    int n_shapes = 4;

    printf("%-20s  %-12s  %-12s\n", "shape (RxC)", "naive (ms)", "shared (ms)");
    printf("──────────────────────────────────────────────────\n");

    for (int s = 0; s < n_shapes; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        long N = (long)R * C;

        float *h = (float*)malloc(N * sizeof(float));
        for (long i = 0; i < N; i++) h[i] = (float)rand() / RAND_MAX;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h, N * sizeof(float), cudaMemcpyHostToDevice));
        
        verify_kernels(R, C, d_in, d_out, BLOCK);    

        float ms_naive  = measure(softmax_naive, R, C, d_in, d_out, BLOCK, WARMUP, REPS);
        float ms_shared = measure_shared(R, C, d_in, d_out, WARMUP, REPS);

        printf("%4d x %-14d  %-12.3f  %-12.3f\n", R, C, ms_naive, ms_shared);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h);
    }

    return 0;
}