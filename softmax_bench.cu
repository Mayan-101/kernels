#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>          
#include "softmax_playground.cuh"
#define warp_size 32
#define BK 32

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)



static float measure_kernel(void (*fn)(int, int, float*, float, float*),
                            int R, int C, float *d_in, float *d_out,
                            int block, int is_warp_mapped, int warmup, int reps)
{
    int grid;
    if (is_warp_mapped) {
        int warps_per_block = block / warp_size;
        grid = (R + warps_per_block - 1) / warps_per_block;
    } else {
        grid = (R + block - 1) / block;
    }

    for (int i = 0; i < warmup; i++)
        fn<<<grid, block>>>(R, C, d_in, 1.f, d_out);
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

static void verify_kernels(int R, int C, float *d_in, float *d_out) {
    long N = (long)R * C;
    float *h_naive  = (float*)malloc(N * sizeof(float));
    float *h_shared = (float*)malloc(N * sizeof(float));
    float *h_warp   = (float*)malloc(N * sizeof(float));

    int block_naive = 256;
    int grid_naive = (R + block_naive - 1) / block_naive;
    softmax_naive<<<grid_naive, block_naive>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaMemcpy(h_naive, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    int block_shared = 1024; 
    int grid_shared = (R + (block_shared / warp_size) - 1) / (block_shared / warp_size);
    softmax_shared<<<grid_shared, block_shared>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaMemcpy(h_shared, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    int block_warp = 256; 
    int grid_warp = (R + (block_warp / warp_size) - 1) / (block_warp / warp_size);
    softmax_warp_shfl<<<grid_warp, block_warp>>>(R, C, d_in, 1.f, d_out);
    CUDA_CHECK(cudaMemcpy(h_warp, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err_shared = 0.0f, max_err_warp = 0.0f;
    for (long i = 0; i < N; i++) {
        float diff_s = fabsf(h_naive[i] - h_shared[i]);
        if (diff_s > max_err_shared) max_err_shared = diff_s;
        float diff_w = fabsf(h_naive[i] - h_warp[i]);
        if (diff_w > max_err_warp) max_err_warp = diff_w;
    }

    const char *status_s = (max_err_shared < 1e-4f) ? "PASS" : "FAIL";
    const char *status_w = (max_err_warp < 1e-4f)   ? "PASS" : "FAIL";
    printf("  Verification %4d×%-4d : Shared %s (%.2e) | Warp %s (%.2e)\n", 
           R, C, status_s, max_err_shared, status_w, max_err_warp);

    free(h_naive);
    free(h_shared);
    free(h_warp);
}

int main(void) {
    const int WARMUP = 1, REPS = 5;
    int shapes[][2] = { {4096, 4096}, {2048, 8192}, {512, 1024}, {4096, 16192} };
    int n_shapes = 4;

    printf("%-18s  %-12s  %-12s  %-12s\n", "shape (RxC)", "naive (ms)", "shared (ms)", "warp shfl (ms)");
    printf("                                   \n");

    for (int s = 0; s < n_shapes; s++) {
        int R = shapes[s][0], C = shapes[s][1];
        long N = (long)R * C;

        float *h = (float*)malloc(N * sizeof(float));
        for (long i = 0; i < N; i++) h[i] = (float)rand() / RAND_MAX;

        float *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h, N * sizeof(float), cudaMemcpyHostToDevice));
        
        verify_kernels(R, C, d_in, d_out);    

        
        float ms_naive  = measure_kernel(softmax_naive, R, C, d_in, d_out, 256, 0, WARMUP, REPS);
        
        float ms_shared = measure_kernel(softmax_shared, R, C, d_in, d_out, 1024, 1, WARMUP, REPS);
        
        float ms_warp   = measure_kernel(softmax_warp_shfl, R, C, d_in, d_out, 256, 1, WARMUP, REPS);

        printf("%4d x %-12d  %-12.3f  %-12.3f  %-12.3f\n", R, C, ms_naive, ms_shared, ms_warp);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        free(h);
    }
    return 0;
}