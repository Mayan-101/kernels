#include <stdio.h>
#include <float.h>
#define BK 32

__global__ void softmax_naive(int R, int C, float *input, float temperture, float *output){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < R)
    {
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
        max_val[ly] = -FLT_MAX;
        denom[ly]   = 0.f;
    }
    __syncthreads();

    for (int phase = 0; phase < C / BK; phase++) {
        sf[ly * BK + lx] = input[(phase * BK + ly) * C + blockIdx.x * BK + lx];
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
        sf[ly * BK + lx] = input[(phase * BK + ly) * C + blockIdx.x * BK + lx];
        __syncthreads();
        output[(phase * BK + ly) * C + blockIdx.x * BK + lx] =
            expf(sf[ly * BK + lx] - max_val[ly]) / denom[ly];
        __syncthreads();
    }
}