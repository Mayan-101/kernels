#include <stdio.h>
#define warp_size 32
//  ─ 1. NAIVE KERNEL                              
__global__ void softmax_naive(int R, int C, float *input, float temperature, float *output) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < R) {
        float max_val = input[row * C];
        for (int i = 1; i < C; i++)
            max_val = fmaxf(max_val, input[row * C + i] / temperature);
            
        float denom = 0.f;
        for (int i = 0; i < C; i++)
            denom += expf(input[row * C + i] / temperature - max_val);
            
        for (int i = 0; i < C; i++)
            output[row * C + i] = expf(input[row * C + i] / temperature - max_val) / denom;
    }
}

//  ─ 2. SHARED MEMORY KERNEL                          
__global__ void softmax_shared(int R, int C, float *input, float temperature, float *output) {
    int warp_id = threadIdx.x / warp_size;   
    int lane_id = threadIdx.x % warp_size;   

    __shared__ float sf[warp_size * warp_size];
    __shared__ float max_val[warp_size];
    __shared__ float denom[warp_size];

    int row = blockIdx.x * (blockDim.x / warp_size) + warp_id;
    bool valid_row = (row < R); 

    if (lane_id == 0) {
        max_val[warp_id] = -3.402823466e+38f;
        denom[warp_id]   = 0.f;
    }
    __syncthreads();

    int phases = (C + warp_size - 1) / warp_size; 

    
    for (int phase = 0; phase < phases; phase++) {
        int col = phase * warp_size + lane_id;
        
        // Read safely
        float val = (valid_row && col < C) ? input[row * C + col] / temperature : -3.402823466e+38f;
        sf[warp_id * warp_size + lane_id] = val;
        __syncthreads();

        if (lane_id == 0 && valid_row) {
            float new_max = max_val[warp_id];
            for (int i = 0; i < warp_size; i++) {
                if (phase * warp_size + i < C) 
                    new_max = fmaxf(new_max, sf[warp_id * warp_size + i]);
            }

            float new_denom = denom[warp_id] * expf(max_val[warp_id] - new_max);
            for (int i = 0; i < warp_size; i++) {
                if (phase * warp_size + i < C) 
                    new_denom += expf(sf[warp_id * warp_size + i] - new_max);
            }

            max_val[warp_id] = new_max;
            denom[warp_id]   = new_denom;
        }
        __syncthreads();
    }

    
    for (int phase = 0; phase < phases; phase++) {
        int col = phase * warp_size + lane_id;
        float val = (valid_row && col < C) ? input[row * C + col] / temperature : 0.0f;
        sf[warp_id * warp_size + lane_id] = val;
        __syncthreads();
        
        if (valid_row && col < C) {
            output[row * C + col] = expf(sf[warp_id * warp_size + lane_id] - max_val[warp_id]) / denom[warp_id];
        }
        __syncthreads();
    }
}


__forceinline__ __device__ float warp_reduce_max(float val) {
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__forceinline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warp_size / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_warp_shfl(int R, int C, float *input, float temperature, float *output) {
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int row = blockIdx.x * warps_per_block + warp_id;

    if (row >= R) return; 

    const float *in_row  = input  + (size_t)row * C;
          float *out_row = output + (size_t)row * C;

    float thread_max = -3.402823466e+38f;
    for (int col = lane_id; col < C; col += warp_size)
        thread_max = fmaxf(thread_max, in_row[col] / temperature);

    float row_max = warp_reduce_max(thread_max);

    float thread_sum = 0.f;
    for (int col = lane_id; col < C; col += warp_size)
        thread_sum += expf(in_row[col] / temperature - row_max);

    float row_sum = warp_reduce_sum(thread_sum);

    for (int col = lane_id; col < C; col += warp_size)
        out_row[col] = expf(in_row[col] / temperature - row_max) / row_sum;
}
