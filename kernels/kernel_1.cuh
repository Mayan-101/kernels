#include "utils.cuh"

__global__ void mysgemm1(int M, int K, int N, float alpha, float *A, float *B, float beta, float *C)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int column = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < M && column < N)
    {
        float dot_prod = 0.f;
        for (int i = 0; i < K; i++)
        {
            dot_prod += A[row * K + i] * B[i * N + column];
        }
        C[row * N + column] = alpha * dot_prod + beta * C[row * N + column];
    }
}