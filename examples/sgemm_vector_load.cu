#include "utils.h"
#include "sgemm.cuh"

#include <assert.h>
#include <cstdio>
// #include <string>

void printMatrix(const float *mat, char *s, int height, int width,
                 int end_row, int end_col, int start_row = 0, int start_col = 0)
{
    assert(start_row >= 0 && start_col >= 0 && end_row <= height && end_col <= width);
    printf("\nmatrix %s: width=%d, height=%d, start_row=%d, end_row=%d, start_col=%d, end_col=%d\n",
           s, width, height, start_row, end_row, start_col, end_col);
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = start_col; j < end_col; j++)
        {
            printf("%g\t", mat[i * width + j]);
        }
        printf("\n");
    }
}

void timingMatMul(const float *A, const float *B, float *C, const int M, const int N, const int K,
                  const float alpha, const float beta)
{
    constexpr int REPEAT_NUM = 100;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < REPEAT_NUM; ++i)
    {
        gemm::launchSgemmSmemKernel_v4(A, B, C, M, N, K, alpha, beta);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("alogrithm: vector load, elapsed_time: %g ms\n", elapsed_time / REPEAT_NUM);
}

int main(int argc, char *argv[])
{
    const int M = 1024;
    const int N = 2048;
    const int K = 512;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *h_a = new float[M * K];
    float *h_b = new float[N * K];
    float *h_c = new float[M * N];

    for (int i = 0; i < M * K; ++i)
    {
        h_a[i] = -(i % 512) + (i % 2048) * 0.1f;
    }

    for (int i = 0; i < N * K; ++i)
    {
        h_b[i] = (i % 1024) - (i % 4096) * 0.7f;
    }

    // printMatrix(h_a, (char *)("Matrix A: "), M, K, M-16, K-16, M-32, K-32);
    // printMatrix(h_b, (char *)("Matrix B: "), K, N, K-16, N-16, K-32, N-32);

    float *d_a;
    float *d_b;
    float *d_c;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_a, sizeof(float) * M * K));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b, sizeof(float) * N * K));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_c, sizeof(float) * M * N));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(float) * N * K, cudaMemcpyHostToDevice));

    timingMatMul(d_a, d_b, d_c, M, N, K, alpha, beta);

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    // printMatrix(h_c, (char *)("Matrix C: "), M, N, M - 16, N - 16, M - 32, N - 32);
    // printMatrix(h_c, (char *)("Matrix C: "), M, N, 32, 32, 0, 0);

    return 0;
}
