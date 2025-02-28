
#include "sgemm.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <mma.h>
#include <cuda/barrier>

namespace gemm
{

    /** naive gemm
     * grid((M+16-1)/16, (N+16-1)/16), block(16, 16)
     */
    __global__ void SgemmNaiveKernel(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                     const float alpha, const float beta)
    {
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < M && col < N)
        {
            float tmp = 0.0f;
            for (int i = 0; i < K; ++i)
            {
                tmp += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = alpha * tmp + beta * C[row * N + col];
        }
    }

    void launchSgemmNaiveKernel(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream)
    {
        dim3 block(32, 32);
        dim3 gird((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        SgemmNaiveKernel<<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    /** 共享内存、分片
     * dim3 block(BN, BM);
     * dim3 gird((N+block.x-1)/block.x, (M+block.y-1)/block.y);
     */
    template <int BM, int BN, int BK>
    __global__ void SgemmSmemKernel_v1(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        const int offset_a = blockIdx.y * blockDim.y * K;
        const int offset_b = blockIdx.x * blockDim.x;
        const int offset_c = blockIdx.y * blockDim.y * N + blockIdx.x * blockDim.x;

        extern __shared__ float s_ptr[];
        float *s_a = s_ptr;
        float *s_b = s_a + BM * BK;

        float tmp = 0.0f;
#pragma unroll
        for (int i = 0; i < K; i += BK)
        {
#pragma unroll
            for (int j = threadIdx.x; j < BK; j += blockDim.x)
            {
                s_a[threadIdx.y * BK + j] = A[offset_a + threadIdx.y * K + i + j];
            }
#pragma unroll
            for (int j = threadIdx.y; j < BK; j += blockDim.y)
            {
                s_b[threadIdx.x + BK * j] = B[offset_b + i * N + j * N + threadIdx.x];
            }
            __syncthreads();

#pragma unroll
            for (int j = 0; j < BK; ++j)
            {
                tmp += s_a[threadIdx.y * BK + j] * s_b[threadIdx.x + BK * j];
            }
            __syncthreads();
        }
        C[offset_c + threadIdx.y * N + threadIdx.x] = tmp * alpha + beta * C[offset_c + threadIdx.y * N + threadIdx.x];
    }

    void launchSgemmSmemKernel_v1(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 32;
        const int BN = 32;
        const int BK = 32;
        dim3 block(BN, BM);
        dim3 gird((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        size_t smem = sizeof(float) * (BM * BK + BN * BK);
        SgemmSmemKernel_v1<BM, BN, BK><<<gird, block, smem, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    /** 每个线程计算多个元素，一维
     * dim3 block(BM * BN / TM = 512);
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int TM>
    __global__ void SgemmSmemKernel_v2(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        assert(BM * BK == blockDim.x);
        assert(BN * BK == blockDim.x);

        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

        const int threadCol = threadIdx.x % BN;
        const int threadRow = threadIdx.x / BN;

        __shared__ float s_a[BM * BK];
        __shared__ float s_b[BK * BN];

        float tmp[TM] = {0.0f};
        int row_a = threadIdx.x / BK;
        int col_a = threadIdx.x % BK;
        int row_b = threadIdx.x / BN;
        int col_b = threadIdx.x % BN;
#pragma unroll
        for (int i = 0; i < K; i += BK)
        {
            s_a[threadIdx.x] = A[offset_a + row_a * K + i + col_a];
            s_b[threadIdx.x] = B[offset_b + row_b * N + i * N + col_b];
            __syncthreads();

#pragma unroll
            for (int j = 0; j < BK; ++j)
            {
                float tmp_b = s_b[j * BN + (threadCol)];
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
                    tmp[tmidx] += s_a[(threadRow * TM + tmidx) * BK + j] * tmp_b;
                }
            }
            __syncthreads();
        }

        for (int i = 0; i < TM; ++i)
        {
            int row_c = (threadRow * TM) + i;
            int col_c = (threadCol);
            C[offset_c + row_c * N + col_c] = alpha * tmp[i] + beta * C[offset_c + row_c * N + col_c];
        }
    }

    void launchSgemmSmemKernel_v2(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 64;
        const int BN = 64;
        const int BK = 8;
        const int TM = 8;
        dim3 block(BN * BM / TM);
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
        SgemmSmemKernel_v2<BM, BN, BK, TM><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    /** 每个线程计算多个元素，拓展到二维
     * dim3 block(BM * BN / (TM * TN));
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void SgemmSmemKernel_v3(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        int row_a, col_a, row_b, col_b;

        // 通过 Padding 避免 bank conflict
        const int extra_col = 4;
        __shared__ float s_a[BM * BK];
        __shared__ float s_b[BK * (BN + extra_col)];

        float tmp[TM * TN] = {0.0f};
        float reg_a[TM];
        float reg_b[TN];

#pragma unroll
        for (int i = 0; i < K; i += BK)
        {
#pragma unroll
            for (int j = threadIdx.x; j < BM * BK; j += blockDim.x)
            {
                row_a = j / BK;
                col_a = j % BK;
                s_a[j] = A[offset_a + row_a * K + i + col_a];
            }

#pragma unroll
            for (int j = threadIdx.x; j < BK * BN; j += blockDim.x)
            {
                row_b = j / BN;
                col_b = j % BN;
                s_b[row_b * (BN + extra_col) + col_b] = B[offset_b + i * N + row_b * N + col_b];
            }

            __syncthreads();

#pragma unroll
            for (int j = 0; j < BK; ++j)
            {
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
                    reg_a[tmidx] = s_a[threadRow * TM * BK + tmidx * BK + j];
                }
#pragma unroll
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    reg_b[tnidx] = s_b[j * (BN + extra_col) + threadCol * TN + tnidx];
                }

#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
#pragma unroll
                    for (int tnidx = 0; tnidx < TN; ++tnidx)
                    {
                        tmp[tmidx * TN + tnidx] += reg_a[tmidx] * reg_b[tnidx];
                    }
                }
            }

            __syncthreads();
        }

        int row_c, col_c;
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                row_c = threadRow * TM + i;
                col_c = threadCol * TN + j;
                C[offset_c + row_c * N + col_c] = alpha * tmp[i * TN + j] + beta * C[offset_c + row_c * N + col_c];
            }
        }
    }

    void launchSgemmSmemKernel_v3(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;
        dim3 block(BN * BM / (TM * TN));
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
        SgemmSmemKernel_v3<BM, BN, BK, TM, TN><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    /** 向量化加载
     * dim3 block(BM * BN / (TM * TN));
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void SgemmSmemKernel_v4(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + blockIdx.x * BN;

        const int threadCol = threadIdx.x % (BN / TN);
        const int threadRow = threadIdx.x / (BN / TN);

        int row_a, col_a, row_b, col_b;

        // 通过 Padding 避免 bank conflict
        const int extra_col = 4;
        __shared__ float s_a[BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
        __shared__ float s_b[BK * (BN + extra_col)];

        float tmp[TM * TN] = {0.0f};
        float reg_a[TM];
        float reg_b[TN];

#pragma unroll
        for (int i = 0; i < K; i += BK)
        {
            // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
            for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
            {
                row_a = j / BK;
                col_a = j % BK;
                float4 tmp = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + i + col_a)[0];

                s_a[col_a * (BM + extra_col) + row_a] = tmp.x;
                s_a[(col_a + 1) * (BM + extra_col) + row_a] = tmp.y;
                s_a[(col_a + 2) * (BM + extra_col) + row_a] = tmp.z;
                s_a[(col_a + 3) * (BM + extra_col) + row_a] = tmp.w;
            }

#pragma unroll
            for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
            {
                row_b = j / BN;
                col_b = j % BN;
                reinterpret_cast<float4 *>(s_b + row_b * (BN + extra_col) + col_b)[0] = reinterpret_cast<const float4 *>(B + offset_b + i * N + row_b * N + col_b)[0];
            }

            __syncthreads();

#pragma unroll
            for (int j = 0; j < BK; ++j)
            {
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
                    reg_a[tmidx] = s_a[threadRow * TM + tmidx + j * (BM + extra_col)];
                }
#pragma unroll
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    reg_b[tnidx] = s_b[j * (BN + extra_col) + threadCol * TN + tnidx];
                }

#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
#pragma unroll
                    for (int tnidx = 0; tnidx < TN; ++tnidx)
                    {
                        tmp[tmidx * TN + tnidx] += reg_a[tmidx] * reg_b[tnidx];
                    }
                }
            }
            __syncthreads();
        }

        int row_c, col_c;
#pragma unroll
        for (int i = 0; i < TM; ++i)
        {
#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                row_c = threadRow * TM + i;
                col_c = threadCol * TN + j;
                C[offset_c + row_c * N + col_c] = alpha * tmp[i * TN + j] + beta * C[offset_c + row_c * N + col_c];
            }
        }
    }

    void launchSgemmSmemKernel_v4(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;
        dim3 block(BN * BM / (TM * TN));
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
        SgemmSmemKernel_v4<BM, BN, BK, TM, TN><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    template <int BM, int BN, int BK>
    __device__ void loadFromGmem(const float *A, const float *B, const int M, const int N, const int K,
                                 float *s_a, float *s_b, int offset_a, int offset_b,
                                 const int bkidx, const int buffer_id, const int extra_col)
    {
        int row_a, col_a, row_b, col_b;
        float4 tmp;
        // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
        for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
        {
            row_a = j / BK;
            col_a = j % BK;

            // 每次从 global mem 加载 4 个 float 到 shared mem
            tmp = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + bkidx + col_a)[0];
            s_a[buffer_id * BK * (BM + extra_col) + col_a * (BM + extra_col) + row_a] = tmp.x;
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 1) * (BM + extra_col) + row_a] = tmp.y;
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 2) * (BM + extra_col) + row_a] = tmp.z;
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 3) * (BM + extra_col) + row_a] = tmp.w;
        }

#pragma unroll
        for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
        {
            row_b = j / BN;
            col_b = j % BN;
            reinterpret_cast<float4 *>(s_b + buffer_id * BK * (BN + extra_col) + row_b * (BN + extra_col) + col_b)[0] =
                reinterpret_cast<const float4 *>(B + offset_b + bkidx * N + row_b * N + col_b)[0];
        }
    }

    template <int BM, int BN, int BK>
    __device__ void loadFromGmemToSmemAndConvertToHalf(const float *A, const float *B, const int M, const int N, const int K,
                                                       half *s_a, half *s_b, int offset_a, int offset_b,
                                                       const int bkidx, const int buffer_id, const int extra_col)
    {
        int row_a, col_a, row_b, col_b;
        float4 tmp4;
        // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
        for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
        {
            row_a = j / BK;
            col_a = j % BK;
            // 每次从 global mem 加载 4 个 float 到 shared mem
            tmp4 = reinterpret_cast<const float4 *>(A + offset_a + row_a * K + bkidx + col_a)[0];
            s_a[buffer_id * BK * (BM + extra_col) + col_a * (BM + extra_col) + row_a] = __float2half(tmp4.x);
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 1) * (BM + extra_col) + row_a] = __float2half(tmp4.y);
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 2) * (BM + extra_col) + row_a] = __float2half(tmp4.z);
            s_a[buffer_id * BK * (BM + extra_col) + (col_a + 3) * (BM + extra_col) + row_a] = __float2half(tmp4.w);
        }

        float2 tmp2;
#pragma unroll
        for (int j = (threadIdx.x << 1); j < BK * BN; j += (blockDim.x << 1))
        {
            row_b = j / BN;
            col_b = j % BN;
            tmp2 = reinterpret_cast<const float2 *>(B + offset_b + bkidx * N + row_b * N + col_b)[0];
            reinterpret_cast<half2 *>(s_b + buffer_id * BK * (BN + extra_col) + row_b * (BN + extra_col) + col_b)[0] = __float22half2_rn(tmp2);
        }
    }

    template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    __device__ void processFromSmem(const float *s_a, const float *s_b, float *reg_a, float *reg_b, float *tmp, const int WMITERS, const int WNITERS,
                                    const int warp_sub_M, const int warp_sub_N, const int warp_row, const int warp_col, const int thread_in_warp_row,
                                    const int thread_in_warp_col, const int buffer_id, const int extra_col)
    {
#pragma unroll
        for (int j = 0; j < BK; ++j)
        {
            // 从 shared mem 加载到 register
#pragma unroll
            for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
            {
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
                    reg_a[w_m_id * TM + tmidx] =
                        s_a[buffer_id * BK * (BM + extra_col) + j * (BM + extra_col) + warp_row * WM + w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx];
                }
            }

#pragma unroll
            for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
            {
#pragma unroll
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    reg_b[w_n_id * TN + tnidx] =
                        s_b[buffer_id * BK * (BN + extra_col) + j * (BN + extra_col) + warp_col * WN + w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx];
                }
            }

#pragma unroll
            for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
            {
#pragma unroll
                for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
                {
#pragma unroll
                    for (int tmidx = 0; tmidx < TM; ++tmidx)
                    {
#pragma unroll
                        for (int tnidx = 0; tnidx < TN; ++tnidx)
                        {
                            tmp[w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx] +=
                                reg_a[w_m_id * TM + tmidx] * reg_b[w_n_id * TN + tnidx];
                        }
                    }
                }
            }
        }
    }

    template <int BM, int BN, int BK, int TM, int TN>
    __device__ void writeFromRegToGmem(const float *tmp, float *C, const int M, const int N, const int WMITERS, const int WNITERS,
                                       const int warp_sub_M, const int warp_sub_N, const int thread_in_warp_row, const int thread_in_warp_col,
                                       const int offset_c, const float alpha, const float beta)
    {
        float4 ret4;
        int row_c, col_c, tmp_id;
#pragma unroll
        for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
        {
#pragma unroll
            for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
            {
#pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
#pragma unroll
                    for (int tnidx = 0; tnidx < TN; tnidx += 4)
                    {

                        row_c = w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx;
                        col_c = w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx;
                        tmp_id = w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx;
                        ret4 = reinterpret_cast<float4 *>(C + offset_c + row_c * N + col_c)[0];
                        ret4.x = alpha * tmp[tmp_id] + beta * ret4.x;
                        ret4.y = alpha * tmp[tmp_id + 1] + beta * ret4.y;
                        ret4.z = alpha * tmp[tmp_id + 2] + beta * ret4.z;
                        ret4.w = alpha * tmp[tmp_id + 3] + beta * ret4.w;
                        reinterpret_cast<float4 *>(C + offset_c + row_c * N + col_c)[0] = ret4;
                    }
                }
            }
        }
    }

    /** warp 分片
     * dim3 block(BM * BN / (TM * TN));
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    __global__ void SgemmSmemKernel_v5(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        // block 内 warp 二维分布的 id
        const int warp_row = (threadIdx.x >> 5) / (BN / WN);
        const int warp_col = (threadIdx.x >> 5) % (BN / WN);
        // M、N 方向每个 warp 迭代次数
        const int WMITERS = 2;
        const int WNITERS = (WM * WN) / (TM * TN * 32 * WMITERS);
        // warp 每次迭代处理的 M、N 方向上的元素个数
        const int warp_sub_M = WM / WMITERS;
        const int warp_sub_N = WN / WNITERS;
        // warp 内 thread 二维分布的 id
        const int thread_in_warp_row = (threadIdx.x & 0x1f) / (warp_sub_N / TN);
        const int thread_in_warp_col = (threadIdx.x & 0x1f) % (warp_sub_N / TN);
        // warp 级偏移量
        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

        // 通过 Padding 避免 bank conflict
        const int extra_col = 4;
        __shared__ float s_a[BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
        __shared__ float s_b[BK * (BN + extra_col)];

        float tmp[WMITERS * TM * WNITERS * TN] = {0.0f};
        float reg_a[WMITERS * TM];
        float reg_b[WNITERS * TN];

#pragma unroll
        for (int i = 0; i < K; i += BK)
        {
            // 每次从 global mem 加载 4 个 float 到 shared mem
            loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i, 0, extra_col);
            __syncthreads();

            processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                        warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, 0, extra_col);
            __syncthreads();
        }

        writeFromRegToGmem<BM, BN, BK, TM, TN>(tmp, C, M, N, WMITERS, WNITERS, warp_sub_M, warp_sub_N, thread_in_warp_row, thread_in_warp_col, offset_c, alpha, beta);
    }

    void launchSgemmSmemKernel_v5(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int WM = 64;
        const int WN = 32;
        const int TM = 4;
        const int TN = 4;

        dim3 block(BN * BM / (WM * WN) * 32);
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);
        SgemmSmemKernel_v5<BM, BN, BK, WM, WN, TM, TN><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    template <int BM, int BN, int BK, typename T>
    __device__ void loadFromGmemBymemcpyAsync(const float *A, const float *B, const int M, const int N, const int K,
                                              float *s_a, float *s_b, int offset_a, int offset_b,
                                              const int bkidx, const int buffer_id, const int extra_col, T &barrier)
    {
        // 每次从 global mem 加载 4 个 float 到 shared mem
#pragma unroll
        for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
        {
            int row_a = j / BK;
            int col_a = j % BK;

            // 每次从 global mem 加载 4 个 float 到 shared mem
            cuda::memcpy_async(&s_a[buffer_id * BK * (BM + extra_col) + col_a * (BM + extra_col) + row_a],
                               &A[offset_a + row_a * K + bkidx + col_a],
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);
            cuda::memcpy_async(&s_a[buffer_id * BK * (BM + extra_col) + (col_a + 1) * (BM + extra_col) + row_a],
                               &A[offset_a + row_a * K + bkidx + col_a + 1],
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);
            cuda::memcpy_async(&s_a[buffer_id * BK * (BM + extra_col) + (col_a + 2) * (BM + extra_col) + row_a],
                               &A[offset_a + row_a * K + bkidx + col_a + 2],
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);
            cuda::memcpy_async(&s_a[buffer_id * BK * (BM + extra_col) + (col_a + 3) * (BM + extra_col) + row_a],
                               &A[offset_a + row_a * K + bkidx + col_a + 3],
                               cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);
        }

#pragma unroll
        for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
        {
            int row_b = j / BN;
            int col_b = j % BN;

            cuda::memcpy_async(&s_b[buffer_id * BK * (BN + extra_col) + row_b * (BN + extra_col) + col_b],
                               &B[offset_b + bkidx * N + row_b * N + col_b],
                               cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), barrier);
        }
    }

    /** double_buffer
     * dim3 block(BM * BN / (TM * TN), 2);
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
    __global__ void SgemmSmemKernel_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        //     printf("in SgemmSmemKernel_v6\n");
        // block 内 warp 二维分布的 id
        const int warp_row = (threadIdx.x >> 5) / (BN / WN);
        const int warp_col = (threadIdx.x >> 5) % (BN / WN);
        // M、N 方向每个 warp 迭代次数
        const int WMITERS = 2;
        const int WNITERS = (WM * WN) / (TM * TN * 32 * WMITERS);
        // warp 每次迭代处理的 M、N 方向上的元素个数
        const int warp_sub_M = WM / WMITERS;
        const int warp_sub_N = WN / WNITERS;
        // warp 内 thread 二维分布的 id
        const int thread_in_warp_row = (threadIdx.x & 0x1f) / (warp_sub_N / TN);
        const int thread_in_warp_col = (threadIdx.x & 0x1f) % (warp_sub_N / TN);
        // warp 级偏移量
        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

        // 通过 Padding 避免 bank conflict
        const int extra_col = 4;
        __shared__ float s_a[2 * BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
        __shared__ float s_b[2 * BK * (BN + extra_col)];

        float tmp[WMITERS * TM * WNITERS * TN] = {0.0f};
        float reg_a[WMITERS * TM];
        float reg_b[WNITERS * TN];

        int buffer_id = 0;

        loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, 0, buffer_id, extra_col);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < K - BK; i += BK)
        {
            loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i + BK, 1 - buffer_id, extra_col);
            processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                        warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, buffer_id, extra_col);
            buffer_id = 1 - buffer_id;
            __syncthreads();
        }

        processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                    warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, buffer_id, extra_col);

        __syncthreads();
        writeFromRegToGmem<BM, BN, BK, TM, TN>(tmp, C, M, N, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                               thread_in_warp_row, thread_in_warp_col, offset_c, alpha, beta);
    }

    void launchSgemmSmemKernel_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int WM = 64;
        const int WN = 32;
        const int TM = 4;
        const int TN = 4;

        assert(K % (2 * BK) == 0);

        dim3 block(BN * BM / (WM * WN) * 32);
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);

        // printf("in launchSgemmSmemKernel_v6: blockdimx: %d blockdimy: %d", block.x, block.y);
        SgemmSmemKernel_v6<BM, BN, BK, WM, WN, TM, TN><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

    template <int BM, int BN, int BK, int WM, int WN, int WMITERS, int WNITERS, int WKITERS, int WMMA_M, int WMMA_N, int WMMA_K, typename T1, typename T2, typename T3>
    __device__ void processFromSmemByTensorCore(const float *s_a, const float *s_b, T1 *a_frag, T2 *b_frag, T3 *acc_frag,
                                                const int warp_row, const int warp_col, const int buffer_id, const int extra_col)
    {
        using namespace nvcuda;
        int shm_offset;

        for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
        {
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                shm_offset = buffer_id * BK * (BM + extra_col) + wkidx * (BM + extra_col) + warp_row * WM + wmidx * WMMA_M;
                wmma::load_matrix_sync(a_frag[wmidx * WKITERS + wkidx], s_a + shm_offset, BM + extra_col);
            }
        }

        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                shm_offset = buffer_id * BK * (BN + extra_col) + wkidx * (BN + extra_col) + warp_col * WN + wnidx * WMMA_N;
                wmma::load_matrix_sync(b_frag[wnidx * WKITERS + wkidx], s_b + shm_offset, BN + extra_col);
            }
        }

        for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
        {
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
            {
                for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
                {
                    wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], a_frag[wmidx * WKITERS + wkidx],
                                   b_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
                }
            }
        }
    }

    template <int BM, int BN, int BK, int WM, int WN, int WMITERS, int WNITERS, int WKITERS, int WMMA_M, int WMMA_N, int WMMA_K, typename T1, typename T2, typename T3>
    __device__ void processFromSmemByTensorCore(const half *s_a, const half *s_b, T1 *a_frag, T2 *b_frag, T3 *acc_frag,
                                                const int warp_row, const int warp_col, const int buffer_id, const int extra_col)
    {
        using namespace nvcuda;
        int shm_offset;
        #pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
        {
            #pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                shm_offset = buffer_id * BK * (BM + extra_col) + wkidx * (BM + extra_col) + warp_row * WM + wmidx * WMMA_M;
                wmma::load_matrix_sync(a_frag[wmidx * WKITERS + wkidx], s_a + shm_offset, BM + extra_col);
            }
        }
        #pragma unroll
        for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
        {
            #pragma unroll
            for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
            {
                shm_offset = buffer_id * BK * (BN + extra_col) + wkidx * (BN + extra_col) + warp_col * WN + wnidx * WMMA_N;
                wmma::load_matrix_sync(b_frag[wnidx * WKITERS + wkidx], s_b + shm_offset, BN + extra_col);
            }
        }
        #pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
        {
            #pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
            {
                #pragma unroll
                for (int wkidx = 0; wkidx < WKITERS; ++wkidx)
                {
                    wmma::mma_sync(acc_frag[wmidx * WNITERS + wnidx], a_frag[wmidx * WKITERS + wkidx],
                                   b_frag[wnidx * WKITERS + wkidx], acc_frag[wmidx * WNITERS + wnidx]);
                }
            }
        }
    }

    template <int WMITERS, int WNITERS, int WMMA_M, int WMMA_N, typename T1, typename T2>
    __device__ void writeFromRegToGmemByTensorCore(T1 *c_frag, T2 *acc_frag, float *C, const int M, const int N,
                                                   const int offset_c, const float alpha, const float beta)
    {
        using namespace nvcuda;
        #pragma unroll
        for (int wmidx = 0; wmidx < WMITERS; ++wmidx)
        {
            #pragma unroll
            for (int wnidx = 0; wnidx < WNITERS; ++wnidx)
            {
                wmma::load_matrix_sync(c_frag[wmidx * WNITERS + wnidx], C + offset_c + wmidx * WMMA_M * N + wnidx * WMMA_N,
                                       N, wmma::mem_row_major);
                                       #pragma unroll
                for (int idx = 0; idx < c_frag[wmidx * WNITERS + wnidx].num_elements; ++idx)
                {
                    c_frag[wmidx * WNITERS + wnidx].x[idx] = alpha * acc_frag[wmidx * WNITERS + wnidx].x[idx] + beta * c_frag[wmidx * WNITERS + wnidx].x[idx];
                }
                wmma::store_matrix_sync(C + offset_c + wmidx * WMMA_M * N + wnidx * WMMA_N, c_frag[wmidx * WNITERS + wnidx],
                                        N, wmma::mem_row_major);
            }
        }
    }

    /** tensor core
     * dim3 block(BN * BM / (WM * WN) * 32);
     * dim3 grid((N+BN-1)/BN, (M+BM-1)/BM)
     */
    template <int BM, int BN, int BK, int WM, int WN, int WMMA_M, int WMMA_N, int WMMA_K>
    __global__ void SgemmSmemKernel_v7(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                       const float alpha, const float beta)
    {
        using namespace nvcuda;
        // block 内 warp 二维分布的 id
        const int warp_row = (threadIdx.x >> 5) / (BN / WN);
        const int warp_col = (threadIdx.x >> 5) % (BN / WN);
        // 单个 warp 处理层面 M、N、K 方向每个 warp 迭代次数
        constexpr int WMITERS = WM / WMMA_M;
        constexpr int WNITERS = WN / WMMA_N;
        constexpr int WKITERS = BK / WMMA_K;
        // warp 级偏移量
        const int offset_a = blockIdx.y * BM * K;
        const int offset_b = blockIdx.x * BN;
        const int offset_c = blockIdx.y * BM * N + warp_row * WM * N + blockIdx.x * BN + warp_col * WN;

        // 通过 Padding 避免 bank conflict
        constexpr int extra_col = 8;
        __shared__ half s_a[2 * BK * (BM + extra_col)]; // transpose s_a to [BK, BM + extra_col]
        __shared__ half s_b[2 * BK * (BN + extra_col)];

        using FragAType = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>;
        using FragBType = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
        using FragAccType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
        using FragCType = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
        FragAType a_frag[WMITERS * WKITERS];
        FragBType b_frag[WNITERS * WKITERS];
        FragAccType acc_frag[WMITERS * WNITERS];
        FragCType c_frag[WMITERS * WNITERS];

        #pragma unroll
        for (int i = 0; i < WMITERS * WNITERS; ++i)
        {
            wmma::fill_fragment(acc_frag[i], 0.0f);
        }

        int buffer_id = 0;

        loadFromGmemToSmemAndConvertToHalf<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, 0, buffer_id, extra_col);
        __syncthreads();

#pragma unroll
        for (int i = 0; i < K - BK; i += BK)
        {
            loadFromGmemToSmemAndConvertToHalf<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, i + BK, 1 - buffer_id, extra_col);

            processFromSmemByTensorCore<BM, BN, BK, WM, WN, WMITERS, WNITERS, WKITERS, WMMA_M, WMMA_N, WMMA_K, FragAType, FragBType, FragAccType>(
                s_a, s_b, a_frag, b_frag, acc_frag, warp_row, warp_col, buffer_id, extra_col);

            buffer_id = 1 - buffer_id;
            __syncthreads();
        }

        processFromSmemByTensorCore<BM, BN, BK, WM, WN, WMITERS, WNITERS, WKITERS, WMMA_M, WMMA_N, WMMA_K, FragAType, FragBType, FragAccType>(
            s_a, s_b, a_frag, b_frag, acc_frag, warp_row, warp_col, buffer_id, extra_col);
        __syncthreads();

        writeFromRegToGmemByTensorCore<WMITERS, WNITERS, WMMA_M, WMMA_N, FragCType, FragAccType>(c_frag, acc_frag, C, M, N, offset_c, alpha, beta);
    }

    void launchSgemmSmemKernel_v7(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream)
    {
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;
        const int WM = 64;
        const int WN = 32;
        constexpr int WMMA_M = 16;
        constexpr int WMMA_N = 16;
        constexpr int WMMA_K = 16;

        dim3 block(BN * BM / (WM * WN) * 32);
        dim3 gird((N + BN - 1) / BN, (M + BM - 1) / BM);

        // printf("in launchSgemmSmemKernel_v6: blockdimx: %d blockdimy: %d", block.x, block.y);
        SgemmSmemKernel_v7<BM, BN, BK, WM, WN, WMMA_M, WMMA_N, WMMA_K><<<gird, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    }

} // namespace gemm
