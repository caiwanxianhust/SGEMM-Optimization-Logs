namespace gemm
{

    // naive kernel
    void launchSgemmNaiveKernel(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                const float alpha, const float beta, cudaStream_t stream = 0);

    // 共享内存、分片
    void launchSgemmSmemKernel_v1(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // 每个线程计算多个元素，一维
    void launchSgemmSmemKernel_v2(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // 每个线程计算多个元素，拓展到二维
    void launchSgemmSmemKernel_v3(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // 向量化加载
    void launchSgemmSmemKernel_v4(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // warp 分片
    void launchSgemmSmemKernel_v5(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // double_buffer
    void launchSgemmSmemKernel_v6(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // wmma api
    void launchSgemmSmemKernel_v7(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                  const float alpha, const float beta, cudaStream_t stream = 0);

    // cublas api
    void launchSgemmcuBlas(const float *A, const float *B, float *C, const int M, const int N, const int K,
                           const float alpha, const float beta, cudaStream_t stream = 0);

}
