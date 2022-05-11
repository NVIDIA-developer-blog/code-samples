#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>


#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};
    
    int m, n, k;
    int lda, ldb, ldc;
    double alpha, beta;

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " cuBLAS dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");

    int size = 4092;
    m = size, k = size, n = size;
    lda = size, ldb = size, ldc = size;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
        " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0, beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    // const std::vector<double> A(m * n);
    // const std::vector<double> B(m * n);
    // const std::vector<double> C(m * n);

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * m * k));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * k * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * m * n));

    // CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice,
    //                            stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice,
    //                            stream));

    /* step 3: compute */
    printf (" Computing matrix product using cuBLAS dgemm function \n\n");

    cudaEventCreate( &startEvent, cudaEventBlockingSync );
    cudaEventRecord( startEvent );

    for (int i =0; i< 10; i++)
    CUBLAS_CHECK(
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEventCreate( &stopEvent, cudaEventBlockingSync );
    cudaEventRecord( stopEvent );
    cudaEventSynchronize( stopEvent );

    printf ("\n Computations completed.\n\n");

    cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
    printf( " Time Elapsed: %0.2f ms \n\n", elapsed_gpu_ms/10);

    /* step 4: copy data to host */
    // CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(double) * C.size(), cudaMemcpyDeviceToHost,
    //                            stream));

    // CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    printf ("\n Deallocating memory \n\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    printf (" Example completed. \n\n");
    return 0;
}