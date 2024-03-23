#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib> 
#include <ctime>

#define CHECK_CUDA(call) \
    do { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CHECK_CUBLAS(status) \
    do { \
        const cublasStatus_t error = status; \
        if (error != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << __FILE__ << ":" << __LINE__ << ", " \
                      << error << std::endl; \
            exit(1); \
        } \
    } while (0)

// Simple CPU-based matrix multiplication for verification
void simpleCpuGemm(int m, int n, int k, const int8_t *A, const int8_t *B, int32_t *C) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            int32_t sum = 0;
            for (int i = 0; i < k; ++i) {
                sum += A[i + row * k] * B[i + col * k];
            }
            C[row + col * m] = sum;
        }
    }
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    int m = 100, n = 32, k = 100;
    std::srand(std::time(0));

    // Allocate and initialize host matrices
    std::vector<int8_t> h_A(m * k);
    std::vector<int8_t> h_B(k * n);
    std::vector<int8_t> h_B_transposed(n * k);
    std::vector<int32_t> h_C(m * n);
    std::vector<int32_t> h_C_cpu(m * n); // For CPU result
    std::vector<int32_t> h_C_cpu_transposed(m * n);

    for (int i = 0; i < m * k; ++i) {
        h_A[i] = static_cast<int8_t>(std::rand() % 127);
    }
    for (int i = 0; i < k * n; ++i) {
        h_B[i] = static_cast<int8_t>(std::rand() % 127);
    }

    int8_t *A, *B;
    int32_t *C; // Assuming the result matrix C is int32
    CHECK_CUDA(cudaMalloc(&A, m * k * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&B, k * n * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&C, m * n * sizeof(int32_t)));

    // Copy matrices A and B to the device
    CHECK_CUDA(cudaMemcpy(A, h_A.data(), m * k * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B, h_B.data(), k * n * sizeof(int8_t), cudaMemcpyHostToDevice));

    const int32_t alpha = 1;
    const int32_t beta = 0;
    // Perform the matrix multiplication using cublasGemmEx
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              m, n, k,
                              &alpha,
                              A, CUDA_R_8I, k,
                              B, CUDA_R_8I, k,
                              &beta,
                              C, CUDA_R_32I, m,
                              CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT));

    CHECK_CUDA(cudaMemcpy(h_C.data(), C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Perform the CPU-based matrix multiplication for verification
    simpleCpuGemm(m, n, k, h_A.data(), h_B.data(), h_C_cpu.data());

    // Compare the results (CUDA vs. CPU)
    bool resultsMatch = true;
    for (int i = 0; i < m * n; ++i) {
        if (h_C_cpu[i] != h_C[i]) {
            std::cout << h_C_cpu[i] << " " << h_C[i] << " " << i << "\n";
            resultsMatch = false;
            break;
        }
    }
    std::cout << "The results " << (resultsMatch ? "MATCH" : "DO NOT MATCH") << "!\n";

    // Clean up resources
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}


