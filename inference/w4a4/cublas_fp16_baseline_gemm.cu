/***************************************************************************************************
  Baseline (single-pass) FP16 GEMM on SM120(a) using cuBLAS:

    D = A * B

  NOTE: CUTLASS' SM120 non-blockscaled collective builders currently focus on F8/F6/F4 paths.
  For a practical FP16 baseline on Blackwell today, this file uses cuBLAS.

  Compile (SM120a):

    nvcc -O3 -std=c++17 \
      -gencode arch=compute_120a,code=sm_120a \
      cutlass_fp16_baseline_gemm.cu -o cublas_fp16_baseline_sm120a -lcublas

  Example:

    ./cublas_fp16_baseline_sm120a --m=8192 --n=8192 --k=8192 --warmup=10 --iters=100 --flush-mb=512

***************************************************************************************************/

#include <cuda_runtime.h>

#include <cublas_v2.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define CUDA_CHECK(expr)                                                                 \
  do {                                                                                   \
    cudaError_t _err = (expr);                                                           \
    if (_err != cudaSuccess) {                                                           \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

#define CUBLAS_CHECK(expr)                                                               \
  do {                                                                                   \
    cublasStatus_t _st = (expr);                                                         \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                                  \
      std::cerr << "cuBLAS error: " << int(_st)                                          \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

struct Options {
  bool help = false;
  int m = 8192;
  int n = 8192;
  int k = 8192;
  int warmup = 10;
  int iters = 100;
  int flush_mb = 512;
  bool accum_fp16 = false;

  void parse(int argc, char const** argv) {
    for (int i = 1; i < argc; ++i) {
      const char* a = argv[i];
      if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
        help = true;
        continue;
      }

      if (!std::strcmp(a, "--accum-fp16") || !std::strcmp(a, "--compute-fp16")) {
        accum_fp16 = true;
        continue;
      }

      auto parse_i = [&](const char* key, int& out) {
        size_t len = std::strlen(key);
        if (!std::strncmp(a, key, len) && a[len] == '=') {
          out = std::atoi(a + len + 1);
          return true;
        }
        return false;
      };

      (void)(parse_i("--m", m) ||
             parse_i("--n", n) ||
             parse_i("--k", k) ||
             parse_i("--warmup", warmup) ||
             parse_i("--iters", iters) ||
             parse_i("--flush-mb", flush_mb));
    }
  }

  void usage() const {
    std::cout
        << "cublas_fp16_baseline_gemm\n"
        << "  --m=<int> --n=<int> --k=<int> [--warmup=<int>] [--iters=<int>] [--flush-mb=<int>] [--accum-fp16]\n";
  }
};

__global__ void flush_cache_kernel(unsigned char* buf, size_t bytes) {
  size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  size_t stride = size_t(blockDim.x) * size_t(gridDim.x);
  for (size_t i = tid; i < bytes; i += stride) {
    buf[i] = static_cast<unsigned char>(buf[i] + 1);
  }
}

static inline __half h2half(float x) {
  return __float2half_rn(x);
}

static void run_one(int m, int n, int k, Options const& opt) {
//   if (m % 16 != 0 || n % 8 != 0 || k % 8 != 0) {
//     std::cerr << "Require m multiple of 16, n multiple of 8, k multiple of 8.\n";
//     std::exit(1);
//   }

  std::vector<float> hA_f(size_t(m) * size_t(k));
  std::vector<float> hB_f(size_t(k) * size_t(n));

  std::srand(0xC0FFEE);
  auto rand_f = []() -> float {
    int r = (std::rand() % 2001) - 1000;
    return float(r) / 1000.0f;
  };
  for (size_t i = 0; i < hA_f.size(); ++i) hA_f[i] = rand_f();
  for (size_t i = 0; i < hB_f.size(); ++i) hB_f[i] = rand_f();

  // Column-major storage for cuBLAS.
  // A: (m x k), B: (k x n), D: (m x n)
  int lda = m;
  int ldb = k;
  int ldc = m;

  std::vector<__half> hA(size_t(m) * size_t(k));
  std::vector<__half> hB(size_t(k) * size_t(n));
  std::vector<__half> hD(size_t(m) * size_t(n));

  // Fill A/B in column-major.
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < m; ++row) {
      hA[size_t(col) * size_t(lda) + size_t(row)] = h2half(hA_f[size_t(row) * size_t(k) + size_t(col)]);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < k; ++row) {
      hB[size_t(col) * size_t(ldb) + size_t(row)] = h2half(hB_f[size_t(row) * size_t(n) + size_t(col)]);
    }
  }
  std::fill(hD.begin(), hD.end(), h2half(0.0f));

  __half *dA = nullptr, *dB = nullptr, *dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, sizeof(__half) * size_t(m) * size_t(k)));
  CUDA_CHECK(cudaMalloc(&dB, sizeof(__half) * size_t(k) * size_t(n)));
  CUDA_CHECK(cudaMalloc(&dD, sizeof(__half) * size_t(m) * size_t(n)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(__half) * size_t(m) * size_t(k), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(__half) * size_t(k) * size_t(n), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dD, hD.data(), sizeof(__half) * size_t(m) * size_t(n), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  __half alpha_h = h2half(1.0f);
  __half beta_h = h2half(0.0f);
  const void* alpha = opt.accum_fp16 ? (const void*)&alpha_h : (const void*)&alpha_f;
  const void* beta  = opt.accum_fp16 ? (const void*)&beta_h  : (const void*)&beta_f;
  cublasComputeType_t compute = opt.accum_fp16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;

  const size_t FLUSH_BYTES = (opt.flush_mb > 0) ? (size_t(opt.flush_mb) * 1024ull * 1024ull) : 0ull;
  unsigned char* d_flush = nullptr;
  dim3 flush_block(256);
  dim3 flush_grid(1024);
  if (FLUSH_BYTES > 0) CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_BYTES));

  for (int i = 0; i < opt.warmup; ++i) {
    if (d_flush) {
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
      alpha,
        dA,
        CUDA_R_16F,
        lda,
        dB,
        CUDA_R_16F,
        ldb,
      beta,
        dD,
        CUDA_R_16F,
        ldc,
      compute,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  double sum_ms = 0.0;
  double best_ms = 1e30;
  for (int it = 0; it < opt.iters; ++it) {
    if (d_flush) {
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      m,
      n,
      k,
      alpha,
      dA,
      CUDA_R_16F,
      lda,
      dB,
      CUDA_R_16F,
      ldb,
      beta,
      dD,
      CUDA_R_16F,
      ldc,
      compute,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    sum_ms += double(ms);
    if (double(ms) < best_ms) best_ms = double(ms);
  }

  double avg_ms = sum_ms / double(opt.iters);
  double ops = 2.0 * double(m) * double(n) * double(k);
  double tflops_best = ops / (best_ms * 1e-3) / 1e12;
  double tflops_avg  = ops / (avg_ms  * 1e-3) / 1e12;

  std::cout << "M,N,K = " << m << "," << n << "," << k << "\n";
  std::cout << "Best time (single GEMM) = " << best_ms << " ms\n";
  std::cout << "Avg  time (single GEMM) = " << avg_ms << " ms\n";
  std::cout << "TFLOPs (2*M*N*K): best " << tflops_best << ", avg " << tflops_avg << "\n";

  CUDA_CHECK(cudaMemcpy(hD.data(), dD, sizeof(__half) * size_t(m) * size_t(n), cudaMemcpyDeviceToHost));
  std::cout << "D[0] = " << __half2float(hD[0]) << "\n";

  if (d_flush) CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUBLAS_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dD));
}

int main(int argc, char const** argv) {
  Options opt;
  opt.parse(argc, argv);
  if (opt.help) {
    opt.usage();
    return 0;
  }
  run_one(opt.m, opt.n, opt.k, opt);
  return 0;
}
