/***************************************************************************************************
  Baseline (single-pass) CUTLASS MXFP8 GEMM on SM120(a):

    D = A * B

  This is meant to provide a clean MXFP8 TFLOPs number without the two-pass
  +0-remap compensation used in razer-cutlass.cu and razer_58.cu.

  Compile (SM120a):

    nvcc -O3 -std=c++17 --expt-relaxed-constexpr \
      -I$CUTLASS/include \
      -I$CUTLASS/tools/util/include \
      -gencode arch=compute_120a,code=sm_120a \
      cutlass_mxfp8_baseline_gemm.cu -o mxfp8_baseline_sm120a

  Example:

    ./mxfp8_baseline_sm120a --m=8192 --n=8192 --k=8192 --warmup=10 --iters=100 --flush-mb=512

***************************************************************************************************/

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cute/tensor.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/command_line.h"

#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

template <typename T>
auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

#define CUDA_CHECK(expr)                                                                 \
  do {                                                                                   \
    cudaError_t _err = (expr);                                                           \
    if (_err != cudaSuccess) {                                                           \
      std::cerr << "CUDA error: " << cudaGetErrorString(_err)                            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

#define CUTLASS_CHECK(expr)                                                              \
  do {                                                                                   \
    cutlass::Status _st = (expr);                                                        \
    if (_st != cutlass::Status::kSuccess) {                                              \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(_st)                      \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

struct GpuTimer {
  cudaEvent_t start_, stop_;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  void stop() {
    CUDA_CHECK(cudaEventRecord(stop_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
  }
  float elapsed_millis() {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

struct Options {
  bool help = false;
  int m = 8192;
  int n = 8192;
  int k = 8192;
  int warmup = 10;
  int iters = 100;
  int flush_mb = 512;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iters", iters);
    cmd.get_cmd_line_argument("flush-mb", flush_mb);
  }

  void usage() const {
    std::cout
        << "cutlass_mxfp8_baseline_gemm\n"
        << "  --m=<int> --n=<int> --k=<int> [--warmup=<int>] [--iters=<int>] [--flush-mb=<int>]\n";
  }
};

__global__ void flush_cache_kernel(unsigned char* buf, size_t bytes) {
  size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  size_t stride = size_t(blockDim.x) * size_t(gridDim.x);
  for (size_t i = tid; i < bytes; i += stride) {
    buf[i] = static_cast<unsigned char>(buf[i] + 1);
  }
}

// CUTLASS kernel configuration (SM120, blockscaled MXFP8)
using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;

// MXFP8 uses a float8 payload with per-block UE8M0 scale factors.
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;

// Alignment is expressed in *elements*. For 8-bit elements, 16 elements = 16 bytes.
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;

using ElementAccumulator = float;
using ElementCompute = float;

using ElementC = float;
using ElementD = float;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentC = 1;
constexpr int AlignmentD = 1;

using ThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using FusionOperation = cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr int K_BLOCK = 16;

static void run_one(int m, int n, int k, Options const& opt) {
  if (k % 64 != 0 || k % K_BLOCK != 0) {
    std::cerr << "Require k multiple of 64 and 16.\n";
    std::exit(1);
  }
  if (m % 16 != 0 || n % 8 != 0) {
    std::cerr << "Require m multiple of 16 and n multiple of 8.\n";
    std::exit(1);
  }

  int num_k_blocks = k / K_BLOCK;

  // Host-side init (only affects correctness/stability; not timed)
  std::vector<float> hA_f(size_t(m) * size_t(k));
  std::vector<float> hB_f(size_t(k) * size_t(n));

  // MXFP8 scale factors: UE8M0 exponent byte. 0x7f corresponds to a scale of 1.0.
  std::vector<uint8_t> h_SFA_u8(size_t(m) * size_t(num_k_blocks), 0x7Fu);
  std::vector<uint8_t> h_SFB_u8(size_t(n) * size_t(num_k_blocks), 0x7Fu);

  std::srand(0xC0FFEE);
  auto rand_f = []() -> float {
    // Keep values in a reasonable range for fp8.
    int r = (std::rand() % 2001) - 1000;
    return float(r) / 1000.0f;
  };

  for (size_t i = 0; i < hA_f.size(); ++i) hA_f[i] = rand_f();
  for (size_t i = 0; i < hB_f.size(); ++i) hB_f[i] = rand_f();

  using AData = typename ElementA::DataType;
  using BData = typename ElementB::DataType;
  using SFAType = typename ElementA::ScaleFactorType;
  using SFBType = typename ElementB::ScaleFactorType;

  static_assert(sizeof(SFAType) == 1, "Unexpected MXFP8 ScaleFactorType size");
  static_assert(sizeof(SFBType) == 1, "Unexpected MXFP8 ScaleFactorType size");

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  auto layout_A = cute::make_layout(cute::make_shape(m, k, 1), stride_A);
  auto layout_B = cute::make_layout(cute::make_shape(n, k, 1), stride_B);
  auto layout_C = cute::make_layout(cute::make_shape(m, n, 1), stride_C);
  auto layout_D = cute::make_layout(cute::make_shape(m, n, 1), stride_D);

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  cutlass::HostTensor<AData, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<BData, cutlass::layout::PackedVectorLayout> block_B;
  cutlass::HostTensor<SFAType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<SFBType, cutlass::layout::PackedVectorLayout> block_SFB;
  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

  block_A.reset(cutlass::make_Coord(int(cute::size(layout_A))));
  block_B.reset(cutlass::make_Coord(int(cute::size(layout_B))));
  block_SFA.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFA)))));
  block_SFB.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFB)))));
  block_C.reset(cutlass::make_Coord(int(cute::size(layout_C))));
  block_D.reset(cutlass::make_Coord(int(cute::size(layout_D))));

  // Pack A/B into CUTLASS layouts.
  {
    using namespace cute;
    auto tA = make_tensor(make_iterator(block_A.host_data()), layout_A);
    auto tB = make_tensor(make_iterator(block_B.host_data()), layout_B);

    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; ++kk) {
        tA(mm, kk, 0) = AData(hA_f[size_t(mm) * size_t(k) + size_t(kk)]);
      }
    }

    // NOTE: layout_B is (n,k) with ColumnMajor stride
    // Host source is KxN row-major: idx = kk*n + nn
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        tB(nn, kk, 0) = BData(hB_f[size_t(kk) * size_t(n) + size_t(nn)]);
      }
    }
  }

  // Pack scales into interleaved layouts (one scale per (row,kblock) and (col,kblock)).
  {
    using namespace cute;
    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);

    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFA_u8[size_t(mm) * size_t(num_k_blocks) + size_t(kblock)];
        SFAType v;
        std::memcpy(&v, &raw, 1);
        tSFA(mm, kk, 0) = v;
      }
    }

    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFB_u8[size_t(nn) * size_t(num_k_blocks) + size_t(kblock)];
        SFBType v;
        std::memcpy(&v, &raw, 1);
        tSFB(nn, kk, 0) = v;
      }
    }
  }

  std::fill(block_C.host_data(), block_C.host_data() + cute::size(layout_C), 0.0f);

  block_A.sync_device();
  block_B.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_C.sync_device();
  block_D.sync_device();

  Gemm gemm;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {block_A.device_data(), stride_A,
       block_B.device_data(), stride_B,
       block_SFA.device_data(), layout_SFA,
       block_SFB.device_data(), layout_SFB},
      {{1.0f, 0.0f}, block_C.device_data(), stride_C, block_D.device_data(), stride_D}};

  CUTLASS_CHECK(gemm.can_implement(args));

  size_t ws = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<uint8_t> workspace(ws);

  const size_t FLUSH_BYTES = (opt.flush_mb > 0) ? (size_t(opt.flush_mb) * 1024ull * 1024ull) : 0ull;
  unsigned char* d_flush = nullptr;
  dim3 flush_block(256);
  dim3 flush_grid(1024);
  if (FLUSH_BYTES > 0) {
    CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_BYTES));
  }

  for (int i = 0; i < opt.warmup; ++i) {
    if (d_flush) {
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUTLASS_CHECK(gemm.initialize(args, workspace.get()));
    CUTLASS_CHECK(gemm.run());
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
    CUTLASS_CHECK(gemm.initialize(args, workspace.get()));
    CUTLASS_CHECK(gemm.run());
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

  block_D.sync_host();
  std::cout << "D[0] = " << block_D.host_data()[0] << "\n";

  if (d_flush) CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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
