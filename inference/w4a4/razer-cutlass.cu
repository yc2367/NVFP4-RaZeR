/***************************************************************************************************
  Two-pass CUTLASS NVFP4 GEMM implementing B(+0) remap to ±5/±8 per 16-K group:

    Pass1: B_main has +0 -> ±4 (sign from meta bit7)
    Pass2: B_comp is mask where +0 -> ±1 (if mag==5) else ±4 (if mag==8), sign from bit7
           and nonzero -> 0

    D = A * B_main
    D = A * B_comp + D

  Compile:
    nvcc -O3 -std=c++17 \
      -I$CUTLASS/include \
      -I$CUTLASS/tools/util/include \
      -gencode arch=compute_120a,code=sm_120a \
      razer-cutlass.cu -o razer_cutlass_two_pass

***************************************************************************************************/

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"

#include "cutlass/numeric_conversion.h"

// Reuse razer_58 debug utilities (window size/formatting/decoders)
#include "razer_58_debug_shared.h"

// CUTLASS utilities you were missing
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/command_line.h"

// GEMM
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

template <typename T>
__host__ __device__ __forceinline__ auto make_iterator(T* ptr) {
  return cute::recast_ptr<T>(ptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Minimal helper macros (no examples/common/helper.h needed)
///////////////////////////////////////////////////////////////////////////////////////////////////

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
  GpuTimer() { CUDA_CHECK(cudaEventCreate(&start_)); CUDA_CHECK(cudaEventCreate(&stop_)); }
  ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  void stop()  { CUDA_CHECK(cudaEventRecord(stop_)); CUDA_CHECK(cudaEventSynchronize(stop_)); }
  float elapsed_millis() {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// CLI
///////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help = false;
  int m = 8192, n = 8192, k = 8192;
  int iters = 100;
  bool breakdown = false;
  bool correctness = true;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    help = cmd.check_cmd_line_flag("help");
    breakdown = cmd.check_cmd_line_flag("breakdown");
    correctness = !cmd.check_cmd_line_flag("no-correctness");
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("iters", iters);
  }

  void usage() const {
    std::cout
      << "cutlass_nvfp4_zero_remap_two_pass\n"
      << "  --m=<int> --n=<int> --k=<int> [--iters=<int>] [--breakdown] [--no-correctness]\n";
  }
};

// Host helpers matching the device nibble encoding used in the prologue
static inline uint8_t fp4_enc_1_host(bool neg) { return neg ? 0xAu : 0x2u; }
static inline uint8_t fp4_enc_4_host(bool neg) { return neg ? 0xEu : 0x6u; }

///////////////////////////////////////////////////////////////////////////////////////////////////
// CUTLASS GEMM configuration (Blackwell SM120, blockscaled NVFP4)
///////////////////////////////////////////////////////////////////////////////////////////////////

using ArchTag  = cutlass::arch::Sm120;
using OpClass  = cutlass::arch::OpClassBlockScaledTensorOp;

// A/B are NVFP4 (e2m1)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA  = cutlass::layout::RowMajor;
using LayoutB  = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;

// Accumulate in float, output float (so we can do two passes)
using ElementAccumulator = float;
using ElementCompute     = float;

using ElementC = float;
using ElementD = float;

using LayoutC  = cutlass::layout::RowMajor;
using LayoutD  = cutlass::layout::RowMajor;

constexpr int AlignmentC = 1;
constexpr int AlignmentD = 1;

// Performance shape (reasonable default; tune later)
using ThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape     = cute::Shape<cute::_1, cute::_1, cute::_1>;

// Epilogue: D = alpha * acc + beta * C
using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    FusionOperation
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,   // MNKL
    CollectiveMainloop,
    CollectiveEpilogue,
    void
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Strides / scale layouts
using StrideA  = typename GemmKernel::StrideA;
using StrideB  = typename GemmKernel::StrideB;
using StrideC  = typename GemmKernel::StrideC;
using StrideD  = typename GemmKernel::StrideD;

using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Prologue kernel: build B_main / B_comp from raw B and dense SFB metadata
///////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int K_BLOCK = 16;

// FP4 e2m1 nibble encodings (same mapping you used):
// +1 => 0x2, -1 => 0xA, +4 => 0x6, -4 => 0xE, 0 => 0x0
__device__ __forceinline__ uint8_t fp4_enc_1(bool neg) { return neg ? 0xAu : 0x2u; }
__device__ __forceinline__ uint8_t fp4_enc_4(bool neg) { return neg ? 0xEu : 0x6u; }

__global__ void build_B_main_and_comp(
  uint8_t* B_main_fp4,              // [k*n]
  uint8_t* B_comp_fp4,              // [k*n]
  const uint8_t* B_raw_fp4,         // [k*n], low nibble used, row-major KxN
  const uint8_t* SFB_dense,         // [n*(k/16)] bytes, has your meta bits
  int n, int k) {

  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int total = k * n;
  if (idx >= total) return;

  // B is dense KxN row-major (idx = kk*n + nn)
  int kk = idx / n;
  int nn = idx - kk * n;

  uint8_t b = B_raw_fp4[idx] & 0xFu;

  int kb = kk / K_BLOCK;
  uint8_t meta = SFB_dense[nn * (k / K_BLOCK) + kb];

  bool neg  = (meta & 0x80u) != 0u;
  bool mag8 = (meta & 0x40u) != 0u;

  if (b == 0x0u) {
    // main inject ±4 always
    B_main_fp4[idx] = fp4_enc_4(neg);
    // comp inject remainder: ±1 for 5, ±4 for 8
    B_comp_fp4[idx] = mag8 ? fp4_enc_4(neg) : fp4_enc_1(neg);
  } else {
    B_main_fp4[idx] = b;
    B_comp_fp4[idx] = 0x0u;
  }
}

// Device-side packing prologue: write remapped B directly into the packed layout that
// CUTLASS consumes (avoids copying B_main/B_comp back to host and repacking).
__global__ void build_B_main_and_comp_packed(
  typename ElementB::DataType* B_main_packed,   // packed per layout_B (N x K)
  typename ElementB::DataType* B_comp_packed,   // packed per layout_B (N x K)
  const typename ElementB::DataType* B_in_packed, // packed per layout_B (N x K)
  const uint8_t* SFB_dense,                     // [n*(k/16)] bytes, has your meta bits
  int n, int k) {

  // Optimized mapping:
  // - Each thread processes one full 16-K group (K_BLOCK=16) for one column nn.
  // - That is exactly 16 FP4 elements = 8 packed bytes.
  // - Single metadata load per (nn,kb) and vectorized 64-bit load/store.
  int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int num_k_blocks = k / K_BLOCK;
  int total_blocks = n * num_k_blocks;
  if (idx >= total_blocks) return;

  int nn = idx / num_k_blocks;
  int kb = idx - nn * num_k_blocks;

  uint8_t meta = SFB_dense[nn * num_k_blocks + kb];
  bool neg  = (meta & 0x80u) != 0u;
  bool mag8 = (meta & 0x40u) != 0u;

  const uint8_t* B_in_bytes = reinterpret_cast<const uint8_t*>(B_in_packed);
  uint8_t* B_main_bytes = reinterpret_cast<uint8_t*>(B_main_packed);
  uint8_t* B_comp_bytes = reinterpret_cast<uint8_t*>(B_comp_packed);

  // Byte addressing in packed layout_B (N x K): linear element index is (nn*k + kk).
  // Packed bytes are contiguous in kk, with 2 FP4 per byte.
  // Per-column byte stride is (k/2), which is a multiple of 8 because k is a multiple of 16.
  int bytes_per_col = k >> 1;
  int start_byte = nn * bytes_per_col + kb * (K_BLOCK >> 1);  // kb*8

  // Aligned vector load/store of the 8 bytes in this k-block.
  const uint64_t in64 = *reinterpret_cast<const uint64_t const*>(B_in_bytes + start_byte);

  uint64_t main64 = 0;
  uint64_t comp64 = 0;

  // Precompute injected nibbles for the +0 sentinel.
  uint8_t inj_main = fp4_enc_4(neg);
  uint8_t inj_comp = mag8 ? fp4_enc_4(neg) : fp4_enc_1(neg);

  #pragma unroll
  for (int bi = 0; bi < 8; ++bi) {
    uint8_t in_byte = uint8_t((in64 >> (8 * bi)) & 0xFFu);
    uint8_t b0 = in_byte & 0xFu;
    uint8_t b1 = (in_byte >> 4) & 0xFu;

    uint8_t o0_main = (b0 == 0x0u) ? inj_main : b0;
    uint8_t o1_main = (b1 == 0x0u) ? inj_main : b1;

    uint8_t o0_comp = (b0 == 0x0u) ? inj_comp : 0x0u;
    uint8_t o1_comp = (b1 == 0x0u) ? inj_comp : 0x0u;

    uint8_t out_main = uint8_t((o1_main << 4) | (o0_main & 0xFu));
    uint8_t out_comp = uint8_t((o1_comp << 4) | (o0_comp & 0xFu));

    main64 |= (uint64_t(out_main) << (8 * bi));
    comp64 |= (uint64_t(out_comp) << (8 * bi));
  }

  *reinterpret_cast<uint64_t*>(B_main_bytes + start_byte) = main64;
  *reinterpret_cast<uint64_t*>(B_comp_bytes + start_byte) = comp64;
}

// Benchmark utility (copied from razer_58.cu)
__global__ void flush_cache_kernel(unsigned char* buf, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * (size_t)gridDim.x;
  for (size_t i = idx; i < n; i += stride) {
    buf[i] = (unsigned char)(i);
  }
}

static int run_case(int m, int n, int k, int iters, bool do_correctness, bool do_bench, bool breakdown) {
  if (k % 64 != 0 || k % K_BLOCK != 0) {
    std::cerr << "Require k multiple of 64 and 16.\n";
    return 1;
  }
  if (m % 16 != 0 || n % 8 != 0) {
    std::cerr << "Require m multiple of 16 and n multiple of 8.\n";
    return 1;
  }

  int num_k_blocks = k / K_BLOCK;

  // Dense host buffers for raw fp4 and dense scales
  std::vector<uint8_t> h_A(size_t(m) * size_t(k));
  std::vector<uint8_t> h_B(size_t(k) * size_t(n));
  // Match razer_58 semantics, but with B treated as weights:
  //  - B scale bytes: unsigned e3m3 payload in bits[5:0] + meta bits [7]=sign, [6]=mag selector (5 vs 8)
  //  - A scale bytes: unsigned ue4m3 payload in bits[6:0]
  // CUTLASS expects scale factors in the ScaleFactorType encoding (ue4m3-like byte). We therefore
  // keep both representations for B: raw (for meta) and mma (for CUTLASS).
  std::vector<uint8_t> h_SFA_mma(size_t(m) * size_t(num_k_blocks));
  std::vector<uint8_t> h_SFB_raw(size_t(n) * size_t(num_k_blocks));
  std::vector<uint8_t> h_SFB_mma(size_t(n) * size_t(num_k_blocks));

  for (size_t i = 0; i < h_A.size(); ++i) h_A[i] = uint8_t(std::rand() & 0xF);
  for (size_t i = 0; i < h_B.size(); ++i) h_B[i] = uint8_t(std::rand() & 0xF);

  auto rand_e3m3_u6_safe = []() -> uint8_t {
    int e = 1 + (std::rand() % 6);   // 1..6
    int m3 = std::rand() & 7;
    return uint8_t(((e << 3) | m3) & 0x3F);
  };

  auto rand_ue4m3_safe = []() -> uint8_t {
    int e = 1 + (std::rand() % 0xE);  // 1..14
    int m3 = std::rand() & 7;
    return uint8_t(((e << 3) | m3) & 0x7F);
  };

  auto e3m3_to_ue4m3_for_mma = [](uint8_t raw) -> uint8_t {
    // razer_58 semantics: numeric scale payload is unsigned; meta bits are separate.
    // Convert e3m3(bias=3) payload in bits[5:0] -> ue4m3(bias=7) payload (7-bit) via +0x20.
    uint8_t payload = raw & 0x3Fu;
    return uint8_t((payload + 0x20u) & 0x7Fu);
  };

  for (int i = 0; i < m * num_k_blocks; ++i) {
    h_SFA_mma[i] = rand_ue4m3_safe();
  }
  for (int i = 0; i < n * num_k_blocks; ++i) {
    uint8_t payload = rand_e3m3_u6_safe();
    uint8_t sign = (std::rand() & 1) ? 0x80u : 0x00u;
    uint8_t mag  = (std::rand() & 1) ? 0x40u : 0x00u;
    uint8_t raw = uint8_t(payload | sign | mag);
    h_SFB_raw[i] = raw;
    h_SFB_mma[i] = e3m3_to_ue4m3_for_mma(raw);
  }

  // Device allocations (dense SFB for prologue)
  cutlass::device_memory::allocation<uint8_t> d_SFB_raw(size_t(n) * size_t(num_k_blocks));
  CUDA_CHECK(cudaMemcpy(d_SFB_raw.get(), h_SFB_raw.data(), h_SFB_raw.size(), cudaMemcpyHostToDevice));

  // CUTLASS HostTensors for A/B_main/B_comp + interleaved SFA/SFB + float C/D
  using SFAType = ElementA::ScaleFactorType;
  using SFBType = ElementB::ScaleFactorType;

  cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B_in;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B_main;
  cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B_comp;

  cutlass::HostTensor<SFAType, cutlass::layout::PackedVectorLayout> block_SFA;
  cutlass::HostTensor<SFBType, cutlass::layout::PackedVectorLayout> block_SFB;

  cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
  cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

  // Strides / layouts
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

  // Allocate host/device storage
  block_A.reset(      cutlass::make_Coord(int(cute::size(layout_A))));
  block_B_in.reset(   cutlass::make_Coord(int(cute::size(layout_B))));
  block_B_main.reset( cutlass::make_Coord(int(cute::size(layout_B))));
  block_B_comp.reset( cutlass::make_Coord(int(cute::size(layout_B))));

  block_SFA.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFA)))));
  block_SFB.reset(cutlass::make_Coord(int(cute::size(cute::filter_zeros(layout_SFB)))));

  block_C.reset(cutlass::make_Coord(int(cute::size(layout_C))));
  block_D.reset(cutlass::make_Coord(int(cute::size(layout_D))));

  // IMPORTANT: HostTensor uses PackedVectorLayout for subbyte (4-bit) storage.
  // We still pack A on host for simplicity, but B_main/B_comp are produced on device by the remap kernel.
  {
    using namespace cute;

    auto tA = make_tensor(make_iterator(block_A.host_data()), layout_A);
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementA::DataType v;
        v.raw() = typename ElementA::DataType::Base::Storage(h_A[size_t(mm) * size_t(k) + size_t(kk)] & 0xFu);
        tA(mm, kk, 0) = v;
      }
    }
  }

  // Pack B into the CUTLASS packed layout (layout_B: N x K, ColumnMajor) on host.
  // h_B is stored as KxN row-major (idx = kk*n + nn).
  {
    using namespace cute;

    auto tB = make_tensor(make_iterator(block_B_in.host_data()), layout_B);
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; ++kk) {
        typename ElementB::DataType v;
        v.raw() = typename ElementB::DataType::Base::Storage(h_B[size_t(kk) * size_t(n) + size_t(nn)] & 0xFu);
        tB(nn, kk, 0) = v;
      }
    }
  }

  // Pack dense SFA/SFB into CUTLASS interleaved layouts via cute::Tensor indexing
  {
    using namespace cute;

    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);

    // SFA: depends on (m, kblock)
    for (int mm = 0; mm < m; ++mm) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFA_mma[size_t(mm) * size_t(num_k_blocks) + size_t(kblock)];
        SFAType v;
        std::memcpy(&v, &raw, 1);
        tSFA(mm, kk, 0) = v;
      }
    }

    // SFB: depends on (n, kblock)
    for (int nn = 0; nn < n; ++nn) {
      for (int kk = 0; kk < k; kk += K_BLOCK) {
        int kblock = kk / K_BLOCK;
        uint8_t raw = h_SFB_mma[size_t(nn) * size_t(num_k_blocks) + size_t(kblock)];
        SFBType v;
        std::memcpy(&v, &raw, 1);
        tSFB(nn, kk, 0) = v;
      }
    }
  }

  // C = 0
  std::fill(block_C.host_data(), block_C.host_data() + cute::size(layout_C), 0.0f);

  // Sync all CUTLASS tensors to device
  block_A.sync_device();
  block_B_in.sync_device();
  // Allocate device storage for B_main/B_comp. Values will be written by the remap kernel.
  block_B_main.sync_device();
  block_B_comp.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();
  block_C.sync_device();
  block_D.sync_device();

  auto launch_remap_packed = [&]() {
    int threads = 128;
    int total_blocks = n * (k / K_BLOCK);
    int blocks = (total_blocks + threads - 1) / threads;
    build_B_main_and_comp_packed<<<blocks, threads>>>(
        block_B_main.device_data(),
        block_B_comp.device_data(),
      block_B_in.device_data(),
        d_SFB_raw.get(),
        n, k);
    CUDA_CHECK(cudaGetLastError());
  };

  // Two GEMM objects so we can initialize once and only call run() in the hot loop
  Gemm gemm_main;
  Gemm gemm_comp;

  typename Gemm::Arguments args1{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {
      block_A.device_data(),      stride_A,
      block_B_main.device_data(), stride_B,
      block_SFA.device_data(),    layout_SFA,
      block_SFB.device_data(),    layout_SFB
    },
    {
      {1.0f, 0.0f},
      block_C.device_data(), stride_C,
      block_D.device_data(), stride_D
    }
  };

  typename Gemm::Arguments args2{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {m, n, k, 1},
    {
      block_A.device_data(),      stride_A,
      block_B_comp.device_data(), stride_B,
      block_SFA.device_data(),    layout_SFA,
      block_SFB.device_data(),    layout_SFB
    },
    {
      {1.0f, 1.0f},
      block_D.device_data(), stride_D,
      block_D.device_data(), stride_D
    }
  };

  size_t ws1 = Gemm::get_workspace_size(args1);
  size_t ws2 = Gemm::get_workspace_size(args2);
  cutlass::device_memory::allocation<uint8_t> workspace(std::max(ws1, ws2));

  CUTLASS_CHECK(gemm_main.can_implement(args1));
  CUTLASS_CHECK(gemm_comp.can_implement(args2));
  CUTLASS_CHECK(gemm_main.initialize(args1, workspace.get()));
  CUTLASS_CHECK(gemm_comp.initialize(args2, workspace.get()));

  if (do_correctness) {
    printf("\n=== Correctness pass %dx%dx%d (CPU ref + 1 GPU run) ===\n", m, n, k);

    // Produce packed B_main/B_comp on device and then run GEMM.
    launch_remap_packed();

    CUTLASS_CHECK(gemm_main.run());
    CUTLASS_CHECK(gemm_comp.run());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Optional remap check for correctness: validate device remap against host expectation.
    block_B_main.sync_host();
    block_B_comp.sync_host();
    {
      using namespace cute;
      auto tB0 = make_tensor(make_iterator(block_B_main.host_data()), layout_B);
      auto tB1 = make_tensor(make_iterator(block_B_comp.host_data()), layout_B);

      for (int kk = 0; kk < k; ++kk) {
        for (int nn = 0; nn < n; ++nn) {
          size_t idx = size_t(kk) * size_t(n) + size_t(nn);
          uint8_t b = h_B[idx] & 0xFu;

          int kb = kk / K_BLOCK;
          uint8_t meta = h_SFB_raw[size_t(nn) * size_t(num_k_blocks) + size_t(kb)];
          bool neg = (meta & 0x80u) != 0u;
          bool mag8 = (meta & 0x40u) != 0u;

          uint8_t exp_main = (b == 0x0u) ? fp4_enc_4_host(neg) : b;
          uint8_t exp_comp = (b == 0x0u) ? (mag8 ? fp4_enc_4_host(neg) : fp4_enc_1_host(neg)) : 0x0u;

          typename ElementB::DataType b0_dt = tB0(nn, kk, 0);
          typename ElementB::DataType b1_dt = tB1(nn, kk, 0);
          uint8_t got_main = uint8_t(b0_dt.raw()) & 0xFu;
          uint8_t got_comp = uint8_t(b1_dt.raw()) & 0xFu;

          if (got_main != exp_main || got_comp != exp_comp) {
            std::cerr << "REMAPPING CHECK FAILED at (kk=" << kk << ", nn=" << nn << ")\n"
                      << "  b_raw=" << int(b)
                      << " meta=0x" << std::hex << int(meta) << std::dec
                      << " (neg=" << neg << ", mag8=" << mag8 << ")\n"
                      << "  expected main=" << int(exp_main) << " comp=" << int(exp_comp) << "\n"
                      << "  got      main=" << int(got_main) << " comp=" << int(got_comp) << "\n";
            return 2;
          }
        }
      }
      std::cout << "Remap check: PASS\n";
    }

    block_D.sync_host();

    using namespace cute;
    auto tA = make_tensor(make_iterator(block_A.host_data()), layout_A);
    auto tB0 = make_tensor(make_iterator(block_B_main.host_data()), layout_B);
    auto tB1 = make_tensor(make_iterator(block_B_comp.host_data()), layout_B);
    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);
    auto tD = make_tensor(block_D.host_data(), layout_D);

    std::vector<float> D_gpu(size_t(m) * size_t(n));
    std::vector<float> D_ref(size_t(m) * size_t(n));

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        D_gpu[size_t(i) * size_t(n) + size_t(j)] = tD(i, j, 0);
      }
    }

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          uint8_t sfa_u8 = 0;
          uint8_t sfb_u8 = 0;
          std::memcpy(&sfa_u8, &tSFA(i, kk, 0), 1);
          std::memcpy(&sfb_u8, &tSFB(j, kk, 0), 1);
          float sa = decode_ue4m3(sfa_u8);
          float sb = decode_ue4m3(sfb_u8);

          typename ElementA::DataType a_dt = tA(i, kk, 0);
          typename ElementB::DataType b0_dt = tB0(j, kk, 0);
          typename ElementB::DataType b1_dt = tB1(j, kk, 0);
          uint8_t a_nib = uint8_t(a_dt.raw()) & 0xFu;
          uint8_t b0_nib = uint8_t(b0_dt.raw()) & 0xFu;
          uint8_t b1_nib = uint8_t(b1_dt.raw()) & 0xFu;
          float a = decode_fp4_e2m1(a_nib) * sa;
          float b0 = decode_fp4_e2m1(b0_nib) * sb;
          float b1 = decode_fp4_e2m1(b1_nib) * sb;
          acc += a * (b0 + b1);
        }
        D_ref[size_t(i) * size_t(n) + size_t(j)] = acc;
      }
    }

    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    for (int i = 0; i < m * n; ++i) {
      float diff = fabsf(D_ref[i] - D_gpu[i]);
      if (diff > max_abs_err) max_abs_err = diff;
      float denom = fmaxf(1.0f, fabsf(D_ref[i]));
      float rel = diff / denom;
      if (rel > max_rel_err) max_rel_err = rel;
    }
    printf("Correctness: Max |D_ref - D_mma| = %e, Max rel err = %e\n", max_abs_err, max_rel_err);

    int start_r = (m > VIEW_R) ? (rand() % (m - VIEW_R + 1)) : 0;
    int start_c = (n > VIEW_C) ? (rand() % (n - VIEW_C + 1)) : 0;
    print_matrix_window("D_ref (correctness)", D_ref.data(), m, n, start_r, start_c);
    print_matrix_window("D_mma (correctness)", D_gpu.data(), m, n, start_r, start_c);

    float diff_block[VIEW_R * VIEW_C];
    for (int r = 0; r < VIEW_R && r < m; ++r) {
      for (int c = 0; c < VIEW_C && c < n; ++c) {
        int rr = start_r + r;
        int cc = start_c + c;
        diff_block[r * VIEW_C + c] = fabsf(D_ref[rr * n + cc] - D_gpu[rr * n + cc]);
      }
    }
    printf("Abs diff (same correctness window at [%d,%d]) =\n", start_r, start_c);
    for (int r = 0; r < VIEW_R && r < m; ++r) {
      printf("  ");
      for (int c = 0; c < VIEW_C && c < n; ++c) {
        printf("%12.3f ", (double)diff_block[r * VIEW_C + c]);
      }
      printf("\n");
    }
    printf("\n");
  }

  if (do_bench) {
#ifndef RAZER_BENCH_WARMUP_ITERS
#define RAZER_BENCH_WARMUP_ITERS 10
#endif
    const int WARMUP_ITERS  = RAZER_BENCH_WARMUP_ITERS;
    const int MEASURE_ITERS = iters;

    const size_t FLUSH_BYTES = 512ull * 1024ull * 1024ull;
    unsigned char* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, FLUSH_BYTES));
    dim3 flush_block(256);
    dim3 flush_grid(1024);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());

      // Measureable pipeline includes remap + 2-pass GEMM.
      launch_remap_packed();
      CUTLASS_CHECK(gemm_main.run());
      CUTLASS_CHECK(gemm_comp.run());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    double sum_ms = 0.0;
    double best_ms = 1e30;

    for (int i = 0; i < MEASURE_ITERS; ++i) {
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaEventRecord(start));

      launch_remap_packed();
      CUTLASS_CHECK(gemm_main.run());
      CUTLASS_CHECK(gemm_comp.run());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));

      float ms = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      sum_ms += ms;
      if (ms < best_ms) best_ms = ms;
    }

    double avg_ms = sum_ms / double(MEASURE_ITERS);
    // Report *effective* GEMM throughput: treat the two-pass method as one logical GEMM.
    // (Timing still includes: remap + 2x GEMM.)
    double effective_ops = 2.0 * double(m) * double(n) * double(k);
    double tflops_best = effective_ops / (best_ms * 1e9);
    double tflops_avg  = effective_ops / (avg_ms  * 1e9);

    printf("M,N,K = %d,%d,%d\n", m, n, k);
    printf("Best: %.3f ms, Avg: %.3f ms over %d iters (warmup %d)\n", best_ms, avg_ms, MEASURE_ITERS, WARMUP_ITERS);
    printf("Timing includes: remap + 2x GEMM\n");
    printf("Effective throughput: best %.2f TFLOPs, avg %.2f TFLOPs (2*M*N*K ops)\n\n", tflops_best, tflops_avg);

    if (breakdown) {
      cudaEvent_t e0, e1, e2, e3;
      CUDA_CHECK(cudaEventCreate(&e0));
      CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventCreate(&e2));
      CUDA_CHECK(cudaEventCreate(&e3));

      // One extra iteration to estimate per-stage GPU time.
      flush_cache_kernel<<<flush_grid, flush_block>>>(d_flush, FLUSH_BYTES);
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaEventRecord(e0));
      launch_remap_packed();
      CUDA_CHECK(cudaEventRecord(e1));
      CUTLASS_CHECK(gemm_main.run());
      CUDA_CHECK(cudaEventRecord(e2));
      CUTLASS_CHECK(gemm_comp.run());
      CUDA_CHECK(cudaEventRecord(e3));
      CUDA_CHECK(cudaEventSynchronize(e3));

      float ms_remap = 0.f, ms_gemm1 = 0.f, ms_gemm2 = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&ms_remap, e0, e1));
      CUDA_CHECK(cudaEventElapsedTime(&ms_gemm1, e1, e2));
      CUDA_CHECK(cudaEventElapsedTime(&ms_gemm2, e2, e3));
      printf("Stage breakdown (GPU time, 1 iter): remap %.3f ms, gemm_main %.3f ms, gemm_comp %.3f ms, sum %.3f ms\n\n",
             ms_remap, ms_gemm1, ms_gemm2, (double)ms_remap + (double)ms_gemm1 + (double)ms_gemm2);

      CUDA_CHECK(cudaEventDestroy(e0));
      CUDA_CHECK(cudaEventDestroy(e1));
      CUDA_CHECK(cudaEventDestroy(e2));
      CUDA_CHECK(cudaEventDestroy(e3));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_flush));
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv) {
  Options opt;
  opt.parse(argc, argv);
  if (opt.help) { opt.usage(); return 0; }

  // Match razer_58 RNG behavior once for the entire program.
  std::srand((unsigned)std::time(nullptr));

  // Optional quick correctness pass first (small fixed shape), then benchmark the requested shape.
  if (opt.correctness) {
    int rc = run_case(512, 512, 512, 1, /*do_correctness=*/true, /*do_bench=*/false, /*breakdown=*/false);
    if (rc != 0) return rc;
  }

  return run_case(opt.m, opt.n, opt.k, opt.iters, /*do_correctness=*/false, /*do_bench=*/true, /*breakdown=*/opt.breakdown);
}
