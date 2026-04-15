#pragma once
#include <torch/extension.h>

// Kernel implementation selection for razer_gemm.
// NOTE: values are part of the public Python API via pybind.
enum class RazerImpl : int {
    Auto  = 0,  // current heuristic (default)
    Gemv  = 1,  // force GEMV path (requires M==1)
    Small = 2,  // force CUDA-core GEMM
};

// Returned launch config (useful for logging / debugging / reproducibility).
struct RazerGemmLaunchConfig {
    int impl = static_cast<int>(RazerImpl::Auto);
    int split_k = 1;   // gridDim.z for GEMM paths; unused for GEMV
    int gemv_g = 0;    // GEMV groups-per-block (1/2/4/8), or 0 when not GEMV
    int small_r = 0;   // small GEMM rows-per-block (blockDim.y), or 0 when not small GEMM
};

// GEMM entry that allows overriding implementation and launch config.
//
// groupsize:
//   - 128: canonical RaZeR path with fp16 scales shaped [K/128, N]
//   - 16 : experimental decode path with packed fp8 scales shaped [K/128, N, 2]
//
// impl:
//   - Auto  : follow built-in heuristics (default)
//   - Gemv  : force GEMV (M==1)
//   - Small : force CUDA-core GEMM
//
// split_k:
//   - -1: choose via built-in heuristics
//   - >=1: force split-K for GEMM paths (gridDim.z). For GEMV, ignored.
//
// gemv_g:
//   - -1: keep current default
//   - 0 : choose via GEMV heuristic
//   - 1/2/4/8: force that template instantiation
RazerGemmLaunchConfig razer_gemm(
    at::Tensor fA,
    at::Tensor qB,
    at::Tensor scaling_factors,
    at::Tensor out,
    int groupsize,
    int impl,
    int split_k,
    int gemv_g
);

// Debug dequantize packed B → dense half matrix using same intrinsics/remap.
at::Tensor razer_dequant(
    at::Tensor qB,              // [K/8, N] int32
    at::Tensor scaling_factors, // g128: [K/128, N] fp16; g16: [K/128, N, 2] int32
    int64_t K,                  // full K (must be multiple of 128 and 8)
    int groupsize
);