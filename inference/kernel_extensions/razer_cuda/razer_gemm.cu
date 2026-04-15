#include <cuda.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include "razer_gemm.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#define CUDA_CHECK_IN(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define DTYPE_CHECK(x,dt) TORCH_CHECK((x).dtype()==(dt), #x " has wrong dtype")
#define CONTIGUOUS(x) do { if(!(x).is_contiguous()) (x) = (x).contiguous(); } while(0)

// Pairwise half2 atomic add helper (requires 4B aligned even index)
static __device__ __forceinline__ void atomicAddHalf2Packed(__half* address_even, __half v0, __half v1) {
  atomicAdd(reinterpret_cast<__half2*>(address_even), __halves2half2(v0, v1));
}

////////////////////////////////////////////////////////////////////////////////
// Canonical FP4 (E2M1) helpers (weights) — *intrinsics* only, no LUTs
////////////////////////////////////////////////////////////////////////////////
// Derive the special value magnitude (5 or 8) from the 2nd MSB (bit 14) of a fp16 scale.
// The sign of the fp16 scale controls the sign of the special value (and the group)
// since the special is applied before scaling.
__device__ __forceinline__ __half scale_msb2_to_fixval_from_bits(uint16_t s_bits)
{
  // fp16(5.0)=0x4500, fp16(8.0)=0x4800
  const uint16_t out = (s_bits & 0x4000u) ? 0x4800u : 0x4500u;
  return __ushort_as_half(out);
}

// Decode a positive numeric fp8 scale stored as an "e3m3-like" payload inside e4m3.
// We reserve bit7 for the special-value sign and bit6 for the special-value magnitude.
// The remaining payload is decoded as e4m3 and rescaled by 16 to correct the bias.
__device__ __forceinline__ __half decode_fp8_scale_from_bits(uint8_t bits)
{
  const uint8_t payload = (uint8_t)(bits & 0x3Fu);
  const __half_raw hraw = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)payload, __NV_E4M3);
  return __hmul(__ushort_as_half(hraw.x), __float2half(16.0f));
}

// g16 experimental path: bit7 encodes special sign and bit6 selects 5.0 vs 8.0.
__device__ __forceinline__ __half fix_from_fp8_meta_bits(uint8_t bits)
{
  uint16_t out = (bits & 0x40u) ? 0x4800u : 0x4500u;
  if (bits & 0x80u) out |= 0x8000u;
  return __ushort_as_half(out);
}


// Dequantize FP4 E2M1 pair from a packed byte and apply RaZeR special-sentinel remap + scaling
// packed: low nibble=n0, high nibble=n1
__device__ __forceinline__ __half2 dequant_fix_scale_e2m1(uint8_t packed, __half hfixB, __half hscale)
{
  // Convert packed E2M1 fp4x2 to half2
  __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)packed, __NV_E2M1);
  __half2 h2 = *reinterpret_cast<__half2*>(&raw);

  // Kernel convention (see quantize.py):
  //   - nibble 0x0 (positive zero) is the *special* sentinel
  //   - nibble 0x8 (negative zero) is used for normal zeros
  const bool z0 = ((packed & 0x0F) == 0x0);   // low nibble
  const bool z1 = ((packed & 0xF0) == 0x00);  // high nibble == 0x0

  __half lo = z0 ? hfixB : __low2half(h2);
  __half hi = z1 ? hfixB : __high2half(h2);

  // Apply uniform scale to both halves with a single vector op
  const __half2 s2 = __halves2half2(hscale, hscale);
  return __hmul2(__halves2half2(lo, hi), s2);
}

// GEMM: block covers R rows (R warps) and 32 columns (lanes)
// Fixes:
//  (1) Stage B_packed (16 words x 32 cols) into shared once per kg (row_w==0 loads)
//      so row_w=1..R-1 do not re-load the same B words.
//  (2) Keep A tiling as R*128 (must be per-row), but avoid any extra replication.
//
// Expected launch:
//   dim3 block(32, R, 1);
//   dim3 grid(ceil_div(M, R), ceil_div(N, 32), splits);
// Shared bytes:
//   (R*128 + 32 + 32) * sizeof(__half) + (16*32) * sizeof(uint32_t)
extern "C" __global__
__global__ void razer_gemm_cuda(
  const __half*   __restrict__ A_fp16,   // [M,K]
  const uint32_t* __restrict__ B_packed, // [K/8, N]
  const __half*   __restrict__ B_scale,  // [K/128, N]
  __half*         __restrict__ D,        // [M,N]
  int M, int N, int Ktotal)
{
  const int lda = Ktotal;
  const int ldb_scale = N;
  const int ldd = N;

  const int lane  = threadIdx.x & 31;
  const int row_w = threadIdx.y;
  const int R     = (int)blockDim.y;

  const int col0  = (int)blockIdx.y * 32;
  const int row0  = (int)blockIdx.x * R;

  const int gj    = col0 + lane;
  const int gi    = row0 + row_w;

  const int groups_total = Ktotal >> 7;
  const int splits = max(1, (int)gridDim.z);
  const int sid    = (int)blockIdx.z;

  int kg_begin = (groups_total * sid) / splits;
  int kg_end   = (groups_total * (sid + 1)) / splits;
  kg_begin = max(kg_begin, 0);
  kg_end   = min(kg_end, groups_total);
  if (kg_begin >= kg_end) return;

  extern __shared__ unsigned char smem_u8[];
  __half2* As2_tile = reinterpret_cast<__half2*>(smem_u8);

  __half* scale_sh = reinterpret_cast<__half*>(As2_tile + (R * 64));
  __half* fix_sh   = scale_sh + 32;

  uintptr_t p = reinterpret_cast<uintptr_t>(fix_sh + 32);
  p = (p + 3u) & ~uintptr_t(3u);
  uint32_t* Bsh = reinterpret_cast<uint32_t*>(p); // 16*32

  float accum_f = 0.0f;

  for (int kg = kg_begin; kg < kg_end; ++kg) {
    const int kbase = (kg << 7);

    // A staging: alignment-safe (two half loads -> pack half2 -> shared half2 store)
    // NOTE: We tried switching this to direct __half2 vector loads, but it benchmarked worse here.
    if (gi < M) {
      #pragma unroll
      for (int t2 = lane; t2 < 64; t2 += 32) {
        const int k2 = kbase + (t2 << 1); // 2 half per half2
        const __half a0 = __ldg(&A_fp16[gi * lda + (k2 + 0)]);
        const __half a1 = __ldg(&A_fp16[gi * lda + (k2 + 1)]);
        As2_tile[row_w * 64 + t2] = __halves2half2(a0, a1);
      }
    } else {
      #pragma unroll
      for (int t2 = lane; t2 < 64; t2 += 32) {
        As2_tile[row_w * 64 + t2] = __float2half2_rn(0.0f);
      }
    }

    // Stage scale/fix + Bsh once per column (row_w==0)
    // NOTE: We tried a per-lane scale/fix staging (one fp16 load per lane, store to shared) to appease
    // coalescing metrics, but it benchmarked worse.
    if (row_w == 0) {
      const int lane_even = lane & ~1;
      const int gj_even   = col0 + lane_even;

      uint32_t s_bits2 = 0u; // packed two fp16 scales (low: even col, high: odd col)

      if (gj_even < N) {
        // Load two fp16 scales as one 32-bit word. This is alignment-friendly when (kg*ldb_scale + gj_even) is even.
        const uint32_t* sptr32 = reinterpret_cast<const uint32_t*>(&B_scale[kg * ldb_scale + gj_even]);
        // If gj_even == N-1, the upper half would be OOB; handle that case.
        if (gj_even + 1 < N) {
          s_bits2 = __ldg(sptr32);
        } else {
          // only even scale valid; odd scale = 0
          const uint16_t s0 = __ldg(reinterpret_cast<const uint16_t*>(&B_scale[kg * ldb_scale + gj_even]));
          s_bits2 = (uint32_t)s0;
        }

        // Store *clean* scales to shared (mask off bit14 so it doesn't perturb scaling).
        // Bit14 is used as metadata (select 5 vs 8) and must not affect the numeric scale.
        const uint16_t s0_bits_raw = (uint16_t)(s_bits2 & 0xFFFFu);
        const uint16_t s1_bits_raw = (uint16_t)(s_bits2 >> 16);
        const uint16_t s0_bits = (uint16_t)(s0_bits_raw & (uint16_t)~0x4000u);
        const uint16_t s1_bits = (uint16_t)(s1_bits_raw & (uint16_t)~0x4000u);
        const uint32_t sc_bits2 = (uint32_t)s0_bits | ((uint32_t)s1_bits << 16);
        if ((lane & 1) == 0) {
          *reinterpret_cast<uint32_t*>(&scale_sh[lane_even]) = sc_bits2;
        }

        // Stage Bsh while the compiler has independent work.
        #pragma unroll
        for (int w_off = 0; w_off < 16; ++w_off) {
          const int widx = (kg << 4) + w_off;
          // Each lane stages its own column word as before:
          const uint32_t w = (gj < N) ? __ldg(&B_packed[widx * N + gj]) : 0u;
          Bsh[w_off * 32 + lane] = w;
        }

        // Now compute fix from bits (cheap, no divergence). Constants are fp16(5)=0x4500, fp16(8)=0x4800.
        const uint16_t f0_bits = (s0_bits_raw & 0x4000u) ? 0x4800u : 0x4500u;
        const uint16_t f1_bits = (s1_bits_raw & 0x4000u) ? 0x4800u : 0x4500u;

        if ((lane & 1) == 0) {
          // Store both fixes as one 32-bit word to shared
          *reinterpret_cast<uint32_t*>(&fix_sh[lane_even]) =
              (uint32_t)f0_bits | ((uint32_t)f1_bits << 16);
        }
      } else {
        // Column block entirely OOB for this pair; still stage Bsh zeros for all lanes.
        if ((lane & 1) == 0) {
          *reinterpret_cast<uint32_t*>(&scale_sh[lane_even]) = 0u;
          *reinterpret_cast<uint32_t*>(&fix_sh[lane_even])   = 0u;
        }
        #pragma unroll
        for (int w_off = 0; w_off < 16; ++w_off) {
          Bsh[w_off * 32 + lane] = 0u;
        }
      }
    }

    __syncthreads();

    if (gj >= N) {
      __syncthreads();
      continue;
    }

    const __half hscale = scale_sh[lane];
    const __half hfixB  = fix_sh[lane];

    __half2 acc2     = __float2half2_rn(0.0f);
    __half  sum_fixA = __float2half(0.0f);

    #pragma unroll
    for (int w_off = 0; w_off < 16; ++w_off) {
      const uint32_t w = Bsh[w_off * 32 + lane];
      const int a2_base = (w_off << 2);

      const uint8_t b0 = (uint8_t)(w);
      const uint8_t b1 = (uint8_t)(w >> 8);
      const uint8_t b2 = (uint8_t)(w >> 16);
      const uint8_t b3 = (uint8_t)(w >> 24);

      const __half2* Arow = &As2_tile[row_w * 64 + a2_base];

      const __half2 a20 = Arow[0];
      const __half2 a21 = Arow[1];
      const __half2 a22 = Arow[2];
      const __half2 a23 = Arow[3];

      // Avoid named bool temporaries in the hot loop
      if (!(b0 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a20));
      if (b0 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a20));

      if (!(b1 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a21));
      if (b1 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a21));

      if (!(b2 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a22));
      if (b2 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a22));

      if (!(b3 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a23));
      if (b3 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a23));

      // Try: batch all fixups first, then do all FMAs at the end.
      // Move all raw decodes to immediately before the HFMA block.
      const __half2_raw raw0 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b0, __NV_E2M1);
      const __half2_raw raw1 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b1, __NV_E2M1);
      const __half2_raw raw2 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b2, __NV_E2M1);
      const __half2_raw raw3 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b3, __NV_E2M1);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw0), a20, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw1), a21, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw2), a22, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw3), a23, acc2);
    }

    __syncthreads();

    __half acc_h = __hadd(__low2half(acc2), __high2half(acc2));
    acc_h = __hadd(acc_h, __hmul(sum_fixA, hfixB));
    accum_f += __half2float(__hmul(acc_h, hscale));
  }

  if (gi < M && gj < N) {
    const __half my_val = __float2half(accum_f);
    if (splits == 1) {
      D[gi * ldd + gj] = my_val;
    } else {
      const unsigned mask = 0xFFFFFFFFu;
      const __half mate = __shfl_down_sync(mask, my_val, 1);
      if ((lane & 1) == 0) {
        const int gc_even = gj;
        atomicAddHalf2Packed(&D[gi * ldd + gc_even],
                             my_val,
                             ((gc_even + 1) < N) ? mate : __float2half(0.0f));
      }
    }
  }
}


// GEMV (M==1) groups-parallel kernel with direct fp16 atomics
// Configurable groups-per-block (G): block covers G consecutive 128-groups and 32 columns
// Optimization: treat -0 nibble (0x8) as zero in the dequant path, but separately
// accumulate sum of A-elements where -0 occurred, and add hfixB * sumA_negzero once.
// GEMV (M==1) column-parallel blocks, K-group parallel across grid.y.
// Each warp in the block covers 32 columns; all warps share the same K/128 group.
// Block covers (32*G) columns and exactly 1 K-group (128 K elems).
template<int G>
__global__ void razer_gemv_cuda(
  const __half*   __restrict__ A_fp16,   // [K]
  const uint32_t* __restrict__ B_packed, // [K/8, N]
  const __half*   __restrict__ B_scale,  // [K/128, N]
  __half*         __restrict__ D,        // [N]
  int N, int Ktotal)
{
  const int lane = threadIdx.x & 31;   // 0..31
  const int warp = threadIdx.y;        // 0..G-1

  // This block covers 32*G consecutive columns. Each warp covers 32 columns.
  const int n = blockIdx.x * (32 * G) + warp * 32 + lane;

  // Each block covers exactly one K/128 group, selected by blockIdx.y.
  const int groups_total = Ktotal >> 7;        // K/128
  const int kg_group     = blockIdx.y;         // 1 group per block
  const bool col_valid   = (n < N);
  const bool grp_valid   = (kg_group < groups_total);

  // Shared A tile for this K-group (same for all warps in the block), stored as half2.
  __shared__ __half2 As2_tile[64];             // 64 half2 == 128 half

  // Load A tile once (warp 0 does it; others wait).
  // Assumes Ktotal % 128 == 0 and A_fp16 base is 4B-aligned for half2 loads.
  if (warp == 0 && grp_valid) {
    const int k0 = kg_group << 7;              // *128
    const __half2* A2 = reinterpret_cast<const __half2*>(A_fp16 + k0);

    #pragma unroll
    for (int t2 = lane; t2 < 64; t2 += 32) {
      As2_tile[t2] = __ldg(&A2[t2]);
    }
  }
  __syncthreads();

  // Per-(group, column) params: scalar fp16 load (alignment-safe) + bitwise fix.
  __half hscale = __float2half(0.0f);
  __half hscale_clean = __float2half(0.0f);
  __half hfixB  = __float2half(0.0f);
  if (grp_valid && col_valid) {
    hscale = B_scale[kg_group * N + n];
    const uint16_t s_bits = __half_as_ushort(hscale);
    const uint16_t f_bits = (s_bits & 0x4000u) ? 0x4800u : 0x4500u; // fp16(8.0) : fp16(5.0)
    hfixB = __ushort_as_half(f_bits);
    hscale_clean = __ushort_as_half((uint16_t)(s_bits & (uint16_t)~0x4000u));
  }

  // Accumulators: normal contributions + fixup sum for +0 nibbles (0x0)
  __half2 acc2     = __float2half2_rn(0.0f);
  __half  sum_fixA = __float2half(0.0f);

  if (grp_valid && col_valid) {
    #pragma unroll
    for (int w_off = 0; w_off < 16; ++w_off) {
      const int widx = (kg_group << 4) + w_off;     // kg*16 + w_off
      const uint32_t w = B_packed[widx * N + n];    // coalesced within each warp

      // For this word: 8 A elems => 4 half2 entries.
      const int a2_base = (w_off << 2);             // w_off * 4 half2

      const uint8_t b0 = (uint8_t)(w);
      const uint8_t b1 = (uint8_t)(w >> 8);
      const uint8_t b2 = (uint8_t)(w >> 16);
      const uint8_t b3 = (uint8_t)(w >> 24);

      const __half2* Arow = &As2_tile[a2_base];

      // Still batches all FMAs at the end (as per the current best ordering).
      const __half2 a20 = Arow[0];
      const __half2 a21 = Arow[1];
      const __half2 a22 = Arow[2];
      const __half2 a23 = Arow[3];

      // Avoid named bool temporaries in the hot loop
      if (!(b0 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a20));
      if (b0 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a20));

      if (!(b1 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a21));
      if (b1 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a21));

      if (!(b2 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a22));
      if (b2 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a22));

      if (!(b3 & 0x0F)) sum_fixA = __hadd(sum_fixA, __low2half(a23));
      if (b3 < 16)      sum_fixA = __hadd(sum_fixA, __high2half(a23));

      // Try: batch all fixups first, then do all FMAs at the end.
      // Move all raw decodes to immediately before the HFMA block.
      const __half2_raw raw0 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b0, __NV_E2M1);
      const __half2_raw raw1 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b1, __NV_E2M1);
      const __half2_raw raw2 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b2, __NV_E2M1);
      const __half2_raw raw3 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b3, __NV_E2M1);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw0), a20, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw1), a21, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw2), a22, acc2);
      acc2 = __hfma2(*reinterpret_cast<const __half2*>(&raw3), a23, acc2);
    }
  }

  // Fold half2 accumulator + apply fixup
  __half acc_h = __hadd(__low2half(acc2), __high2half(acc2));
  acc_h = __hadd(acc_h, __hmul(sum_fixA, hfixB));

  // Scale and atomic add to fp16 output using half2 packing within each warp
  const unsigned mask = 0xFFFFFFFFu;
  const __half my_val = (grp_valid && col_valid) ? __hmul(acc_h, hscale_clean) : __float2half(0.0f);
  const __half mate   = __shfl_down_sync(mask, my_val, 1);

  if ((lane & 1) == 0) {
    const int n_even = n; // even lane => even column within this warp's 32-column slice
    if (n_even < N) {
      atomicAddHalf2Packed(&D[n_even],
                           my_val,
                           ((n_even + 1) < N) ? mate : __float2half(0.0f));
    }
  }
}


// Experimental g16 GEMV path (M==1 only). We process eight 16-element groups at a time
// so the grid.y extent remains K/128, matching the canonical g128 GEMV decomposition.
template<int G>
__global__ void razer_gemv_g16_fp8_groups8(
  const __half*      __restrict__ A_fp16,        // [K]
  const uint32_t*    __restrict__ B_packed,      // [K/8, N]
  const uint32_t*    __restrict__ B_scale_fp8,   // [K/128, N, 2] packed fp8 scales
  __half*            __restrict__ D,             // [N]
  int N, int Ktotal)
{
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.y;
  const int col = blockIdx.x * (32 * G) + warp * 32 + lane;
  const int kg8 = blockIdx.y; // one super-group of eight 16-element groups
  const int groups8_total = Ktotal >> 7; // K/128
  if (kg8 >= groups8_total) return;

  __shared__ __half2 As2_tile[64];
  if (warp == 0) {
    const int k0 = kg8 << 7;
    const __half2* A2 = reinterpret_cast<const __half2*>(A_fp16 + k0);
    #pragma unroll
    for (int t2 = lane; t2 < 64; t2 += 32) {
      As2_tile[t2] = __ldg(&A2[t2]);
    }
  }
  __syncthreads();

  const bool col_valid = (col < N);
  __half2 acc2 = __float2half2_rn(0.0f);

  if (col_valid) {
    const int scale_base = ((kg8 * N) + col) * 2;
    const uint32_t s0 = __ldg(&B_scale_fp8[scale_base + 0]);
    const uint32_t s1 = __ldg(&B_scale_fp8[scale_base + 1]);

    #pragma unroll
    for (int sg = 0; sg < 8; ++sg) {
      const int kg16 = (kg8 << 3) + sg;
      const int wbase = (kg16 << 1);
      const uint32_t pack = (sg < 4) ? s0 : s1;
      const int shift = 8 * (sg & 3);
      const uint8_t sb = (uint8_t)((pack >> shift) & 0xFFu);
      const __half hscale = decode_fp8_scale_from_bits(sb);
      const __half hfixB = fix_from_fp8_meta_bits(sb);
      const __half2 s2 = __halves2half2(hscale, hscale);
      __half sum_fixA = __float2half(0.0f);

      #pragma unroll
      for (int w_off = 0; w_off < 2; ++w_off) {
        const uint32_t w = __ldg(&B_packed[(wbase + w_off) * N + col]);
        const uint8_t b0 = (uint8_t)(w);
        const uint8_t b1 = (uint8_t)(w >> 8);
        const uint8_t b2 = (uint8_t)(w >> 16);
        const uint8_t b3 = (uint8_t)(w >> 24);

        const int a2_base = (sg * 8) + (w_off << 2);
        const __half2 a20 = As2_tile[a2_base + 0];
        const __half2 a21 = As2_tile[a2_base + 1];
        const __half2 a22 = As2_tile[a2_base + 2];
        const __half2 a23 = As2_tile[a2_base + 3];

        if (!(b0 & 0x0F)) sum_fixA = __hadd(sum_fixA, __hmul(__low2half(a20), hscale));
        if (b0 < 16)      sum_fixA = __hadd(sum_fixA, __hmul(__high2half(a20), hscale));
        if (!(b1 & 0x0F)) sum_fixA = __hadd(sum_fixA, __hmul(__low2half(a21), hscale));
        if (b1 < 16)      sum_fixA = __hadd(sum_fixA, __hmul(__high2half(a21), hscale));
        if (!(b2 & 0x0F)) sum_fixA = __hadd(sum_fixA, __hmul(__low2half(a22), hscale));
        if (b2 < 16)      sum_fixA = __hadd(sum_fixA, __hmul(__high2half(a22), hscale));
        if (!(b3 & 0x0F)) sum_fixA = __hadd(sum_fixA, __hmul(__low2half(a23), hscale));
        if (b3 < 16)      sum_fixA = __hadd(sum_fixA, __hmul(__high2half(a23), hscale));

        const __half2_raw r0 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b0, __NV_E2M1);
        const __half2_raw r1 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b1, __NV_E2M1);
        const __half2_raw r2 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b2, __NV_E2M1);
        const __half2_raw r3 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b3, __NV_E2M1);

        acc2 = __hfma2(*reinterpret_cast<const __half2*>(&r0), __hmul2(a20, s2), acc2);
        acc2 = __hfma2(*reinterpret_cast<const __half2*>(&r1), __hmul2(a21, s2), acc2);
        acc2 = __hfma2(*reinterpret_cast<const __half2*>(&r2), __hmul2(a22, s2), acc2);
        acc2 = __hfma2(*reinterpret_cast<const __half2*>(&r3), __hmul2(a23, s2), acc2);
      }

      const __half fix_contrib = __hmul(sum_fixA, hfixB);
      acc2 = __hadd2(acc2, __halves2half2(fix_contrib, __float2half(0.0f)));
    }
  }

  const __half acc_h = __hadd(__low2half(acc2), __high2half(acc2));
  const __half my_val = col_valid ? acc_h : __float2half(0.0f);
  const unsigned mask = 0xFFFFFFFFu;
  const __half mate = __shfl_down_sync(mask, my_val, 1);

  if ((lane & 1) == 0) {
    const int col_even = col;
    if (col_even < N) {
      atomicAddHalf2Packed(
        &D[col_even],
        my_val,
        ((col_even + 1) < N) ? mate : __float2half(0.0f));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// ATen wrappers
////////////////////////////////////////////////////////////////////////////////

// Perf GEMM entry
RazerGemmLaunchConfig razer_gemm(
  at::Tensor fA,              // [M, K], fp16
  at::Tensor qB,              // [K/8, N], int32, canonical FP4 packed
  at::Tensor scaling_factors, // g128: [K/128, N] fp16; g16: [K/128, N, 2] int32
  at::Tensor out,             // [M, N], fp16
  int groupsize,
  int impl,
  int split_k,
  int gemv_g)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    CONTIGUOUS(fA); CONTIGUOUS(qB); CONTIGUOUS(scaling_factors); CONTIGUOUS(out);
    CUDA_CHECK_IN(fA); CUDA_CHECK_IN(qB); CUDA_CHECK_IN(scaling_factors); CUDA_CHECK_IN(out);
    DTYPE_CHECK(fA, torch::kHalf);
    DTYPE_CHECK(qB, torch::kInt32);
    DTYPE_CHECK(out, torch::kHalf);

    const int M = fA.size(0);
    const int K = fA.size(1);
    const int N = qB.size(1);

    RazerGemmLaunchConfig cfg;
    cfg.impl = impl;
    cfg.split_k = 1;
    cfg.gemv_g = 0;
    cfg.small_r = 0;

    TORCH_CHECK(groupsize == 16 || groupsize == 128, "groupsize must be 16 or 128");
    TORCH_CHECK((K % 128) == 0, "K must be multiple of 128");
    TORCH_CHECK((K %   8) == 0, "K must be multiple of 8");
    TORCH_CHECK(qB.size(0) == K/8, "qB must be [K/8, N]");
    TORCH_CHECK(out.sizes() == at::IntArrayRef({M, N}), "out must be [M, N]");

    const __half*   A_ptr   = reinterpret_cast<const __half*>(fA.data_ptr<at::Half>());
    const uint32_t* qB_u32  = reinterpret_cast<const uint32_t*>(qB.data_ptr<int32_t>());
    __half*         out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

    const __half*   sc_ptr = nullptr;
    const uint32_t* sc16_ptr = nullptr;
    if (groupsize == 128) {
      DTYPE_CHECK(scaling_factors, torch::kHalf);
      TORCH_CHECK(scaling_factors.sizes() == at::IntArrayRef({K/128, N}), "g128 scales must be [K/128, N]");
      sc_ptr = reinterpret_cast<const __half*>(scaling_factors.data_ptr<at::Half>());
    } else {
      DTYPE_CHECK(scaling_factors, torch::kInt32);
      TORCH_CHECK(
        scaling_factors.dim() == 3 &&
        scaling_factors.size(0) == K / 128 &&
        scaling_factors.size(1) == N &&
        scaling_factors.size(2) == 2,
        "g16 scales must be [K/128, N, 2] int32");
      sc16_ptr = reinterpret_cast<const uint32_t*>(scaling_factors.data_ptr<int32_t>());
    }

    const RazerImpl impl_e = (RazerImpl)impl;
    const bool force_gemv  = (impl_e == RazerImpl::Gemv);
    const bool force_small = (impl_e == RazerImpl::Small);

    auto choose_gemv_grouping = [&] () {
      const int groups = K / 128;
      const int candidates[4] = {8, 4, 2, 1};
      int chosen_G = 1;
      if (gemv_g == -1) {
        chosen_G = 4;
      } else if (gemv_g == 0) {
        for (int ci = 0; ci < 4; ++ci) {
          const int G = candidates[ci];
          const int grid_x = (N + (32 * G - 1)) / (32 * G);
          const long total_grid = (long)grid_x * (long)groups;
          if (total_grid >= 16384) { chosen_G = G; break; }
        }
      } else {
        TORCH_CHECK(gemv_g == 1 || gemv_g == 2 || gemv_g == 4 || gemv_g == 8, "gemv_g must be one of {-1,0,1,2,4,8}");
        chosen_G = gemv_g;
      }
      return chosen_G;
    };

    auto run_gemv_g128 = [&] () {
      out.zero_();
      const int groups = K / 128;
      const int chosen_G = choose_gemv_grouping();
      cfg.impl = (int)RazerImpl::Gemv;
      cfg.gemv_g = chosen_G;
      dim3 threads(32, chosen_G);
      dim3 grid((N + (32 * chosen_G - 1)) / (32 * chosen_G), groups);
      switch (chosen_G) {
        case 8:
          razer_gemv_cuda<8><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc_ptr, out_ptr, N, K);
          break;
        case 4:
          razer_gemv_cuda<4><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc_ptr, out_ptr, N, K);
          break;
        case 2:
          razer_gemv_cuda<2><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc_ptr, out_ptr, N, K);
          break;
        default:
          razer_gemv_cuda<1><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc_ptr, out_ptr, N, K);
          break;
      }
    };

    auto run_gemv_g16 = [&] () {
      out.zero_();
      const int groups = K / 128;
      const int chosen_G = choose_gemv_grouping();
      cfg.impl = (int)RazerImpl::Gemv;
      cfg.gemv_g = chosen_G;
      dim3 threads(32, chosen_G);
      dim3 grid((N + (32 * chosen_G - 1)) / (32 * chosen_G), groups);
      switch (chosen_G) {
        case 8:
          razer_gemv_g16_fp8_groups8<8><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc16_ptr, out_ptr, N, K);
          break;
        case 4:
          razer_gemv_g16_fp8_groups8<4><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc16_ptr, out_ptr, N, K);
          break;
        case 2:
          razer_gemv_g16_fp8_groups8<2><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc16_ptr, out_ptr, N, K);
          break;
        default:
          razer_gemv_g16_fp8_groups8<1><<<grid, threads, 0, stream>>>(A_ptr, qB_u32, sc16_ptr, out_ptr, N, K);
          break;
      }
    };

    auto run_small_gemm = [&] () {
      const int R = (M < 8) ? M : 8;
      cfg.impl = (int)RazerImpl::Small;
      cfg.small_r = R;
      dim3 block(32, R, 1);

      const int grid_x = (M + R - 1) / R;
      const int grid_y = (N + 31) / 32;
      const int groups_total = K / 128;
      int splitK = 1;
      if (groups_total > 1) {
        const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
        const int sms = props ? props->multiProcessorCount : 0;
        const long base_blocks = (long)grid_x * (long)grid_y;
        int target_warps_per_sm = 256 / max(1, R);
        target_warps_per_sm = max(64, min(128, target_warps_per_sm));
        const long target_warps = (long)max(1, sms) * (long)target_warps_per_sm;
        const long target_blocks = (target_warps + (long)R - 1) / (long)R;

        if (base_blocks < target_blocks) {
          int s = 1;
          while (s < groups_total && (base_blocks * (long)s) < target_blocks) {
            s <<= 1;
          }
          if (s > groups_total) s = groups_total;
          splitK = s;
        }
      }

      if (split_k >= 1) {
        splitK = split_k;
      }
      if (splitK < 1) splitK = 1;
      if (splitK > groups_total) splitK = groups_total;
      cfg.split_k = splitK;
      dim3 grid(grid_x, grid_y, splitK);
      if (splitK != 1) {
        out.zero_();
      }

      size_t smem = (size_t)(R * 128 + 32 + 32) * sizeof(__half);
      smem = (smem + 3u) & ~size_t(3u);
      smem += (size_t)(16 * 32) * sizeof(uint32_t);

      razer_gemm_cuda<<<grid, block, smem, stream>>>(A_ptr, qB_u32, sc_ptr, out_ptr, M, N, K);
    };

    if (groupsize == 16) {
      TORCH_CHECK(!force_small, "groupsize=16 does not support the small-GEMM kernel path");
      TORCH_CHECK(M == 1, "groupsize=16 extension path only supports M==1; use razer_dequant + GEMM fallback otherwise");
      run_gemv_g16();
      return cfg;
    }

    if (force_gemv) {
      TORCH_CHECK(M == 1, "impl=gemv requires M==1");
      run_gemv_g128();
    } else if (force_small) {
      TORCH_CHECK(M >= 1, "impl=small requires M>=1");
      run_small_gemm();
    } else {
      if (M == 1) {
        run_gemv_g128();
      } else {
        run_small_gemm();
      }
    }

    return cfg;
}

// Dequant entry: rebuild dense B [K,N] (fp16) from packed qB/scales
// using the same intrinsics+remap as the kernel.
//
// Performance notes:
// - Dequantizes 2 consecutive K elements per thread via a packed byte -> half2.
// - Assumes K is a multiple of 8 (and thus even).
__global__ void dequant_kernel_g128(
  const uint32_t* __restrict__ B_packed, int N, int K,
  const __half*   __restrict__ B_scale,
  __half*         __restrict__ B_dense)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int kp  = blockIdx.y * blockDim.y + threadIdx.y;
  const int k0  = kp << 1;
  if (col >= N || k0 >= K) return;

  const int kg = k0 >> 7;
  const __half hscale = __ldg(&B_scale[kg * N + col]);
  const uint16_t s_bits = __half_as_ushort(hscale);
  const __half hfixB = scale_msb2_to_fixval_from_bits(s_bits);
  const __half hscale_clean = __ushort_as_half((uint16_t)(s_bits & (uint16_t)~0x4000u));

  const uint32_t word = __ldg(&B_packed[(k0 >> 3) * N + col]);
  const int slot = (k0 & 7);
  const uint8_t packed_byte = (uint8_t)((word >> (4 * slot)) & 0xFFu);
  const __half2 out2 = dequant_fix_scale_e2m1(packed_byte, hfixB, hscale_clean);
  B_dense[k0 * N + col] = __low2half(out2);
  B_dense[(k0 + 1) * N + col] = __high2half(out2);
}

__global__ void dequant_kernel_g16(
  const uint32_t* __restrict__ B_packed, int N, int K,
  const uint32_t* __restrict__ B_scale_fp8,
  __half*         __restrict__ B_dense)
{
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int kp  = blockIdx.y * blockDim.y + threadIdx.y;
  const int k0  = kp << 1;
  if (col >= N || k0 >= K) return;

  const int kg16 = k0 >> 4;
  const int kg8 = kg16 >> 3;
  const int sg = kg16 & 7;
  const uint32_t pack = __ldg(&B_scale_fp8[((kg8 * N) + col) * 2 + (sg >> 2)]);
  const uint8_t sb = (uint8_t)((pack >> (8 * (sg & 3))) & 0xFFu);
  const __half hscale = decode_fp8_scale_from_bits(sb);
  const __half hfixB = fix_from_fp8_meta_bits(sb);

  const uint32_t word = __ldg(&B_packed[(k0 >> 3) * N + col]);
  const int slot = (k0 & 7);
  const uint8_t packed_byte = (uint8_t)((word >> (4 * slot)) & 0xFFu);
  const __half2 out2 = dequant_fix_scale_e2m1(packed_byte, hfixB, hscale);
  B_dense[k0 * N + col] = __low2half(out2);
  B_dense[(k0 + 1) * N + col] = __high2half(out2);
}

at::Tensor razer_dequant(
  at::Tensor qB,
  at::Tensor scaling_factors,
  int64_t K,
  int groupsize)
{
  CUDA_CHECK_IN(qB); CUDA_CHECK_IN(scaling_factors);
  DTYPE_CHECK(qB, torch::kInt32);
  TORCH_CHECK(groupsize == 16 || groupsize == 128, "groupsize must be 16 or 128");

  const int N = qB.size(1);
  TORCH_CHECK((K % 2) == 0, "K must be even");
  TORCH_CHECK((K % 128) == 0, "K must be multiple of 128");
  TORCH_CHECK(qB.size(0) == K/8, "qB must be [K/8, N]");

  auto out = torch::empty({(int)K, N}, qB.options().dtype(torch::kHalf));
  const uint32_t* qB_u32 = reinterpret_cast<const uint32_t*>(qB.data_ptr<int32_t>());
  __half* out_ptr = reinterpret_cast<__half*>(out.data_ptr<at::Half>());

  dim3 block(32, 8);
  const int K_pairs = (int)K / 2;
  dim3 grid((N + block.x - 1)/block.x, (K_pairs + block.y - 1)/block.y);

  if (groupsize == 128) {
    DTYPE_CHECK(scaling_factors, torch::kHalf);
    TORCH_CHECK(scaling_factors.sizes() == at::IntArrayRef({(int)(K/128), N}), "g128 scales must be [K/128, N]");
    const __half* sc_ptr = reinterpret_cast<const __half*>(scaling_factors.data_ptr<at::Half>());
    dequant_kernel_g128<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(qB_u32, N, (int)K, sc_ptr, out_ptr);
  } else {
    DTYPE_CHECK(scaling_factors, torch::kInt32);
    TORCH_CHECK(
      scaling_factors.dim() == 3 &&
      scaling_factors.size(0) == K / 128 &&
      scaling_factors.size(1) == N &&
      scaling_factors.size(2) == 2,
      "g16 scales must be [K/128, N, 2] int32");
    const uint32_t* sc_ptr = reinterpret_cast<const uint32_t*>(scaling_factors.data_ptr<int32_t>());
    dequant_kernel_g16<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(qB_u32, N, (int)K, sc_ptr, out_ptr);
  }

  return out;
}
