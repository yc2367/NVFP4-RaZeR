# w4a4

This directory contains the current-hardware W4A4 implementation sources used for the appendix experiments.

Source files:

- `razer-cutlass.cu`: two-pass RaZeR-CUTLASS implementation
- `cutlass_nvfp4_baseline_gemm.cu`: NVFP4 baseline
- `cutlass_mxfp8_baseline_gemm.cu`: MXFP8 baseline
- `cublas_fp16_baseline_gemm.cu`: cuBLAS FP16 baseline
- `razer_58_debug_shared.h`: shared helpers for the RaZeR-CUTLASS implementation

Prerequisites:

- CUDA toolkit with `nvcc`
- CUTLASS checkout available locally
- a toolchain/GPU target that supports `sm_120a` if you use the commands below unchanged

Example compile commands:

```bash
CUTLASS=/path/to/cutlass
SM=120a
INC="-I${CUTLASS}/include -I${CUTLASS}/tools/util/include"
GENCODE="-gencode arch=compute_${SM},code=sm_${SM}"

nvcc -O3 -std=c++17 ${INC} ${GENCODE}   razer-cutlass.cu -o razer_cutlass_two_pass_sm120a

nvcc -O3 -std=c++17 --expt-relaxed-constexpr ${INC} ${GENCODE}   cutlass_nvfp4_baseline_gemm.cu -o nvfp4_baseline_sm120a

nvcc -O3 -std=c++17 --expt-relaxed-constexpr ${INC} ${GENCODE}   cutlass_mxfp8_baseline_gemm.cu -o mxfp8_baseline_sm120a

nvcc -O3 -std=c++17 ${GENCODE}   cublas_fp16_baseline_gemm.cu -o cublas_fp16_baseline_sm120a -lcublas
```
