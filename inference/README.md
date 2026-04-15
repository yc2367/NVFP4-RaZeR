# Inference Artifact

This subtree contains the public inference-facing artifact for NVFP4-RaZeR.

It is split into three parts:

- `kernel_extensions/`: CUDA/C++ extension sources used by the weight-only inference stack.
- `gpt_fast_integration/`: a trimmed `gpt-fast` integration layer for quantization, loading, generation, and benchmarking.
- `w4a4/`: the two-pass CUTLASS W4A4 benchmark code used for the current-hardware appendix experiments.

The weight-only integration is organized around three kernel families:

- `razer_cuda`: CUDA-core RaZeR weight-only path.
- `marlin_razer`: Marlin-RaZeR tensor-core path.
- `marlin_fp4`: Marlin-FP4 baseline path.

This artifact contains the inference code paths used for `razer`, `marlinrazer`, and `marlinfp4`.
