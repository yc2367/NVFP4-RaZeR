# razer_cuda

This directory contains the CUDA-core RaZeR extension source.

Build:

```bash
pip install -e .
```

The exported extension provides:

- `razer_gemm(A, qB, scales, bias, groupsize)` for weight-only matmul
- `razer_dequant(qB, scales, groupsize)` for explicit dense fallback paths

Supported group sizes:

- `128`: canonical RaZeR path
- `16`: experimental decode-oriented path with kernel acceleration for `M == 1` and dequantization fallback for larger `M`
