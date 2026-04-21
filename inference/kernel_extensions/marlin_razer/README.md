# RaZeR FP4 Weight-Only Linear Kernel

This repository contains the modified CUDA kernel and Python bindings used for the RaZeR FP4 weight-only linear layer described in [RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping
](https://arxiv.org/abs/2501.04052).

The implementation is derived from the [Marlin](https://github.com/IST-DASLab/marlin) weight-only linear kernel and adapts that codebase to an FP16 x FP4 execution path with RaZeR-specific packing, scaling, and negative-zero remap handling.

## Overview

The repository provides:

- `marlin_razer/`: CUDA extension, Python bindings, layer wrapper, and launch configuration table
- `test.py`: correctness tests for the FP4 path
- `benchmark.py`: dense-vs-kernel benchmarking utilities
- `autotune.py`: launch parameter search helpers
- `generate_marlin_small_table.py`: tuning-table generation utility
- `razer-fp4-weight-only.pdf`: accompanying paper / technical report

The main extension package is `marlin_razer`, which exposes:

- `marlin_razer.mul(...)`: low-level kernel entry point
- `marlin_razer.Layer`: high-level PyTorch module wrapper with weight packing utilities

## Requirements

- Linux with NVIDIA CUDA toolkit
- CUDA `>= 12.8`
- NVIDIA GPU with compute capability `>= 10.0`
- Python with `torch`
- `numpy`

The extension is built with PyTorch's CUDA extension tooling and expects `nvcc` to be compatible with the installed PyTorch build.

## Installation

Install the package from the repository root:

```bash
pip install .
```

This builds the `marlin_razer_cuda` extension and installs the `marlin-razer` Python package locally.

## Usage

### High-level layer wrapper

```python
import torch
from marlin_razer import Layer

linear_fp16 = torch.nn.Linear(4096, 4096, bias=False).half().cuda()
layer = Layer(
    infeatures=linear_fp16.in_features,
    outfeatures=linear_fp16.out_features,
    groupsize=128,
).cuda()

layer.quick_quantize_razer4(linear_fp16)

x = torch.randn(16, 4096, dtype=torch.half, device="cuda")
y = layer(x)
```

### Low-level kernel call

```python
import torch
from marlin_razer import mul

m, k, n = 16, 4096, 4096
A = torch.randn((m, k), dtype=torch.half, device="cuda")
B = torch.randint(0, 16, (k // 16, n * 16 // 8), dtype=torch.int32, device="cuda")
C = torch.empty((m, n), dtype=torch.half, device="cuda")
s = torch.randn((k // 128, n), dtype=torch.half, device="cuda")
workspace = torch.zeros(n // 128 * 16, dtype=torch.int32, device="cuda")

mul(A, B, C, s, workspace)
```

`B` and `s` must already be packed into the layout expected by the kernel. The recommended path is to use `Layer.pack(...)` or `Layer.quick_quantize_razer4(...)` rather than preparing these tensors manually.

## Validation

Run correctness tests:

```bash
python test.py
```

Run the benchmark script:

```bash
python benchmark.py
```

If you want to regenerate or tune launch configurations, use:

```bash
python autotune.py
```

## Relationship To Marlin

This repository is a derivative work of the original Marlin kernel:

- Upstream project: [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin)
- Original paper: Frantar et al., "MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models"

The RaZeR implementation reuses Marlin's overall kernel structure and optimization strategy, while modifying the code for the RaZeR FP4 weight-only format and associated quantization flow.

## License

This repository is released under the Apache License 2.0. See [LICENSE](./LICENSE).

Because this code is derived from Marlin, the release preserves Apache-2.0 licensing and includes upstream attribution in [NOTICE](./NOTICE). If you redistribute this repository or derivative versions of it, keep the license and attribution notices intact.

## Citation

If you use this repository in academic work, cite the RaZeR paper:

```bibtex
@article{chen2026razer,
  title   = {RaZeR: Pushing the Limits of NVFP4 Quantization with Redundant Zero Remapping},
  author  = {Chen, Yuzong and Dai, Xilai and Hyun, Jake and Chang, Chi-Chih and Jang, Wonsuk and Wu, Yuheng and Tambe, Thierry and Seo, Jae-sun and Abdelfattah, Mohamed S.},
  journal = {arXiv preprint arXiv:2501.04052},
  year    = {2026}
}
```

