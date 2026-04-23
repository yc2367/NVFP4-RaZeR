# marlin-fp4

`marlin-fp4` is a research CUDA kernel for FP16 x FP4 matrix multiplication, adapted from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), which targets FP16 x INT4.

This repository is intended as open-source research code for experimenting with weight-only FP4 linear layers for LLM inference. The current implementation focuses on:

- FP16 activations and FP4 weights
- Marlin-style offline weight packing and scale shuffling
- PyTorch integration through a custom CUDA extension
- Per-device launch configuration lookup for tuned shapes

## Status

Current repository highlights:

- CUDA extension: [`marlin_fp4/marlin_fp4_cuda_kernel.cu`](./marlin_fp4/marlin_fp4_cuda_kernel.cu)
- PyTorch bindings: [`marlin_fp4/marlin_fp4_cuda.cpp`](./marlin_fp4/marlin_fp4_cuda.cpp)
- Python API: [`marlin_fp4/__init__.py`](./marlin_fp4/__init__.py)
- Tests: [`test.py`](./test.py)
- Benchmarks/autotuning: [`benchmark.py`](./benchmark.py), [`autotune.py`](./autotune.py)

## Requirements

- Linux
- NVIDIA GPU
- CUDA toolkit compatible with your PyTorch install
- PyTorch with CUDA support

The build currently sets `TORCH_CUDA_ARCH_LIST=12.0a+PTX` in [`setup.py`](./setup.py), the intended target is Blackwell (SM120) GPUs.

## Install

```bash
pip install .
```

If extension compilation fails, first verify that:

- `nvcc` is available
- the CUDA toolkit version matches the CUDA version used by PyTorch closely enough for extension builds
- your GPU architecture is supported by the build settings

## Quick Start

Low-level kernel call:

```python
import torch
from marlin_fp4 import mul

m, k, n = 16, 4096, 4096
A = torch.randn((m, k), dtype=torch.half, device="cuda")
B = torch.empty((k // 16, n * 16 // 8), dtype=torch.int32, device="cuda")  # packed FP4 weights
C = torch.empty((m, n), dtype=torch.half, device="cuda")
s = torch.empty((k // 128, n), dtype=torch.half, device="cuda")            # group scales
workspace = torch.zeros(n // 128 * 16, dtype=torch.int32, device="cuda")

mul(A, B, C, s, workspace)
```

High-level layer wrapper:

```python
import torch
from marlin_fp4 import Layer

linear = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.half).cuda()
layer = Layer(4096, 4096, groupsize=128).cuda()
layer.quick_quantize_fp4(linear)

x = torch.randn((1, 4096), dtype=torch.half, device="cuda")
y = layer(x)
```

## Validation And Benchmarking

Run the basic correctness test:

```bash
python test.py
```

Run the benchmark script:

```bash
python benchmark.py
```

Autotune launch parameters and regenerate the compact launch table:

```bash
python autotune.py
```

## Repository Notes

- The FP4 path currently uses E2M1-style 4-bit floating-point encoding and Marlin-style packed layouts.
- The project name refers to its Marlin lineage; it is an adaptation, not the original Marlin release.

## Attribution

This repository contains code adapted from:

- [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin), licensed under Apache-2.0

See [`NOTICE`](./NOTICE) for third-party attribution details and [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for bundled third-party license text where needed.

## License

This repository is distributed under the Apache License 2.0. See [`LICENSE`](./LICENSE).

