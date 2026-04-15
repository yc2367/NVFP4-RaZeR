# gpt_fast_integration

This directory contains the trimmed `gpt-fast` integration layer used by the public NVFP4-RaZeR inference artifact.

## Credit

This subtree is adapted from Meta's [`gpt-fast`](https://github.com/meta-pytorch/gpt-fast). The original project provides the base model conversion flow, core model definition, and the overall generation structure that this artifact builds on. The upstream license is preserved in [../third_party_licenses/GPT_FAST_LICENSE](/data/syphon/NVFP4-RaZeR/inference/third_party_licenses/GPT_FAST_LICENSE).

## Scope

This subtree is intended for throughput-oriented inference measurement of the released weight-only kernels. It is not an accuracy-optimized quantization pipeline, and it does not add extra accuracy-oriented methods such as AWQ-style activation-aware quantization.

The checkpoints produced here should be interpreted as performance-measurement checkpoints for the released inference implementations.

## What Is Here

Included entry points:

- `quantize.py`: produces `razer`, `marlinrazer`, and `marlinfp4` checkpoints
- `generate.py`: loads those checkpoints for single-GPU generation
- `benchmark.py`: runs a throughput sweep over the supported checkpoints
- `plot_benchmark_results.py`: plots the benchmark JSON output

Included linear/kernel paths:

- `RaZeRLinear`
- `MarlinRazerLinear`
- `MarlinFP4Linear`

## Environment

At minimum, the Python environment needs:

- `torch`
- `sentencepiece`
- `tiktoken`
- `tokenizers` if your model uses `tokenizer.json` instead of `tokenizer.model`

The easiest setup path is to start from the original `gpt-fast` environment and install the RaZeR extension from this repo:

```bash
cd /path/to/NVFP4-RaZeR
pip install -e inference/kernel_extensions/razer_cuda
```

If you want to use the `marlinrazer` or `marlinfp4` paths, install those extensions from their sibling directories once their source is present.

All commands below assume you are running from this directory:

```bash
cd /path/to/NVFP4-RaZeR/inference/gpt_fast_integration
```

## Model Files

This subtree expects the converted `gpt-fast` checkpoint layout, not a raw Hugging Face `model.safetensors` file.

A model directory should look like:

```text
<MODEL_DIR>/
  model.pth
  config.json
  tokenizer.model    # or tokenizer.json
```

The loader also uses the checkpoint directory name to identify the model architecture, so keep the original model directory name such as `Llama-3.2-1B`, `Llama-3.1-8B`, or `Qwen3-8B`.

## Downloading Base Checkpoints

This artifact does not duplicate the upstream weight-download helpers. The recommended flow is:

1. Use the original `gpt-fast` repository to prepare the base model directory.
2. Copy or point the commands below at that prepared directory.

For example, upstream `gpt-fast` uses:

```bash
export MODEL_REPO=meta-llama/Llama-3.2-1B
./scripts/prepare.sh $MODEL_REPO
```

After that step, you should have a directory like:

```text
checkpoints/meta-llama/Llama-3.2-1B/model.pth
```

with the accompanying tokenizer and config files in the same directory.

## Quantization

### RaZeR g128

```bash
python quantize.py \
  --checkpoint_path /path/to/checkpoints/meta-llama/Llama-3.2-1B/model.pth \
  --mode razer \
  --groupsize 128 \
  --fp_dtype fp16 \
  --use_cuda
```

This writes:

```text
/path/to/checkpoints/meta-llama/Llama-3.2-1B/model_razer.g128.fp16.pth
```

### RaZeR g16

```bash
python quantize.py \
  --checkpoint_path /path/to/checkpoints/meta-llama/Llama-3.2-1B/model.pth \
  --mode razer \
  --groupsize 16 \
  --fp_dtype fp16 \
  --use_cuda
```

This writes:

```text
/path/to/checkpoints/meta-llama/Llama-3.2-1B/model_razer.g16.fp16.pth
```

`g16` is an experimental path: it uses the dedicated kernel for `M == 1` decode and dequantization plus dense GEMM for larger effective batch sizes.

### Marlin-RaZeR / Marlin-FP4

Once the corresponding extensions are installed, the same interface applies:

```bash
python quantize.py \
  --checkpoint_path /path/to/model.pth \
  --mode marlinrazer \
  --groupsize 128 \
  --fp_dtype fp16 \
  --use_cuda

python quantize.py \
  --checkpoint_path /path/to/model.pth \
  --mode marlinfp4 \
  --groupsize 128 \
  --fp_dtype fp16 \
  --use_cuda
```

## Generation

### Basic run

```bash
python generate.py \
  --checkpoint_path /path/to/checkpoints/meta-llama/Llama-3.2-1B/model_razer.g128.fp16.pth \
  --prompt "The capital of France is" \
  --batch_size 1 \
  --max_new_tokens 128 \
  --device cuda \
  --fp_dtype fp16
```

### Compile decode

`--compile` compiles the decode path. This is the flag to use when measuring steady-state decode throughput.

```bash
python generate.py \
  --checkpoint_path /path/to/checkpoints/meta-llama/Llama-3.2-1B/model_razer.g128.fp16.pth \
  --prompt "The capital of France is" \
  --batch_size 1 \
  --max_new_tokens 128 \
  --device cuda \
  --fp_dtype fp16 \
  --compile \
  --razer_matmul kernel
```

For RaZeR specifically, `--razer_matmul` can be set to:

- `auto`: default heuristic
- `kernel`: force the kernel path when supported
- `dequant_gemm`: force dequantization plus dense GEMM

When `--compile` is enabled, the first generation iteration is a compile warmup. Measure steady-state throughput using later iterations or the benchmark script below.

## Benchmarking

`benchmark.py` runs multiple checkpoints and batch sizes and writes a JSON results file for throughput measurement. The example below matches the decode-compiled measurement path used for throughput evaluation.

Example: compare `g128` and `g16` on batch-1 decode.

```bash
python benchmark.py \
  --checkpoint_dirs /path/to/checkpoints/meta-llama/Llama-3.2-1B \
  --checkpoint_files model_razer.g128.fp16.pth model_razer.g16.fp16.pth \
  --batch_sizes 1 \
  --prompt "The capital of France is" \
  --max_new_tokens 128 \
  --num_samples 5 \
  --device cuda \
  --fp_dtype fp16 \
  --out benchmark_results.json
```
