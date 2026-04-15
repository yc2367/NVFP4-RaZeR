import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
DEFAULT_CHECKPOINT_FILES = [
    "model.pth",
    "model_razer.g128.fp16.pth",
    "model_razer.g16.fp16.pth",
    "model_marlinrazer.g128.fp16.pth",
    "model_marlinfp4.g128.fp16.pth",
]


def _mode_specs_for_checkpoint(name: str) -> List[Dict[str, Optional[str]]]:
    if name.startswith("model_razer"):
        return [
            {"method": "razer", "mode": "razer_kernel", "razer_matmul": "kernel"},
            {"method": "razer", "mode": "razer_dequant_gemm", "razer_matmul": "dequant_gemm"},
        ]
    return [{"method": "normal", "mode": "normal", "razer_matmul": None}]


def _resolve_checkpoint_paths(checkpoint_dirs: List[Path], checkpoint_files: List[str]) -> List[Path]:
    paths: List[Path] = []
    for d in checkpoint_dirs:
        for f in checkpoint_files:
            p = (d / f).expanduser()
            if p.is_file():
                paths.append(p)
    return paths


def _run_generate(
    checkpoint_path: Path,
    *,
    batch_size: int,
    prompt: Optional[str],
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    compile: bool,
    compile_prefill: bool,
    device: str,
    fp_dtype: str,
    razer_matmul: Optional[str],
) -> Dict[str, Any]:
    sentinel = "__RAZER_BENCH__"
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "batch_size": int(batch_size),
        "prompt": prompt,
        "num_samples": int(num_samples),
        "max_new_tokens": int(max_new_tokens),
        "top_k": int(top_k),
        "temperature": float(temperature),
        "compile": bool(compile),
        "compile_prefill": bool(compile_prefill),
        "device": str(device),
        "fp_dtype": str(fp_dtype),
        "razer_matmul": razer_matmul,
    }
    code = f"""
import json
import generate

payload = {json.dumps(payload)}
metrics = generate.main(
    prompt=payload['prompt'],
    interactive=False,
    num_samples=payload['num_samples'],
    max_new_tokens=payload['max_new_tokens'],
    batch_size=payload['batch_size'],
    top_k=payload['top_k'],
    temperature=payload['temperature'],
    checkpoint_path=payload['checkpoint_path'],
    compile=payload['compile'],
    compile_prefill=payload['compile_prefill'],
    profile=None,
    draft_checkpoint_path=None,
    speculate_k=5,
    device=payload['device'],
    stream=False,
    fp_dtype=payload['fp_dtype'],
    noop_linears=False,
    razer_matmul=payload['razer_matmul'],
    quiet=True,
    return_metrics=True,
)
print('{sentinel}' + json.dumps(metrics, sort_keys=True))
"""
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parent,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"generate.main failed for {checkpoint_path.name} batch={batch_size}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith(sentinel):
            return json.loads(line[len(sentinel):])
    raise RuntimeError(f"Could not parse metrics for {checkpoint_path.name}; stdout was:\n{proc.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark public RaZeR inference artifact checkpoints")
    parser.add_argument("--checkpoint_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint_files", nargs="+", default=DEFAULT_CHECKPOINT_FILES)
    parser.add_argument("--out", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp_dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--compile_prefill", action="store_true")
    args = parser.parse_args()

    checkpoint_paths = _resolve_checkpoint_paths(args.checkpoint_dirs, args.checkpoint_files)
    if not checkpoint_paths:
        raise SystemExit("No checkpoints found for the requested directories/files")

    rows: List[Dict[str, Any]] = []
    for ckpt in checkpoint_paths:
        for spec in _mode_specs_for_checkpoint(ckpt.name):
            for batch_size in args.batch_sizes:
                metrics = _run_generate(
                    ckpt,
                    batch_size=batch_size,
                    prompt=args.prompt,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    compile=not args.no_compile,
                    compile_prefill=args.compile_prefill,
                    device=args.device,
                    fp_dtype=args.fp_dtype,
                    razer_matmul=spec['razer_matmul'],
                )
                rows.append({
                    "checkpoint_path": str(ckpt),
                    "checkpoint_name": ckpt.name,
                    "method": spec["method"],
                    "mode": spec["mode"],
                    "batch_size": int(batch_size),
                    **metrics,
                })
                print(
                    f"{ckpt.name} batch={batch_size} mode={spec['mode']} avg_trimmed={metrics['avg_tokens_per_sec_trimmed']:.2f}"
                )

    payload = {
        "batch_sizes": args.batch_sizes,
        "checkpoint_files": args.checkpoint_files,
        "results": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
