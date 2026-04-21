#!/usr/bin/env python3
# autotune_marlin.py

import os
import json
import time
import math
import csv
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
import marlin_razer
from generate_marlin_small_table import process as generate_launch_config


# Allowed launch configs
THREAD_CHOICES: List[Tuple[int, int]] = [(128, 128), (64, 256)]

# Kernel assumptions (matches your benchmark script)
GROUPSIZE = 128
MAX_PAR = 16


# -----------------------------
# Benchmark utilities
# -----------------------------
def _benchmark(fn, warmup: int = 10, iters: int = 200) -> float:
    """
    Similar philosophy to your benchmark:
    - no sync inside loop (hides launch overhead),
    - sync around timing boundaries,
    - return seconds / iter.
    """
    # Warmup + iters
    for i in range(warmup + iters):
        fn(i)

        if i == warmup - 1:
            torch.cuda.synchronize()
            t0 = time.time()

    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters


def _tflops(m: int, n: int, k: int, seconds: float) -> float:
    # GEMM FLOPs = 2*m*n*k
    return (2.0 * m * n * k) / seconds / 1e12


def _gbps_marlin(m: int, n: int, k: int, seconds: float) -> float:
    """
    Rough bandwidth estimate similar to your script.
    - A: fp16 -> 2 bytes
    - B: packed int32 length (k*n/8) -> 4 bytes
    - C: fp16 -> 2 bytes
    - scaling: fp16 -> 2 bytes, shape (k/groupsize, n)
    """
    bytes_a = 2 * (m * k)
    bytes_b = 4 * (k * n // 8)
    bytes_c = 2 * (m * n)
    bytes_s = 2 * ((k // GROUPSIZE) * n)

    total_bytes = bytes_a + bytes_b + bytes_c + bytes_s
    return total_bytes / seconds / 1e9


def _gbps_dense(m: int, n: int, k: int, seconds: float) -> float:
    # Dense FP16 matmul: A,B,C fp16 read/write approx
    bytes_a = 2 * (m * k)
    bytes_b = 2 * (k * n)
    bytes_c = 2 * (m * n)
    total_bytes = bytes_a + bytes_b + bytes_c
    return total_bytes / seconds / 1e9


# -----------------------------
# Problem generation
# -----------------------------
@dataclass
class Problems:
    As: List[torch.Tensor]
    Bs_fp16: List[torch.Tensor]
    Bs_marlin: List[torch.Tensor]
    Ss: List[torch.Tensor]
    Cs: List[torch.Tensor]


def make_problems(m: int, n: int, k: int, num_pairs: int, device: torch.device) -> Problems:
    """
    Create independent problems so we can cycle through them during timing,
    which is closer to inference behavior.
    """
    # Basic alignment checks (Marlin usually expects these)
    if (k * n) % 8 != 0:
        raise ValueError(f"(k*n) must be divisible by 8 for packed weights. Got k={k}, n={n}")
    if k % GROUPSIZE != 0:
        raise ValueError(f"k must be divisible by groupsize={GROUPSIZE}. Got k={k}")
    if n % 128 != 0:
        raise ValueError(f"n must be divisible by 128 (workspace/tile constraint). Got n={n}")

    As, Bs_fp16, Bs_marlin, Ss, Cs = [], [], [], [], []
    for _ in range(num_pairs):
        A = torch.randn((m, k), dtype=torch.half, device=device)
        B_fp16 = torch.randn((k, n), dtype=torch.half, device=device)

        # Packed weights as int32 (torch.int == int32)
        B_marlin = torch.randint(
            low=-2**31,
            high=2**31 - 1,
            size=(k * n // 8,),
            dtype=torch.int32,
            device=device,
        )

        S = torch.rand((k // GROUPSIZE, n), dtype=torch.half, device=device)
        C = torch.zeros((m, n), dtype=torch.half, device=device)

        As.append(A)
        Bs_fp16.append(B_fp16)
        Bs_marlin.append(B_marlin)
        Ss.append(S)
        Cs.append(C)

    torch.cuda.synchronize()
    return Problems(As=As, Bs_fp16=Bs_fp16, Bs_marlin=Bs_marlin, Ss=Ss, Cs=Cs)


# -----------------------------
# SMS sweep
# -----------------------------
def _sms_candidates(sm_hw: int) -> List[int]:
    """
    Choose SM counts in steps up to hardware SMs;
    include the exact hardware count if it is not a multiple of step.
    """
    step = 8
    candidates = list(range(step, sm_hw + 1, step))

    # Ensure the hardware count is present when not already covered.
    if sm_hw % step != 0 or not candidates:
        candidates.append(sm_hw)

    # Deduplicate while keeping ascending order.
    candidates = sorted({c for c in candidates if c > 0})
    return candidates



def _packed_b_bytes(k: int, n: int) -> int:
    """
    Marlin packed B uses 4-bit weights => 0.5 bytes per element.
    Total bytes = k*n/2.
    """
    return (k * n) // 2


def choose_num_pairs_from_l2(k: int, n: int, l2_cache_bytes: int, min_pairs: int = 2) -> int:
    """
    Choose NUM_PAIRS so that total B footprint is at least 2x L2 size.
    Enforces a minimum number of pairs.
    """
    b_bytes = _packed_b_bytes(k, n)
    if b_bytes <= 0:
        return min_pairs

    target = 2 * l2_cache_bytes
    pairs = math.ceil(target / b_bytes)
    return max(min_pairs, pairs)


# -----------------------------
# Tuner core
# -----------------------------
@dataclass
class TrialResult:
    m: int
    n: int
    k: int
    thread_k: int
    thread_n: int
    sms: int
    seconds_per_iter: float
    tflops: float
    gbps: float
    repeat_id: int
    mode: str  # sweep identifier for logging


def _run_marlin_trial(
    problems: Problems,
    thread_k: int,
    thread_n: int,
    sms: int,
    iters: int,
    warmup: int,
    repeat_id: int,
    mode: str,
) -> TrialResult:
    dev = problems.As[0].device
    m, k = problems.As[0].shape
    n = problems.Cs[0].shape[1]

    # Workspace: must be >= n/128 * MAX_PAR int entries
    workspace = torch.zeros((n // 128) * MAX_PAR, dtype=torch.int32, device=dev)

    num_pairs = len(problems.As)

    # keep outputs clean-ish
    for c in problems.Cs:
        c.zero_()
    workspace.zero_()
    torch.cuda.synchronize()

    def _kernel(i: int):
        idx = i % num_pairs
        marlin_razer.mul(
            problems.As[idx],
            problems.Bs_marlin[idx],
            problems.Cs[idx],
            problems.Ss[idx],
            workspace,
            thread_k=thread_k,
            thread_n=thread_n,
            sms=sms,
            max_par=MAX_PAR,
        )

    seconds = _benchmark(_kernel, warmup=warmup, iters=iters)
    return TrialResult(
        m=m,
        n=n,
        k=k,
        thread_k=thread_k,
        thread_n=thread_n,
        sms=sms,
        seconds_per_iter=seconds,
        tflops=_tflops(m, n, k, seconds),
        gbps=_gbps_marlin(m, n, k, seconds),
        repeat_id=repeat_id,
        mode=mode,
    )


def _run_dense_baseline(
    problems: Problems,
    iters: int,
    warmup: int,
) -> Dict[str, float]:
    """
    Optional: dense FP16 baseline for speedup reporting.
    """
    num_pairs = len(problems.As)
    for c in problems.Cs:
        c.zero_()
    torch.cuda.synchronize()

    def _dense(i: int):
        idx = i % num_pairs
        torch.matmul(problems.As[idx], problems.Bs_fp16[idx], out=problems.Cs[idx])

    seconds = _benchmark(_dense, warmup=warmup, iters=iters)
    m, k = problems.As[0].shape
    n = problems.Cs[0].shape[1]
    return {
        "seconds_per_iter": seconds,
        "tflops": _tflops(m, n, k, seconds),
        "gbps": _gbps_dense(m, n, k, seconds),
    }


def tune_marlin(
    m: int,
    n: int,
    k: int,
    iters: int = 300,
    warmup: int = 20,
    num_pairs: Optional[int] = None,
    l2_cache_mb: Optional[float] = None,
    repeats: int = 2,
    do_dense_baseline: bool = True,
    tune_threads: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    Auto-tune for one (m,n,k).
    Returns:
      - best config
      - all trial results
      - optional dense baseline
    """
    if device is None:
        device = torch.device("cuda:0")

    props = torch.cuda.get_device_properties(device)
    sm_hw = props.multi_processor_count

    if num_pairs is None:
        if l2_cache_mb is None:
            raise ValueError("Either num_pairs or l2_cache_mb must be provided.")

        l2_bytes = int(l2_cache_mb * 1024 * 1024)
        num_pairs = choose_num_pairs_from_l2(k, n, l2_bytes, min_pairs=2)

    problems = make_problems(m, n, k, num_pairs=num_pairs, device=device)

    # Baseline (optional)
    dense = None
    if do_dense_baseline:
        dense = _run_dense_baseline(problems, iters=iters, warmup=warmup)

    all_trials: List[TrialResult] = []

    best_time = float("inf")
    best_cfg = None

    # -------- SM sweep (multiples of 16 plus hardware count) --------
    sms_list = _sms_candidates(sm_hw)

    thread_configs = THREAD_CHOICES if tune_threads else [(-1, -1)]

    for (tk, tn) in thread_configs:
        for sms in sms_list:
            for r in range(repeats):
                tr = _run_marlin_trial(
                    problems,
                    thread_k=tk,
                    thread_n=tn,
                    sms=sms,
                    iters=iters,
                    warmup=warmup,
                    repeat_id=r,
                    mode="sweep",
                )
                all_trials.append(tr)

    # Final best across everything (min time)
    for t in all_trials:
        if t.seconds_per_iter < best_time:
            best_time = t.seconds_per_iter
            best_cfg = t

    assert best_cfg is not None

    best = asdict(best_cfg)
    best["speedup_over_dense"] = None
    if dense is not None:
        best["speedup_over_dense"] = dense["seconds_per_iter"] / best_cfg.seconds_per_iter

    result = {
        "shape": {"m": m, "n": n, "k": k},
        "gpu": {
            "name": props.name,
            "sm_count": sm_hw,
            "total_memory_gb": props.total_memory / (1024**3),
        },
        "dense_baseline": dense,
        "best": best,
        "trials": [asdict(t) for t in all_trials],
    }
    return result


# -----------------------------
# Reporting
# -----------------------------
def write_csv_all_trials(path: str, all_results: List[Dict[str, object]]) -> None:
    """
    One row per trial.
    """
    rows = []
    for res in all_results:
        shape = res["shape"]
        gpu = res["gpu"]
        for t in res["trials"]:
            row = dict(t)
            row.update(
                {
                    "gpu_name": gpu["name"],
                    "gpu_sm_count": gpu["sm_count"],
                    "gpu_mem_gb": gpu["total_memory_gb"],
                    "m": shape["m"],
                    "n": shape["n"],
                    "k": shape["k"],
                }
            )
            rows.append(row)

    # stable header
    fieldnames = [
        "gpu_name",
        "gpu_sm_count",
        "gpu_mem_gb",
        "m",
        "n",
        "k",
        "thread_k",
        "thread_n",
        "sms",
        "seconds_per_iter",
        "tflops",
        "gbps",
        "repeat_id",
        "mode",
    ]
    # create folder if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_csv_best(path: str, all_results: List[Dict[str, object]]) -> None:
    """
    One row per shape: best config (+ optional speedup).
    """
    rows = []
    for res in all_results:
        best = res["best"]
        shape = res["shape"]
        gpu = res["gpu"]
        dense = res.get("dense_baseline", None)

        row = {
            "gpu_name": gpu["name"],
            "gpu_sm_count": gpu["sm_count"],
            "gpu_mem_gb": gpu["total_memory_gb"],
            "m": shape["m"],
            "n": shape["n"],
            "k": shape["k"],
            "best_thread_k": best["thread_k"],
            "best_thread_n": best["thread_n"],
            "best_sms": best["sms"],
            "best_seconds_per_iter": best["seconds_per_iter"],
            "best_tflops": best["tflops"],
            "best_gbps": best["gbps"],
            "speedup_over_dense": best.get("speedup_over_dense", None),
        }

        if dense is not None:
            row["dense_seconds_per_iter"] = dense["seconds_per_iter"]
            row["dense_tflops"] = dense["tflops"]
            row["dense_gbps"] = dense["gbps"]
        else:
            row["dense_seconds_per_iter"] = None
            row["dense_tflops"] = None
            row["dense_gbps"] = None

        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    # create folder if not exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# -----------------------------
# Main
# -----------------------------
def main():
    torch.set_grad_enabled(False)

    # pair of (in_features, out_features)
    unique_shapes = {(4096, 12288), (3072, 3072), (4096, 1024), (2560, 1024), (12288, 4096), (2560, 9728), 
                     (5120, 7168), (2048, 8192), (4096, 4096), (2560, 4096), (17408, 5120), (5120, 10240), 
                     (5120, 27648), (4096, 24576), (2048, 3072), (3072, 1024), (4096, 22016), (5120, 5120), 
                     (5120, 51200), (5120, 8192), (2048, 512), (8192, 2048), (2560, 19456), (5120, 25600), 
                     (4096, 11008), (8192, 5120), (13824, 5120), (4096, 2560), (9728, 2560), (5120, 17408), 
                     (14336, 4096), (4096, 28672), (3072, 16384), (4096, 14336), (5120, 34816), (11008, 4096), 
                     (8192, 3072), (3072, 5120), (2560, 6144), (3072, 8192), (4096, 6144), (5120, 15360), 
                     (25600, 5120), (5120, 1024), (2048, 2048), (2048, 16384)}
    # unique_shapes = {(2048, 512)}
    # You can edit this list directly.
    # Format: (M, N, K)
    SHAPES: List[Tuple[int, int, int]] = []
    for (k, n) in unique_shapes:
        # Typical M values for transformer models
        for m in [1, 2, 4, 8, 16, 32, 64, 128]:
            SHAPES.append((m, n, k))


    # Tuning parameters
    ITERS = 1000
    WARMUP = 100
    L2_CACHE_MB = 96.0
    REPEATS = 3

    results: List[Dict[str, object]] = []

    gpu = torch.cuda.get_device_properties(0)
    tqdm.write(f"[INFO] GPU: {gpu.name}, SMs={gpu.multi_processor_count}, Mem={gpu.total_memory/(1024**3):.2f} GB")
    tqdm.write(f"[INFO] Tuning {len(SHAPES)} shapes...")

    for (m, n, k) in tqdm(SHAPES, desc="Tuning shapes"):
        tqdm.write(f"\n=== Tuning shape M={m}, N={n}, K={k} ===")
        res = tune_marlin(
            m=m,
            n=n,
            k=k,
            iters=ITERS,
            warmup=WARMUP,
            l2_cache_mb=L2_CACHE_MB,
            repeats=REPEATS,
            do_dense_baseline=True,
            tune_threads=False,
            device=torch.device("cuda:0"),
        )
        best = res["best"]
        dense = res["dense_baseline"]

        tqdm.write(
            f"[BEST] (thread_k,thread_n)=({best['thread_k']},{best['thread_n']}), "
            f"sms={best['sms']}, "
            f"{best['seconds_per_iter']*1e6:.2f} us/iter, "
            f"{best['tflops']:.2f} TFLOP/s, {best['gbps']:.2f} GB/s"
        )
        if dense is not None and best.get("speedup_over_dense", None) is not None:
            tqdm.write(
                f"[DENSE] {dense['seconds_per_iter']*1e6:.2f} us/iter, "
                f"speedup={best['speedup_over_dense']:.3f}x"
            )

        results.append(res)

        # Small cool-down between shapes (optional)
        time.sleep(0.5)

    # Output files
    out_all = "autotune/marlin_tuning_all_results.csv"
    out_best = "autotune/marlin_tuning_best_configs.csv"
    out_json = "autotune/marlin_tuning_summary.json"
    default_launch_path = "marlin_razer/launch_config.py"

    write_csv_all_trials(out_all, results)
    write_csv_best(out_best, results)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    tqdm.write("\n[INFO] Wrote:")
    tqdm.write(f"  - {out_all}")
    tqdm.write(f"  - {out_best}")
    tqdm.write(f"  - {out_json}")

    # Optional launch config generation
    resp = input(f"Generate launch config at {default_launch_path}? [y/N]: ").strip().lower()
    if resp in ("y", "yes"):
        try:
            generate_launch_config(out_best, default_launch_path)
        except Exception as e:
            tqdm.write(f"[WARN] Failed to write launch config: {e}")
    else:
        tqdm.write("[INFO] Skipping launch config generation.")


if __name__ == "__main__":
    main()
