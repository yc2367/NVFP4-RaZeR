import numpy as np
import torch
import marlin_razer
import time


def benchmark(f, warmup=10, iter=100):
    """
    Benchmark helper.
    If f accepts an integer argument, we pass the loop index i so the caller can
    cycle through multiple matrix pairs; otherwise we call f() for backward compatibility.
    """
    for i in range(warmup + iter):
        try:
            f(i)
        except TypeError:
            print("TypeError caught, calling f() without arguments")
            f()

        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()

    torch.cuda.synchronize()
    res = (time.time() - tick) / iter # type: ignore

    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.0)
    return res


def get_problem(m, n, k):
    # print(f'get_problem: m: {m}, n: {n}, k: {k}')
    groupsize = 128
    dev = torch.device('cuda:0')

    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B_fp16 = torch.randn((k, n), dtype=torch.half, device=dev)

    # Marlin expects packed int32 weights
    B_marlin = torch.randint(low=-2**31, high=2**31, size=(k * n // 8,), device=dev)
    scaling_marlin = torch.rand((k // groupsize, n), dtype=torch.half, device=dev)

    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B_fp16, B_marlin, scaling_marlin, C


def benchmark_dense(As, Bs_fp16, Cs, iter=100):
    num_pairs = len(As)
    res = benchmark(
        lambda i: torch.matmul(
            As[i % num_pairs],
            Bs_fp16[i % num_pairs],
            out=Cs[i % num_pairs]
        ),
        iter=iter
    )

    # Report throughput for one representative problem (all pairs are same shape)
    A0, B0, C0 = As[0], Bs_fp16[0], Cs[0]
    return {
        's': res,
        'TFLOP/s': 2 * A0.numel() * C0.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A0.numel() + 2 * B0.numel() + 2 * C0.numel()) / res / 10 ** 9
    }


def benchmark_marlin(As, Bs_marlin, Cs, scalings_marlin, thread_k, thread_n, sms, iter=100):
    num_pairs = len(As)

    # One workspace is fine (kernel is launched sequentially here)
    workspace = torch.zeros(Cs[0].shape[1] // 128 * 16, device=torch.device('cuda:0'))

    res = benchmark(
        lambda i: marlin_razer.mul(
            As[i % num_pairs],
            Bs_marlin[i % num_pairs],
            Cs[i % num_pairs],
            scalings_marlin[i % num_pairs],
            workspace,
            thread_k,
            thread_n,
            sms
        ),
        iter=iter
    )

    # Report throughput for one representative problem
    A0, B0, C0, S0 = As[0], Bs_marlin[0], Cs[0], scalings_marlin[0]
    return {
        's': res,
        'TFLOP/s': 2 * A0.numel() * C0.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A0.numel() + 4 * B0.numel() + 2 * C0.numel() + 2 * S0.numel()) / res / 10 ** 9
    }


def marlin_only(M=1, N=8192, K=8192, ITER=500, NUM_PAIRS=32, print_output=True):

    # Build multiple independent matrix pairs
    
        
    problems = [get_problem(M, N, K) for _ in range(NUM_PAIRS)]
    As        = [p[0] for p in problems]
    Bs_fp16   = [p[1] for p in problems]
    Bs_marlin = [p[2] for p in problems]
    scalings  = [p[3] for p in problems]
    Cs        = [p[4] for p in problems]
    SMS = torch.cuda.get_device_properties(0).multi_processor_count
    if print_output:
        print(f'Prepared {NUM_PAIRS} problem pairs of size M={M}, N={N}, K={K}...')
    
    result_dense = benchmark_dense(As, Bs_fp16, Cs, iter=ITER)
    

    result_marlin = benchmark_marlin(As, Bs_marlin, Cs, scalings, -1, -1, SMS, iter=ITER)
    
    if print_output:
        print(f"Dense FP16: {result_dense}")
        print(f"Marlin: {result_marlin}")
        print(f"Speedup: {result_dense['s'] / result_marlin['s']:.3f}x")

    return result_dense, result_marlin


def plot_speedup(Ms, speedups, extra_information=''):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7,5.5))
    plt.plot(Ms, speedups, marker='o')
    # annotate each point with its speedup value
    for i, speedup in enumerate(speedups):
        plt.annotate(f'{speedup:.3f}', (Ms[i], speedups[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.xscale('log', base=2)
    plt.xlabel('M dimension')
    plt.ylabel('Speedup')
    plt.title('Marlin Speedup over Dense FP16 vs M Dimension')
    plt.grid(True, which="both", ls="--")
    plt.ylim(0,5)
    # add extra information as text box
    plt.gcf().text(0.02, 0.02, extra_information, fontsize=8, va='bottom', ha='left', wrap=True)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.savefig('marlin_speedup_vs_M.png', dpi=300, bbox_inches='tight')
    # plt.show()

def main_sweep():
    SMS = torch.cuda.get_device_properties(0).multi_processor_count
    Ms = [1,2,4,8,16,32,64,128, 256, 512]
    N = 256 * SMS
    K = 256 * SMS
    # N = 4096
    # K = 4096
    ITERS = 500
    num_pairs = 4
    marlin_results = []
    dense_results = []    
    for m in Ms:
        dense_result, marlin_result = marlin_only(M=m, N=N, K=K, ITER=ITERS, NUM_PAIRS=num_pairs, print_output=True)
        dense_results.append(dense_result)
        marlin_results.append(marlin_result)
    speedups = [d['s']/m['s'] for d,m in zip(dense_results, marlin_results)]
    
    # build extra information string
    extra_info = f'N={N}, K={K}, Iter={ITERS}, NumPairs={num_pairs}'
    extra_info += f'\nDense TFLOP/s: {[round(d["TFLOP/s"], 3) for d in dense_results]}'
    extra_info += f'\nDense GB/s: {[round(d["GB/s"], 3) for d in dense_results]}'
    extra_info += f'\nMarlin TFLOP/s: {[round(m["TFLOP/s"], 3) for m in marlin_results]}'
    extra_info += f'\nMarlin GB/s: {[round(m["GB/s"], 3) for m in marlin_results]}'
    # add gpu info
    gpu_info = torch.cuda.get_device_properties(0)
    extra_info += f'\nGPU: {gpu_info.name}, SMs: {gpu_info.multi_processor_count}'
    # get gpu memory
    gpu_memory = gpu_info.total_memory / (1024 ** 3)
    extra_info += f'\nGPU Memory: {gpu_memory:.2f} GB'
   
    
    plot_speedup(Ms, speedups, extra_information=extra_info)
    
    


if __name__ == '__main__':
    # marlin_only(1)
    main_sweep()
