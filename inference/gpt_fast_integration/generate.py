# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

create_block_mask = torch.compile(create_block_mask)

# support running without installing as a package
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer
import custom_linears


def _is_linear_like_in_layers(mod: nn.Module) -> bool:
    return isinstance(
        mod,
        (
            nn.Linear,
            custom_linears.RaZeRLinear,
            custom_linears.MarlinRazerLinear,
            custom_linears.MarlinFP4Linear,
        ),
    )


def replace_layer_linears_with_noop(model: nn.Module) -> int:
    """Replace only linear-like modules inside Transformer.layers.* with NoOpLinear."""

    layers = getattr(model, "layers", None)
    if layers is None:
        raise AttributeError("Model has no attribute 'layers'; expected a Transformer-like module")

    replaced = 0

    def _recurse(module: nn.Module):
        nonlocal replaced
        for name, child in module.named_children():
            if _is_linear_like_in_layers(child):
                in_features = getattr(child, "in_features")
                out_features = getattr(child, "out_features")
                setattr(module, name, custom_linears.NoOpLinear(in_features, out_features))
                replaced += 1
            else:
                _recurse(child)

    _recurse(layers)
    return replaced

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, block_mask, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens

    max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:, T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision):
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "marlinrazer" in str(checkpoint_path):
        print("Using Marlin-RaZeR weight-only quantization!")
        from quantize import WeightOnlyMarlinRaZeRQuantHandler
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-3][1:])
        fp_dtype = path_comps[-2]
        simple_quantizer = WeightOnlyMarlinRaZeRQuantHandler(model, groupsize, fp_dtype)
        model = simple_quantizer.convert_for_runtime()
    elif "marlinfp4" in str(checkpoint_path):
        print("Using Marlin-FP4 weight-only quantization!")
        from quantize import WeightOnlyMarlinFP4QuantHandler
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-3][1:])
        fp_dtype = path_comps[-2]
        simple_quantizer = WeightOnlyMarlinFP4QuantHandler(model, groupsize, fp_dtype)
        model = simple_quantizer.convert_for_runtime()
    elif ("razer" in checkpoint_path.name) or ("razer4" in checkpoint_path.name):
        print("Using RaZeR 4-bit canonical weight-only quantization!")
        from quantize import WeightOnlyRaZeRQuantHandler
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-3][1:])
        fp_dtype = path_comps[-2]
        simple_quantizer = WeightOnlyRaZeRQuantHandler(model, 4, groupsize, fp_dtype)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    device=default_device,
    stream: bool = False,
    fp_dtype=None,
    noop_linears: bool = False,
    razer_matmul: Optional[str] = None,
    sqllm_matmul: Optional[str] = None,
    awq_matmul: Optional[str] = None,
    quiet: bool = False,
    return_metrics: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    if not tokenizer_path.is_file():
        tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    global print
    rank = 0
    use_tp = False
    if quiet:
        print = lambda *args, **kwargs: None

    # Configure RaZeRLinear behavior for this run.
    # This is a global knob in custom_linears so we don't need to thread kwargs
    # through model.forward().
    if razer_matmul is not None:
        custom_linears.set_razer_matmul_mode(razer_matmul)

    print(f"Using device={device}")
    if fp_dtype is not None:
        if fp_dtype == 'fp16':
            precision = torch.float16
        elif fp_dtype == 'bf16':
            precision = torch.bfloat16
        elif fp_dtype == 'fp32':
            precision = torch.float32
        else:
            raise ValueError(f"Unknown fp_dtype {fp_dtype}")
        print(f"Using precision={precision} from command line argument")
    else:
        if '.fp16.' in str(checkpoint_path):
            precision = torch.float16
        elif '.fp32.' in str(checkpoint_path):
            precision = torch.float32
        elif '.bf16.' in str(checkpoint_path):
            precision = torch.bfloat16
        else:
            print("Model precision not specified in checkpoint name, assuming bfloat16")
            precision = torch.bfloat16
        print(f"Using precision={precision}")
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision)
    else:
        draft_model = None

    if noop_linears:
        n = replace_layer_linears_with_noop(model)
        if draft_model is not None and draft_model is not model:
            n_draft = replace_layer_linears_with_noop(draft_model)
        else:
            n_draft = 0
        print(f"No-op linear mode enabled: replaced {n} layer linears" + (f" (+{n_draft} in draft)" if n_draft else ""))

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    if prompt is None:
        encoded = torch.tensor([tokenizer.bos_id()], dtype=torch.int, device=device)
    elif isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # generate a fully synthetic prompt
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device) # MKG
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if stream or interactive and i >= 0:
            # Interactive streaming assumes batch_size == 1 for printing
            assert batch_size == 1, "interactive streaming currently supports batch_size == 1"
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                # x is a 1D tensor of shape [B]; stream the first element
                token_id = x.view(-1)[0].item()
                text = tokenizer.decode([period_id, token_id])[1:]
                print(text, end='', flush=True)
                if token_id == tokenizer.eos_id():
                    done_generating = True
        else:
            callback = lambda x : x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y, metrics = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        device_sync(device=device) # MKG
        t = time.perf_counter() - t0

        if not interactive:
            # Just displaying one random sample from the batch
            if batch_size > 1:
                print("Only displaying one generation from the batch...")
            print(tokenizer.decode(y[torch.randint(0, batch_size, (1,)).item()].tolist()))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    if return_metrics:
        # Provide a lightweight dict for benchmark scripts.
        tps = [float(x) for x in aggregate_metrics['tokens_per_sec']]
        avg = float(sum(tps) / len(tps)) if tps else 0.0
        if len(tps) >= 3:
            trimmed = sorted(tps)[1:-1]
            avg_trimmed = float(sum(trimmed) / len(trimmed))
        else:
            avg_trimmed = avg
        return {
            "tokens_per_sec": tps,
            "avg_tokens_per_sec": avg,
            "avg_tokens_per_sec_trimmed": avg_trimmed,
            "prompt_length": int(prompt_length),
            "max_new_tokens": int(max_new_tokens),
            "batch_size": int(batch_size),
            "checkpoint_path": str(checkpoint_path),
            "noop_linears": bool(noop_linears),
        }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run generation for the public RaZeR inference artifact.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=str, default=None, help="Input prompt. If it is not given, a single BOS token is fed as input prompt.")
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')
    parser.add_argument('--stream', action='store_true', help='Whether to stream the output token-by-token')
    parser.add_argument('--fp_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='Floating point dtype to use')
    parser.add_argument('--noop_linears', action='store_true', help='Replace Transformer.layers.* linears with a no-op (shape-preserving zeros) to benchmark upper-bound tok/s without linear compute')
    parser.add_argument(
        '--razer_matmul',
        type=str,
        default=None,
        choices=['auto', 'kernel', 'dequant_gemm'],
        help='RaZeRLinear matmul path: auto (heuristic), kernel (force kernel for M<=128), dequant_gemm (force dequant+GEMM).',
    )
    parser.add_argument('--quiet', action='store_true', help='Reduce printing (useful for scripts)')
    args = parser.parse_args()
    main(
        prompt=args.prompt,
        interactive=args.interactive,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        compile_prefill=args.compile_prefill,
        profile=args.profile,
        draft_checkpoint_path=args.draft_checkpoint_path,
        speculate_k=args.speculate_k,
        device=args.device,
        stream=args.stream,
        fp_dtype=args.fp_dtype,
        noop_linears=args.noop_linears,
        razer_matmul=args.razer_matmul,
        quiet=args.quiet,
        return_metrics=False,
    )
