import importlib

import torch
import razer_ext


_OPTIONAL_MODULE_CACHE = {}
_OPTIONAL_IMPORT_ERRS = {}


def _get_optional_module(name: str):
    if name not in _OPTIONAL_MODULE_CACHE:
        try:
            _OPTIONAL_MODULE_CACHE[name] = importlib.import_module(name)
            _OPTIONAL_IMPORT_ERRS[name] = None
        except Exception as exc:
            _OPTIONAL_MODULE_CACHE[name] = None
            _OPTIONAL_IMPORT_ERRS[name] = exc
    return _OPTIONAL_MODULE_CACHE[name], _OPTIONAL_IMPORT_ERRS[name]


def _require_module(name: str, label: str):
    mod, err = _get_optional_module(name)
    if mod is None:
        detail = f": {err}" if err is not None else ""
        raise RuntimeError(
            f"{label} extension is not available{detail}. "
            f"Install it from inference/kernel_extensions/{label}."
        )
    return mod


@torch.library.custom_op("plugin::marlin_razer_mul", mutates_args={"output", "workspace"})
def marlin_razer_mul(
    x: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    s: torch.Tensor,
    workspace: torch.Tensor,
    thread_k: int = -1,
    thread_n: int = -1,
    sms: int = -1,
    max_par: int = 16,
) -> None:
    mod = _require_module("marlin_razer", "marlin_razer")
    return mod.mul(x, B, output, s, workspace, thread_k, thread_n, sms, max_par)


@marlin_razer_mul.register_fake
def _(x, B, output, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    return None


@torch.library.custom_op("plugin::marlin_fp4_mul", mutates_args={"output", "workspace"})
def marlin_fp4_mul(
    x: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    s: torch.Tensor,
    workspace: torch.Tensor,
    thread_k: int = -1,
    thread_n: int = -1,
    sms: int = -1,
    max_par: int = 16,
) -> None:
    mod = _require_module("marlin_fp4", "marlin_fp4")
    return mod.mul(x, B, output, s, workspace, thread_k, thread_n, sms, max_par)


@marlin_fp4_mul.register_fake
def _(x, B, output, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    return None


@torch.library.custom_op("plugin::razer_gemm", mutates_args={"out"})
def razer_gemm(
    fA: torch.Tensor,
    qB: torch.Tensor,
    scaling_factors: torch.Tensor,
    out: torch.Tensor,
    groupsize: int = 128,
    impl: int = 0,
    split_k: int = -1,
    gemv_g: int = -1,
) -> None:
    razer_ext.razer_gemm(fA, qB, scaling_factors, out, groupsize, impl, split_k, gemv_g)
    return None


@razer_gemm.register_fake
def _(fA, qB, scaling_factors, out, groupsize=128, impl=0, split_k=-1, gemv_g=-1):
    return None


@torch.library.custom_op("plugin::razer_dequant", mutates_args={})
def razer_dequant(
    qB: torch.Tensor,
    scaling_factors: torch.Tensor,
    in_features: int,
    groupsize: int = 128,
) -> torch.Tensor:
    return razer_ext.razer_dequant(qB, scaling_factors, in_features, groupsize)


@razer_dequant.register_fake
def _(qB, scaling_factors, in_features, groupsize=128):
    k = int(in_features)
    n = qB.shape[1]
    return torch.empty((k, n), dtype=torch.float16, device=qB.device)
