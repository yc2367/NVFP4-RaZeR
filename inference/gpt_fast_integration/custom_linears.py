import importlib

import torch
import torch.nn as nn

import kernels


_OPTIONAL_MODULE_CACHE = {}


def _get_optional_module(name: str):
    if name not in _OPTIONAL_MODULE_CACHE:
        try:
            _OPTIONAL_MODULE_CACHE[name] = importlib.import_module(name)
        except Exception:
            _OPTIONAL_MODULE_CACHE[name] = None
    return _OPTIONAL_MODULE_CACHE[name]


_RAZER_MATMUL_MODE: str = "auto"


def _normalize_razer_matmul_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    aliases = {
        "auto": "auto",
        "kernel": "kernel",
        "dequant_gemm": "dequant_gemm",
    }
    if mode not in aliases:
        raise ValueError(
            f"Unknown razer matmul mode: {mode!r} (expected one of {sorted(aliases.keys())})"
        )
    return aliases[mode]


def set_razer_matmul_mode(mode: str) -> None:
    global _RAZER_MATMUL_MODE
    if mode is None:
        mode = "auto"
    _RAZER_MATMUL_MODE = _normalize_razer_matmul_mode(mode)


def get_razer_matmul_mode() -> str:
    return _RAZER_MATMUL_MODE


class NoOpLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.output = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")
        orig_shape = x.shape[:-1]
        n = x.numel() // self.in_features
        expected = (n, self.out_features)
        if (
            self.output is None
            or self.output.shape != expected
            or self.output.device != x.device
            or self.output.dtype != x.dtype
        ):
            self.output = torch.zeros(expected, device=x.device, dtype=x.dtype)
        return self.output.view(*orig_shape, self.out_features)


class RaZeRLinear(nn.Module):
    def __init__(self, in_features, out_features, bits, groupsize, fp_dtype, bias=False, padding: bool = True):
        super().__init__()
        if in_features % 128 != 0:
            raise ValueError(f"in_features {in_features} must be a multiple of 128 for RaZeR")

        def _find_multiple(n: int, k: int) -> int:
            return n if (n % k) == 0 else n + k - (n % k)

        self.origin_in_features = in_features
        if padding:
            in_features = _find_multiple(in_features, 128)

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.groupsize = groupsize
        self.fp_dtype = fp_dtype
        self.padding = padding

        if groupsize not in (16, 128):
            raise ValueError("RaZeR only supports groupsize 16 or groupsize 128.")
        if bits != 4:
            raise ValueError("Only 4-bit RaZeR is currently supported.")
        if fp_dtype != torch.float16:
            raise ValueError("RaZeR currently only supports fp16")

        self.register_buffer(
            "razer_qB",
            torch.empty((in_features // 8, out_features), dtype=torch.int32, device="cuda"),
            persistent=True,
        )
        if groupsize == 128:
            scale_shape = (in_features // 128, out_features)
            scale_dtype = self.fp_dtype
        else:
            scale_shape = (in_features // 128, out_features, 2)
            scale_dtype = torch.int32
        self.register_buffer(
            "razer_scaling",
            torch.empty(scale_shape, dtype=scale_dtype, device="cuda"),
            persistent=True,
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=self.fp_dtype, device="cuda"), persistent=True)
        else:
            self.bias = None
        self.output = torch.zeros((1, out_features), dtype=self.fp_dtype, device="cuda")

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs):
        if x.device != self.razer_qB.device:
            raise RuntimeError("Input and buffers must be on the same CUDA device.")
        if x.dtype != self.fp_dtype:
            raise TypeError(f"Input dtype {x.dtype} != layer dtype {self.fp_dtype}")
        if self.padding:
            if x.shape[-1] != self.origin_in_features:
                raise ValueError(f"Expected last dim {self.origin_in_features}, got {x.shape[-1]}")
        else:
            if x.shape[-1] != self.in_features:
                raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        batch_size, seq_len = x.shape[0], x.shape[1]
        expected_output_shape = (batch_size * seq_len, self.out_features)
        if self.output.shape != expected_output_shape:
            self.output = torch.empty(expected_output_shape, dtype=self.fp_dtype, device=x.device)

        x = x.view(-1, x.shape[-1])
        m = int(x.shape[0])

        mode = kwargs.get("razer_matmul") or kwargs.get("razer_matmul_mode")
        if mode is None:
            mode = get_razer_matmul_mode()
        mode = _normalize_razer_matmul_mode(mode)

        if self.padding and self.in_features != self.origin_in_features:
            x = torch.nn.functional.pad(x, pad=(0, self.in_features - self.origin_in_features))

        force_dequant = (
            (m > 128)
            or (mode == "dequant_gemm")
            or (mode == "auto" and m > 16)
            or (self.groupsize == 16 and m > 1)
        )

        if not force_dequant and m == 1:
            kernels.razer_gemm(x, self.razer_qB, self.razer_scaling, self.output, self.groupsize, 1, -1, -1)
        elif not force_dequant and m <= 128:
            kernels.razer_gemm(x, self.razer_qB, self.razer_scaling, self.output, self.groupsize, 2, -1, -1)
        else:
            b_dense = kernels.razer_dequant(self.razer_qB, self.razer_scaling, self.in_features, self.groupsize)
            torch.addmm(self.output, x, b_dense, beta=0.0, alpha=1.0, out=self.output)

        self.output = self.output.view(batch_size, seq_len, self.out_features)
        if self.bias is not None:
            self.output += self.bias
        return self.output


class MarlinRazerLinear(nn.Module):
    def __init__(self, in_features, out_features, groupsize, fp_dtype, bias=False):
        super().__init__()
        if in_features % 128 != 0:
            raise ValueError(f"in_features {in_features} must be divisible by 128 for Marlin-RaZeR")
        if out_features % 256 != 0:
            raise ValueError(f"out_features {out_features} must be divisible by 256 for Marlin-RaZeR")
        if groupsize not in [-1, 128]:
            raise ValueError(f"groupsize {groupsize} must be -1 (per-channel) or 128 for Marlin-RaZeR")
        if fp_dtype != torch.float16:
            raise ValueError("Marlin-RaZeR currently only supports fp16")

        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize if groupsize != -1 else in_features
        self.fp_dtype = fp_dtype

        self.register_buffer(
            "marlinrazer_B",
            torch.empty((in_features // 16, out_features * 16 // 8), dtype=torch.int32),
        )
        self.register_buffer(
            "marlinrazer_s",
            torch.empty((in_features // self.groupsize, out_features), dtype=torch.float16),
        )
        self.register_buffer(
            "workspace",
            torch.zeros(out_features // 128 * 16, dtype=torch.int32),
            persistent=False,
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self.output = torch.zeros((1, out_features), dtype=self.fp_dtype, device="cuda")
        self._m_config_table = {}
        self.SMS = -1
        self._cfg_device = None
        self._rebuild_launch_config_table()

    def _rebuild_launch_config_table(self):
        dev = self.marlinrazer_B.device
        self._cfg_device = dev
        if dev.type != "cuda":
            self._m_config_table = {}
            self.SMS = -1
            return
        props = torch.cuda.get_device_properties(dev.index)
        self.SMS = props.multi_processor_count
        mod = _get_optional_module("marlin_razer")
        tuning = getattr(mod, "MARLIN_RAZER_TUNING", None) if mod is not None else None
        dev_tbl = tuning.get(props.name) if isinstance(tuning, dict) else None
        m_table = {}
        if dev_tbl is not None:
            for (m0, k0, n0), cfg in dev_tbl.items():
                if k0 == self.in_features and n0 == self.out_features:
                    m_table[m0] = cfg
        self._m_config_table = m_table

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._rebuild_launch_config_table()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs):
        input_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        if self.output.shape[0] != x_flat.shape[0] or self.output.device != x.device:
            self.output = torch.zeros((x_flat.shape[0], self.out_features), dtype=self.fp_dtype, device=x.device)
        else:
            self.output.zero_()
        thread_k, thread_n, sms = self._m_config_table.get(x_flat.shape[0], (-1, -1, self.SMS))
        kernels.marlin_razer_mul(
            x_flat,
            self.marlinrazer_B,
            self.output,
            self.marlinrazer_s,
            self.workspace,
            thread_k,
            thread_n,
            sms,
            16,
        )
        if self.bias is not None:
            self.output += self.bias
        return self.output.view(input_shape[:-1] + (self.out_features,))


class MarlinFP4Linear(nn.Module):
    def __init__(self, in_features, out_features, groupsize, fp_dtype, bias=False):
        super().__init__()
        if in_features % 128 != 0:
            raise ValueError(f"in_features {in_features} must be divisible by 128 for Marlin-FP4")
        if out_features % 256 != 0:
            raise ValueError(f"out_features {out_features} must be divisible by 256 for Marlin-FP4")
        if groupsize not in [-1, 128]:
            raise ValueError(f"groupsize {groupsize} must be -1 (per-channel) or 128 for Marlin-FP4")
        if fp_dtype != torch.float16:
            raise ValueError("Marlin-FP4 currently only supports fp16")

        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize if groupsize != -1 else in_features
        self.fp_dtype = fp_dtype

        self.register_buffer(
            "marlinfp4_B",
            torch.empty((in_features // 16, out_features * 16 // 8), dtype=torch.int32),
        )
        self.register_buffer(
            "marlinfp4_s",
            torch.empty((in_features // self.groupsize, out_features), dtype=torch.float16),
        )
        self.register_buffer(
            "workspace",
            torch.zeros(out_features // 128 * 16, dtype=torch.int32),
            persistent=False,
        )
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

        self.output = torch.zeros((1, out_features), dtype=self.fp_dtype, device="cuda")
        self._m_config_table = {}
        self.SMS = -1
        self._cfg_device = None
        self._rebuild_launch_config_table()

    def _rebuild_launch_config_table(self):
        dev = self.marlinfp4_B.device
        self._cfg_device = dev
        if dev.type != "cuda":
            self._m_config_table = {}
            self.SMS = -1
            return
        props = torch.cuda.get_device_properties(dev.index)
        self.SMS = props.multi_processor_count
        mod = _get_optional_module("marlin_fp4")
        tuning = getattr(mod, "MARLIN_FP4_TUNING", None) if mod is not None else None
        dev_tbl = tuning.get(props.name) if isinstance(tuning, dict) else None
        m_table = {}
        if dev_tbl is not None:
            for (m0, k0, n0), cfg in dev_tbl.items():
                if k0 == self.in_features and n0 == self.out_features:
                    m_table[m0] = cfg
        self._m_config_table = m_table

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self._rebuild_launch_config_table()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, **kwargs):
        input_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        if self.output.shape[0] != x_flat.shape[0] or self.output.device != x.device:
            self.output = torch.zeros((x_flat.shape[0], self.out_features), dtype=self.fp_dtype, device=x.device)
        else:
            self.output.zero_()
        thread_k, thread_n, sms = self._m_config_table.get(x_flat.shape[0], (-1, -1, self.SMS))
        kernels.marlin_fp4_mul(
            x_flat,
            self.marlinfp4_B,
            self.output,
            self.marlinfp4_s,
            self.workspace,
            thread_k,
            thread_n,
            sms,
            16,
        )
        if self.bias is not None:
            self.output += self.bias
        return self.output.view(input_shape[:-1] + (self.out_features,))
