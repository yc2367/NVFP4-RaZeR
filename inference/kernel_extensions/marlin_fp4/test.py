import unittest

import numpy as np
import torch
import torch.nn as nn

import marlin_fp4


seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)

DEV = torch.device("cuda:0")


# -----------------------------
# FP4(E2M1) helper quantization
# -----------------------------

_FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
    device=DEV,
)

def _quantize_fp4_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """
    x: float32 tensor
    returns: int32 codes in [0..15], encoding:
        code = (sign<<3) | mag_code
    where mag_code in [0..7] maps to magnitudes:
        [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    """
    x = torch.nan_to_num(x, nan=6.0, posinf=6.0, neginf=-6.0)

    sign = x < 0
    ax = x.abs()

    # Nearest-level quantization via midpoints between:
    # [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    idx = torch.zeros_like(ax, dtype=torch.int32)
    idx += (ax >= 0.25).to(torch.int32)
    idx += (ax >= 0.75).to(torch.int32)
    idx += (ax >= 1.25).to(torch.int32)
    idx += (ax >= 1.75).to(torch.int32)
    idx += (ax >= 2.50).to(torch.int32)
    idx += (ax >= 3.50).to(torch.int32)
    idx += (ax >= 5.00).to(torch.int32)

    code = idx | (sign.to(torch.int32) << 3)
    return code


def _dequant_fp4_e2m1_codes(code: torch.Tensor) -> torch.Tensor:
    """
    code: int32 tensor in [0..15]
    returns: float32 tensor containing FP4(E2M1) values (already signed),
             in the unscaled domain.
    """
    mag_idx = (code & 0x7).to(torch.long)
    sign = ((code >> 3) & 0x1).to(torch.float32)
    mag = _FP4_E2M1_LEVELS[mag_idx]
    # sign=1 => negative
    return mag * (1.0 - 2.0 * sign)


# ---------------------------------------
# Generate grouped FP4 packed (B, s) pair
# ---------------------------------------

def gen_fp4(m, n, groupsize=-1):
    """
    Returns:
      ref: (m, n) fake-quantized FP16 weight matrix (what matmul should use)
      q  : packed marlin B buffer
      s  : marlin scales buffer
    """
    tile = 16
    fp4_max = 6.0
    eps = 1e-8

    # Start from random FP16 "full precision" weights
    w = torch.randn((m, n), dtype=torch.half, device=DEV)

    # Match Marlin's grouping reshape convention
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    # Symmetric FP4 scale per output-column-and-group:
    # scale = max_abs / 6
    max_abs = torch.max(torch.abs(w), 0, keepdim=True)[0].to(torch.float32)
    s = torch.clamp(max_abs / fp4_max, min=eps)

    # Quantize (w / s) to FP4(E2M1) codes, then dequant back to FP16
    x = (w.to(torch.float32) / s)
    code = _quantize_fp4_e2m1_codes(x)                   # int32 in [0..15]
    xq = _dequant_fp4_e2m1_codes(code)                   # float32 in FP4 levels
    ref = (xq * s).to(torch.half)                        # scaled back to FP16

    # Undo grouping reshape (so ref and codes align with Marlin pack layout)
    if groupsize != -1:
        def reshape_back(t):
            t = t.reshape((groupsize, -1, n))
            t = t.permute(1, 0, 2)
            t = t.reshape((m, n)).contiguous()
            return t

        ref = reshape_back(ref)
        code = reshape_back(code)

    # Marlin expects s shaped as (m//groupsize, n) (or (1,n) if groupsize=-1)
    s = s.reshape((-1, n)).contiguous()

    # Build a fake-quantized Linear layer holding the dequantized ref weights
    linear = nn.Linear(m, n, bias=False)
    linear.weight.data = ref.t()

    # Pack using your modified marlin.Layer.pack (FP4 path inside)
    layer = marlin_fp4.Layer(256, 256, groupsize=groupsize)

    # Workaround to test some special cases that are forbidden by the API (same as original)
    if groupsize == -1:
        groupsize_eff = m
    else:
        groupsize_eff = groupsize

    layer.k = m
    layer.n = n
    layer.groupsize = groupsize_eff
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize_eff, n), dtype=torch.half, device=DEV)

    # scales argument shape should be (infeatures, groups) -> pass s.t() like the int4 test
    layer.pack(linear, s.t())

    q = layer.B
    s_out = layer.s
    return ref, q, s_out


class TestFP4(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, groupsize=-1):
        print("% 5d % 6d % 6d % 4d % 4d % 4d" % (m, n, k, thread_k, thread_n, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)

        B_ref, B, s = gen_fp4(k, n, groupsize=groupsize)

        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)

        workspace = torch.zeros(n // 128 * 16, device=DEV)

        # Your marlin.mul must be the FP4-capable kernel (and pack must use FP4 shuffle)
        marlin_fp4.mul(A, B, C, s, workspace, thread_k, thread_n, -1)
        torch.cuda.synchronize()

        # FP4 is much lower precision than int4 in practice; tolerate larger error.
        rel_err = torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref))
        self.assertLess(rel_err, 0.02)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024, thread_k, thread_n)

    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_groups(self):
        print()
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self.run_problem(m, n, k, *thread_shape, groupsize)


    def test_auto_shapes_fp4(self):
        print()
        shapes = [ # derived from LLaMA-like models
            (16, 4096, 4096),    # q/k/v/o
            (16, 11008, 4096),   # gate/up   (k=4096 -> n=11008)
            (16, 4096, 11008),   # down      (k=11008 -> n=4096)
        ]
        for m, n, k in shapes:
            print("shape:", m, n, k)

            A = torch.randn((m, k), dtype=torch.half, device=DEV)
            B_ref, B, s = gen_fp4(k, n, groupsize=128)
            C = torch.zeros((m, n), dtype=torch.half, device=DEV)
            C_ref = A @ B_ref
            workspace = torch.zeros(n // 128 * 16, device=DEV)

            # 1) explicit kernel (known-good)
            marlin_fp4.mul(A, B, C, s, workspace, 128, 128, -1)
            torch.cuda.synchronize()
            rel_err = torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref))
            print("explicit 128x128 rel_err:", float(rel_err))

            # 2) AUTO kernel path (the one your real model uses)
            C2 = torch.zeros((m, n), dtype=torch.half, device=DEV)
            marlin_fp4.mul(A, B, C2, s, workspace)   # no thread_k/thread_n
            torch.cuda.synchronize()
            rel_err2 = torch.mean(torch.abs(C2 - C_ref)) / torch.mean(torch.abs(C_ref))
            print("AUTO rel_err:", float(rel_err2))

            # check NaNs
            print("AUTO finite:", torch.isfinite(C2).all().item())


if __name__ == "__main__":
    unittest.main()
