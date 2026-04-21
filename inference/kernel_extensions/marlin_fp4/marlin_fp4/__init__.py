'''
Marlin-FP4: Fast 4-bit Linear Layers with Negative-Zero Remap Quantization matmul kernel.

Usage:
    1. Kernel mul (low-level):
        from marlin_fp4 import mul
        m, n, k = 16, 4096, 4096
        A = torch.randn((m, k), dtype=torch.half, device='cuda')
        B = torch.randint(0, 16, (k // 16, n * 16 // 8), dtype=torch.int32, device='cuda')  # pre-packed
        C = torch.zeros((m, n), dtype=torch.half, device='cuda')
        s = torch.randn((k // 128, n), dtype=torch.half, device='cuda')  # groupsize=128
        workspace = torch.zeros(n // 128 * 16, dtype=torch.int32, device='cuda')
        mul(A, B, C, s, workspace)
    
    2. Linear layer (high-level):
        from marlin_fp4 import Layer
        linear_fp16 = layer.self_attn.q_proj # derived from the model, type torch.nn.Linear
        marlin_layer = Layer(linear_fp16.in_features, linear_fp16.out_features, groupsize=128).cuda()
        marlin_layer.quick_quantize_fp4(linear_fp16) # quantize and pack weights
        output = marlin_layer(input)
        
    3. launch config table (MARLIN_FP4_TUNING):
        provied best launch configurations for various GPUs and problem sizes.
        format:
        MARLIN_FP4_TUNING = {
          'GPU Name': {
            (M, K, N): (thread_k, thread_n, sms),
            ...
          },
          ...
        }
'''


import torch
import torch.nn as nn


import marlin_fp4_cuda

def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xFP4 with negatize zero remap multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / groupsize, n)`
    @workspace: `torch.int` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_fp4_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


# Precompute permutations for Marlin weight and scale shuffling 

def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = torch.tensor(perm, dtype=torch.int64)
    interleave = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7], dtype=torch.int64)
    perm = perm.view(-1, 8)[:, interleave].reshape(-1)

    scale_perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)], dtype=torch.int64)
    scale_perm_single = torch.tensor(
        [2 * i + j for i in range(4) for j in [0, 1, 8, 9, 16, 17, 24, 25]],
        dtype=torch.int64,
    )

    return perm, scale_perm, scale_perm_single

_perm, _scale_perm, _scale_perm_single = _get_perms()

# launch configurations table
try:
    # Use explicit relative import so it works when `marlin_fp4` is installed as a package
    from .launch_config import MARLIN_FP4_TUNING
    print("Imported MARLIN_FP4_TUNING from launch_config.py")
except ImportError:
    print("launch_config.py not found, using empty MARLIN_FP4_TUNING")
    MARLIN_FP4_TUNING = {}



def _cvt_to_fp4_e2m1(w: torch.Tensor) -> torch.Tensor:
    """
    Convert FP16/FP32 tensor to FP4 E2M1 codes (4-bit) via rounding.

    We use the (finite) E2M1 value set (magnitudes): {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    and encode as:
    code = (sign<<3) | mag_code
    where mag_code in [0..7] corresponds to the ordered list above.

    Notes:
    - E2M1 has no Inf/NaN encodings; conversions saturate to max-norm. :contentReference[oaicite:1]{index=1}
    - Max magnitude is 6 for FP4 E2M1. :contentReference[oaicite:2]{index=2}
    - Later will use a cuda kernel for speed, this is a pytorch architecture-independent reference implementation.
    - Negative zero is mapped to positive zero.    
    """
    # x = w.to(torch.float32)

    # Match CUDA's "satfinite-ish" behavior: clamp infinities; map NaN -> +maxnorm.
    x = torch.nan_to_num(w, nan=6.0, posinf=6.0, neginf=-6.0)

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
    idx += (ax >= 5.00).to(torch.int32)  # >= midpoint(4,6) => 6

    code = idx  # 0..7
    code |= (sign.to(torch.int32) << 3)  # add sign bit -> 0..15
    # replace negative zero codes with positive zero codes
    code[(code == 8) ] = 0
    return code

class Layer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit fp4 (symmetric grouped) linear layer without bias.
    functions:
    1. __init__: create empty Marlin layer, with specified input/output features and groupsize.
    2. forward: perform Marlin matmul on input tensor.
    3. pack: pack a fake-quantized linear layer into Marlin representation (for test purposes only).
    4. quick_quantize_fp4: quantize and pack a torch.nn.Linear layer using basic FP4(E2M1) quantization.
    All other methods are internal helper functions.
    """

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer('workspace', torch.zeros(self.n // 128 * 16, dtype=torch.int), persistent=False)
        
        # build launch config table
        self._m_config_table = {}
        self.SMS = -1
        self._cfg_device = None  # torch.device or None


    def forward(self, A):
        C = torch.empty(A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device)
        A_r = A.view((-1, A.shape[-1]))
        thread_k, thread_n, sms = self._m_config_table.get(A_r.shape[0], (-1, -1, self.SMS))
        mul(A_r, self.B, C.view((-1, C.shape[-1])), self.s, self.workspace, thread_k, thread_n, sms)
        return C

    def pack(self, linear, scales):
        """
        Original packer used for test purposes.
        Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        # maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = _cvt_to_fp4_e2m1(w / s)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, _scale_perm.numel()))[:, _scale_perm.to(s.device)]
        else:
            s = s.reshape((-1, _scale_perm_single.numel()))[:, _scale_perm_single.to(s.device)]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        perm = _perm.to(res.device)
        res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)
        q = torch.zeros((res.shape[0], res.shape[1] // 8), dtype=torch.int32, device=res.device)
        for i in range(8):
            q |= res[:, i::8] << (4 * i)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)


    def _rebuild_launch_config_table(self):
        dev = self.B.device  # buffer device is the real module device
        self._cfg_device = dev

        if dev.type != "cuda":
            self._m_config_table = {}
            self.SMS = -1
            return

        props = torch.cuda.get_device_properties(dev.index)
        self.SMS = props.multi_processor_count
        dev_tbl = MARLIN_FP4_TUNING.get(props.name)
        m_table = {}
        if dev_tbl is not None:
            for (m0, k0, n0), cfg in dev_tbl.items():
                if k0 == self.k and n0 == self.n:
                    m_table[m0] = cfg
        self._m_config_table = m_table
        
    def _apply(self, fn, recurse=True):
        # Let PyTorch move params/buffers first
        super()._apply(fn, recurse=recurse)

        # Rebuild tuning table after device move
        self._rebuild_launch_config_table()
        return self


    def _compute_scales(self, w: torch.Tensor) -> torch.Tensor:
        """Phase 1: compute fp16 scales of shape (n, k / groupsize)."""
        fp4_max = 6.0
        eps = 1e-8
        n, k = w.shape
        w_grouped = w.to(torch.float32).reshape(n, k // self.groupsize, self.groupsize)
        scales = (w_grouped.abs().amax(dim=2) / fp4_max).clamp(min=eps).to(torch.half)
        return scales

    def _quantize_fp4_codes(self, w: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Phase 2: quantize with scales -> int32 codes shaped (n, k)."""
        n, k = w.shape
        w_grouped = w.to(torch.float32).reshape(n, k // self.groupsize, self.groupsize)
        scale_broadcast = scales.to(torch.float32).view(n, k // self.groupsize, 1)
        return _cvt_to_fp4_e2m1(w_grouped / scale_broadcast).reshape(n, k).to(torch.int32)

    def _pack_weights_and_scales(self, scales: torch.Tensor, q_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Phase 3: permute + bit-pack weights and shuffle scales.
        scales: (n, k / groupsize)  dtype=torch.half
        q_codes: (n, k)         dtype=torch.int32, values in [0..15]
        returns:
            packed: (k / 16, n * 16 / 8) dtype=torch.int32
            s: (k / groupsize, n)        dtype=torch.half
        """
        s = scales.transpose(0, 1).contiguous()
        w_int = q_codes.transpose(0, 1).contiguous()

        if self.groupsize != self.k:
            perm_s = _scale_perm.to(s.device)
            s = s.reshape(1, -1)
        else:
            perm_s = _scale_perm_single.to(s.device)

        s = s.reshape(-1, perm_s.numel())[:, perm_s]
        s = s.reshape(-1, self.n).contiguous()

        tile = 16
        w_tiles = w_int.reshape(self.k // tile, tile, self.n // tile, tile)
        w_tiles = w_tiles.permute(0, 2, 1, 3)
        w_tiles = w_tiles.reshape(self.k // tile, self.n * tile)

        perm_w = _perm.to(w_tiles.device)
        w_perm = w_tiles.reshape(-1, perm_w.numel())[:, perm_w]
        w_perm = w_perm.reshape_as(w_tiles).to(torch.int32)

        w_chunks = w_perm.reshape(w_perm.shape[0], -1, 8)
        shifts = torch.arange(8, device=w_perm.device, dtype=torch.int32).view(1, 1, 8) * 4
        packed = ((w_chunks & 0xF) << shifts).sum(dim=2).to(torch.int32)

        return packed, s

    def quick_quantize_fp4(self, linear: torch.nn.Linear):
        """Quantize and pack in three explicit phases (scales, codes, pack)."""
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')

        group_size = self.groupsize
        if group_size <= 0:
            raise ValueError('`groupsize` must be positive.')

        with torch.no_grad():
            w = linear.weight.data
            n, k = w.shape
            if k % group_size != 0:
                raise ValueError(f"infeatures ({k}) must be divisible by groupsize ({group_size}).")

            scales = self._compute_scales(w)
            q_codes = self._quantize_fp4_codes(w, scales)
            packed, packed_scales = self._pack_weights_and_scales(scales, q_codes)

            self.B[:, :] = packed.to(self.B.device)
            self.s[:, :] = packed_scales.to(self.s.device)

        torch.cuda.empty_cache()
        

        
        
