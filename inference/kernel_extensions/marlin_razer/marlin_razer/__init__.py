'''
Marlin-Razer: Fast 4-bit Linear Layers with Negative-Zero Remap Quantization matmul kernel.

Usage:
    1. Kernel mul (low-level):
        from marlin_razer import mul
        m, n, k = 16, 4096, 4096
        A = torch.randn((m, k), dtype=torch.half, device='cuda')
        B = torch.randint(0, 16, (k // 16, n * 16 // 8), dtype=torch.int32, device='cuda')  # pre-packed
        C = torch.zeros((m, n), dtype=torch.half, device='cuda')
        s = torch.randn((k // 128, n), dtype=torch.half, device='cuda')  # groupsize=128
        workspace = torch.zeros(n // 128 * 16, dtype=torch.int32, device='cuda')
        mul(A, B, C, s, workspace)
    
    2. Linear layer (high-level):
        from marlin_razer import Layer
        linear_fp16 = layer.self_attn.q_proj # derived from the model, type torch.nn.Linear
        marlin_layer = Layer(linear_fp16.in_features, linear_fp16.out_features, groupsize=128).cuda()
        marlin_layer.quick_quantize_razer4(linear_fp16) # quantize and pack weights
        output = marlin_layer(input)
        
    3. launch config table (MARLIN_RAZER_TUNING):
        provied best launch configurations for various GPUs and problem sizes.
        format:
        MARLIN_RAZER_TUNING = {
          'GPU Name': {
            (M, K, N): (thread_k, thread_n, sms),
            ...
          },
          ...
        }
'''


import torch
import torch.nn as nn


import marlin_razer_cuda

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
    marlin_razer_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


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
    # Use explicit relative import so it works when `marlin_razer` is installed as a package
    from .launch_config import MARLIN_RAZER_TUNING
    print("Imported MARLIN_RAZER_TUNING from launch_config.py")
except ImportError:
    print("launch_config.py not found, using empty MARLIN_RAZER_TUNING")
    MARLIN_RAZER_TUNING = {}



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
    """PyTorch compatible Marlin layer; 4-bit fp4/razer4 (symmetric grouped) linear layer without bias.
    functions:
    1. __init__: create empty Marlin layer, with specified input/output features and groupsize.
    2. forward: perform Marlin matmul on input tensor.
    3. pack: pack a fake-quantized linear layer into Marlin representation (for test purposes only).
    4. quick_quantize_fp4: quantize and pack a torch.nn.Linear layer using basic FP4(E2M1) quantization.
    5. quick_quantize_razer4: quantize and pack a torch.nn.Linear layer using FP4(E2M1) with negative-zero remap quantization.
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
        dev_tbl = MARLIN_RAZER_TUNING.get(props.name)
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

    def _quantize_fp4_negzero_remap(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast groupwise FP4(E2M1) quantization with negative-zero remap special value selection.

        Tries 4 modes per group:
        0: +5 special, scale = max_abs/6,  scale sign +, LSB(flag)=0
        1: +8 special, scale = max_abs/8,  scale sign +, LSB(flag)=1
        2: -5 special, scale = max_abs/6,  scale sign -, LSB(flag)=0
        3: -8 special, scale = max_abs/8,  scale sign -, LSB(flag)=1

        IMPORTANT:
        Kernel truncates the scale LSB before use (scale is effectively fp15).
        Therefore we quantize using the same "effective scale" = fp16(scale) with LSB forced to 0.
        """
        if w.dtype != torch.half:
            raise ValueError("Only torch.half weights are supported.")
        if w.dim() != 2:
            raise ValueError("w must be 2D (n, k).")

        eps = 1e-8
        n, k = w.shape
        G = self.groupsize
        if k % G != 0:
            raise ValueError(f"infeatures ({k}) must be divisible by groupsize ({G}).")

        groups = k // G
        device = w.device

        # (n, groups, G)
        w_g = w.to(torch.float32).reshape(n, groups, G)
        max_abs = w_g.abs().amax(dim=2)  # (n, groups)

        # ---- fp15-effective scale helper (match kernel: clear mantissa LSB) ----
        def _to_fp15_effective(scale_f32: torch.Tensor) -> torch.Tensor:
            h = scale_f32.clamp(min=eps).to(torch.half)  # fp16
            u16 = h.view(torch.uint16)
            u32 = u16.to(torch.int32)
            u32 = (u32 & 0xFFFE)  # clear mantissa LSB
            return u32.to(torch.uint16).view(torch.half)

        # scale candidates (fp15-effective)
        scale6_eff_h = _to_fp15_effective(max_abs / 6.0)  # (n, groups) half
        scale8_eff_h = _to_fp15_effective(max_abs / 8.0)

        scale6_eff = scale6_eff_h.to(torch.float32)  # (n, groups) float32
        scale8_eff = scale8_eff_h.to(torch.float32)

        # scaled domain
        x6 = w_g / scale6_eff.unsqueeze(-1)  # (n, groups, G)
        x8 = w_g / scale8_eff.unsqueeze(-1)

        # ---- fast base FP4(E2M1) rounding (no special, no -0) ----
        # magnitudes: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        mag_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                device=device, dtype=torch.float32)
        # midpoints:  [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        th = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00],
                        device=device, dtype=torch.float32)

        def _fp4_base_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Returns:
            qx: float32 quantized values in FP4 base set
            code: int32 codes in [0..15], with -0 (code==8) forced to +0 (code==0)
            """
            sign = (x < 0)
            ax = x.abs()

            # idx in [0..7]
            idx = torch.zeros_like(ax, dtype=torch.int32)
            idx += (ax >= th[0]).to(torch.int32)
            idx += (ax >= th[1]).to(torch.int32)
            idx += (ax >= th[2]).to(torch.int32)
            idx += (ax >= th[3]).to(torch.int32)
            idx += (ax >= th[4]).to(torch.int32)
            idx += (ax >= th[5]).to(torch.int32)
            idx += (ax >= th[6]).to(torch.int32)

            mag = mag_levels[idx.to(torch.long)]
            qx = torch.where(sign, -mag, mag)

            code = ((sign.to(torch.int32) << 3) | idx).to(torch.int32)
            # remove accidental -0 => +0
            code = torch.where(code == 8, torch.zeros_like(code), code)
            return qx, code

        q6_base, code6_base = _fp4_base_quant(x6)
        q8_base, code8_base = _fp4_base_quant(x8)

        # ---- special masks (in scaled domain) ----
        ax6 = x6.abs()
        ax8 = x8.abs()

        # # +5 special: only positive, only between (4,5,6) midpoints => [4.5, 5.5)
        # mask_p5 = (x6 >= 0) & (ax6 >= 4.5) & (ax6 < 5.5)
        # # -5 special: only negative, symmetric window
        # mask_n5 = (x6 < 0) & (ax6 >= 4.5) & (ax6 < 5.5)

        # # +8 special: only positive, saturate above midpoint(6,8)=7
        # mask_p8 = (x8 >= 0) & (ax8 >= 7.0)
        # # -8 special: only negative, saturate below -7
        # mask_n8 = (x8 < 0) & (ax8 >= 7.0)
        
        mask_p5 = (x6 < 0) & (ax6 >= 4.5) & (ax6 < 5.5)
        mask_n5 = (x6 >= 0) & (ax6 >= 4.5) & (ax6 < 5.5)
        mask_p8 = (x8 < 0) & (ax8 >= 7.0)
        mask_n8 = (x8 >= 0) & (ax8 >= 7.0)

        # ---- MSE per group for 4 modes (no building q tensors) ----
        # error in original domain: (x - qx)^2 * scale^2
        err6_base = (x6 - q6_base) ** 2
        err8_base = (x8 - q8_base) ** 2

        mse0 = torch.where(mask_p5, (x6 - 5.0) ** 2, err6_base).mean(dim=2) * (scale6_eff ** 2)  # +5
        mse2 = torch.where(mask_n5, (x6 + 5.0) ** 2, err6_base).mean(dim=2) * (scale6_eff ** 2)  # -5
        mse1 = torch.where(mask_p8, (x8 - 8.0) ** 2, err8_base).mean(dim=2) * (scale8_eff ** 2)  # +8
        mse3 = torch.where(mask_n8, (x8 + 8.0) ** 2, err8_base).mean(dim=2) * (scale8_eff ** 2)  # -8

        mse_all = torch.stack([mse0, mse1, mse2, mse3], dim=0)  # (4, n, groups)
        best = torch.argmin(mse_all, dim=0)                     # (n, groups)

        # ---- build code candidates (int32), special lane is code==8 ----
        code0 = torch.where(mask_p5, torch.full_like(code6_base, 8), code6_base)  # +5
        code2 = torch.where(mask_n5, torch.full_like(code6_base, 8), code6_base)  # -5
        code1 = torch.where(mask_p8, torch.full_like(code8_base, 8), code8_base)  # +8
        code3 = torch.where(mask_n8, torch.full_like(code8_base, 8), code8_base)  # -8

        # negative modes: negative-scale trick => flip sign bit for NON-special, NON-zero magnitudes
        def _flip_for_negative(codes: torch.Tensor) -> torch.Tensor:
            # do not flip +0 -> -0 (would become special lane)
            flip = ((codes & 0x7) != 0)
            return torch.where(flip, codes ^ 0x8, codes)

        code2 = _flip_for_negative(code2)
        code3 = _flip_for_negative(code3)

        # ---- select final codes by best ----
        q_final = torch.empty((n, groups, G), device=device, dtype=torch.int32)
        q_final = torch.where((best == 0).unsqueeze(-1), code0, q_final)
        q_final = torch.where((best == 1).unsqueeze(-1), code1, q_final)
        q_final = torch.where((best == 2).unsqueeze(-1), code2, q_final)
        q_final = torch.where((best == 3).unsqueeze(-1), code3, q_final)

        q_codes = q_final.reshape(n, k).contiguous()

        # ---- encode scales: sign + LSB flag (0=>5, 1=>8), keep fp15-effective magnitude ----
        flag = (best & 1).to(torch.int32)   # 0 => 5, 1 => 8
        neg = (best >= 2)

        scale_eff_h = torch.where(flag == 0, scale6_eff_h, scale8_eff_h)  # (n, groups) half, LSB=0
        scale_signed = torch.where(neg, -scale_eff_h, scale_eff_h)

        # set LSB flag in stored scale (kernel reads it, then truncates it away)
        u16 = scale_signed.view(torch.uint16)
        u32 = u16.to(torch.int32)
        u32 = (u32 & 0xFFFE) | flag
        scales_encoded = u32.to(torch.uint16).view(torch.half).contiguous()

        return scales_encoded, q_codes

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
        
    def quick_quantize_razer4(self, linear: torch.nn.Linear):
        """Quantize and pack in three explicit phases (scales, codes, pack).
            with negative-zero remap special value handling.
        """
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

            scales, q_codes = self._quantize_fp4_negzero_remap(w)
            packed, packed_scales = self._pack_weights_and_scales(scales, q_codes)

            self.B[:, :] = packed.to(self.B.device)
            self.s[:, :] = packed_scales.to(self.s.device)

        torch.cuda.empty_cache()
        
        
        
