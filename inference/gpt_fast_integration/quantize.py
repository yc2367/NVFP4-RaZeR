import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

import custom_linears
from model import Transformer


def _is_layers_linear_fqn(fqn: str, mod: nn.Module) -> bool:
    """Return True only for nn.Linear modules inside Transformer.layers.*"""
    return isinstance(mod, nn.Linear) and isinstance(fqn, str) and fqn.startswith("layers.")


def _layers_root(mod: nn.Module) -> nn.Module:
    """Return module subtree to operate on (prefer Transformer.layers when present)."""
    return getattr(mod, "layers", mod)


def _resolve_fp_dtype(fp_dtype):
    if isinstance(fp_dtype, torch.dtype):
        return fp_dtype
    s = str(fp_dtype).lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"):  return torch.float16
    raise ValueError("fp_dtype must be 'bf16'/'bfloat16' or 'fp16'/'float16' (or a torch.dtype)")

class WeightOnlyMarlinRaZeRQuantHandler:
    """Marlin-RaZeR FP16xFP4(E2M1) weight-only quantization handler.

    Uses the marlin-razer extension's reference quantization+packing:
      - marlin_razer.Layer.quick_quantize_razer4(linear)

    Exports per Linear:
      - marlinrazer_B : int32 [K//16, N*16//8]
      - marlinrazer_s : fp16  [K//groupsize, N]

    Requirements:
      - in_features % 128 == 0
      - out_features % 256 == 0
      - groupsize in [-1, 128]
      - fp_dtype == fp16
    """

    def __init__(self, mod: nn.Module, groupsize: int, fp_dtype: str):
        self.mod = mod
        self.groupsize = groupsize
        self.fp_dtype = _resolve_fp_dtype(fp_dtype)

        if groupsize not in [-1, 128]:
            raise ValueError("Marlin-RaZeR only supports groupsize -1 (per-channel) or 128")
        if self.fp_dtype != torch.float16:
            raise ValueError("Marlin-RaZeR currently only supports fp16")

    @staticmethod
    def _check_compat(in_features: int, out_features: int, groupsize: int) -> tuple[bool, str]:
        if in_features % 128 != 0:
            return False, f"in_features {in_features} must be divisible by 128"
        if out_features % 256 != 0:
            return False, f"out_features {out_features} must be divisible by 256"
        if groupsize not in [-1, 128]:
            return False, f"groupsize {groupsize} must be -1 or 128"
        if groupsize != -1 and (in_features % groupsize) != 0:
            return False, f"in_features {in_features} must be divisible by groupsize {groupsize}"
        return True, "OK"

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda: bool = True):
        device = torch.device("cuda" if use_cuda else "cpu")
        import marlin_razer

        cur_state_dict = self.mod.state_dict()
        for fqn, mod in tqdm(list(self.mod.named_modules()), desc="Marlin-RaZeR quantization"):
            if not _is_layers_linear_fqn(fqn, mod):
                continue

            compatible, msg = self._check_compat(mod.in_features, mod.out_features, self.groupsize)
            if not compatible:
                raise ValueError(f"Layer {fqn} not compatible with Marlin-RaZeR: {msg}")

            # Ensure FP16 weights on target device
            weight = mod.weight.detach().to(device=device, dtype=torch.float16)

            fake_linear = nn.Linear(mod.in_features, mod.out_features, bias=False).to(
                device=device, dtype=torch.float16
            )
            fake_linear.weight.data.copy_(weight)

            mr_layer = marlin_razer.Layer(mod.in_features, mod.out_features, groupsize=self.groupsize).to(device)
            mr_layer.quick_quantize_razer4(fake_linear)

            cur_state_dict[f"{fqn}.marlinrazer_B"] = mr_layer.B.detach().cpu()
            cur_state_dict[f"{fqn}.marlinrazer_s"] = mr_layer.s.detach().cpu()

            if f"{fqn}.weight" in cur_state_dict:
                del cur_state_dict[f"{fqn}.weight"]
            if f"{fqn}.bias" in cur_state_dict:
                del cur_state_dict[f"{fqn}.bias"]

        return cur_state_dict

    def convert_for_runtime(self):
        root = _layers_root(self.mod)
        for name, child in root.named_children():
            if isinstance(child, nn.Linear):
                compatible, msg = self._check_compat(child.in_features, child.out_features, self.groupsize)
                if not compatible:
                    raise ValueError(f"Layer {name} not compatible with Marlin-RaZeR: {msg}")
                setattr(
                    root,
                    name,
                    custom_linears.MarlinRazerLinear(
                        child.in_features,
                        child.out_features,
                        self.groupsize,
                        self.fp_dtype,
                        bias=False,
                    ),
                )
            else:
                WeightOnlyMarlinRaZeRQuantHandler(child, self.groupsize, "fp16").convert_for_runtime()
        return self.mod

# --------------------------- Marlin-FP4 (FP16xFP4(E2M1)) ---------------------------

class WeightOnlyMarlinFP4QuantHandler:
    """Marlin-FP4 FP16xFP4(E2M1) weight-only quantization handler.

    Uses the marlin-fp4 extension's reference quantization+packing:
      - marlin_fp4.Layer.quick_quantize_fp4(linear)

    Exports per Linear:
      - marlinfp4_B : int32 [K//16, N*16//8]
      - marlinfp4_s : fp16  [K//groupsize, N]

    Requirements:
      - in_features % 128 == 0
      - out_features % 256 == 0
      - groupsize in [-1, 128]
      - fp_dtype == fp16
    """

    def __init__(self, mod: nn.Module, groupsize: int, fp_dtype: str):
        self.mod = mod
        self.groupsize = groupsize
        self.fp_dtype = _resolve_fp_dtype(fp_dtype)

        if groupsize not in [-1, 128]:
            raise ValueError("Marlin-FP4 only supports groupsize -1 (per-channel) or 128")
        if self.fp_dtype != torch.float16:
            raise ValueError("Marlin-FP4 currently only supports fp16")

    @staticmethod
    def _check_compat(in_features: int, out_features: int, groupsize: int) -> tuple[bool, str]:
        if in_features % 128 != 0:
            return False, f"in_features {in_features} must be divisible by 128"
        if out_features % 256 != 0:
            return False, f"out_features {out_features} must be divisible by 256"
        if groupsize not in [-1, 128]:
            return False, f"groupsize {groupsize} must be -1 or 128"
        if groupsize != -1 and (in_features % groupsize) != 0:
            return False, f"in_features {in_features} must be divisible by groupsize {groupsize}"
        return True, "OK"

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda: bool = True):
        device = torch.device("cuda" if use_cuda else "cpu")
        import marlin_fp4

        cur_state_dict = self.mod.state_dict()
        for fqn, mod in tqdm(list(self.mod.named_modules()), desc="Marlin-FP4 quantization"):
            if not _is_layers_linear_fqn(fqn, mod):
                continue

            compatible, msg = self._check_compat(mod.in_features, mod.out_features, self.groupsize)
            if not compatible:
                raise ValueError(f"Layer {fqn} not compatible with Marlin-FP4: {msg}")

            # Ensure FP16 weights on target device
            weight = mod.weight.detach().to(device=device, dtype=torch.float16)

            fake_linear = nn.Linear(mod.in_features, mod.out_features, bias=False).to(
                device=device, dtype=torch.float16
            )
            fake_linear.weight.data.copy_(weight)

            mr_layer = marlin_fp4.Layer(mod.in_features, mod.out_features, groupsize=self.groupsize).to(device)
            mr_layer.quick_quantize_fp4(fake_linear)

            cur_state_dict[f"{fqn}.marlinfp4_B"] = mr_layer.B.detach().cpu()
            cur_state_dict[f"{fqn}.marlinfp4_s"] = mr_layer.s.detach().cpu()
            if f"{fqn}.weight" in cur_state_dict:
                del cur_state_dict[f"{fqn}.weight"]
            if f"{fqn}.bias" in cur_state_dict:
                del cur_state_dict[f"{fqn}.bias"]

        return cur_state_dict

    def convert_for_runtime(self):
        root = _layers_root(self.mod)
        for name, child in root.named_children():
            if isinstance(child, nn.Linear):
                compatible, msg = self._check_compat(child.in_features, child.out_features, self.groupsize)
                if not compatible:
                    raise ValueError(f"Layer {name} not compatible with Marlin-FP4: {msg}")
                setattr(
                    root,
                    name,
                    custom_linears.MarlinFP4Linear(
                        child.in_features,
                        child.out_features,
                        self.groupsize,
                        self.fp_dtype,
                        bias=False,
                    ),
                )
            else:
                WeightOnlyMarlinFP4QuantHandler(child, self.groupsize, "fp16").convert_for_runtime()
        return self.mod


class WeightOnlyRaZeRQuantHandler:
    """
        RaZeR (FP4 e2m1) weight-only quantizer.

        Exports per Linear:
            - razer_qB      : int32 [K/8, N]
            - razer_scaling :
                * g128 -> fp16 [K/128, N]
                  bit14 selects 5 vs 8 and scale sign selects special sign.
                * g16  -> int32 [K/128, N, 2]
                  two packed uint32 words storing eight fp8-like scale bytes.
                  bit7 selects special sign, bit6 selects 5 vs 8, and the lower
                  6 bits encode a positive E3M3-like scale payload.
    """

    _RAZER_SPECIAL_U32 = [0x40A00000, 0x41000000, 0xC0A00000, 0xC1000000]  # +5,+8,-5,-8

    def __init__(self, mod: nn.Module, bits, groupsize, fp_dtype):
        assert groupsize in (16, 128), "RaZeR currently only supports groupsize=16 or groupsize=128"
        assert bits == 4, "RaZeR only supports 4-bit quantization"
        self.mod = mod
        self.groupsize = groupsize
        self.bits = bits
        self.fp_dtype = _resolve_fp_dtype(fp_dtype)
        assert self.fp_dtype == torch.float16, "RaZeR currently only supports fp16"
        self.padding = True

    @staticmethod
    def _u32_to_f32(u: int) -> float:
        import struct
        return struct.unpack('!f', struct.pack('!I', u))[0]

    @staticmethod
    def _fp4_e2m1_nibble_to_float(nib: int) -> float:
        s = (nib >> 3) & 1
        e = (nib >> 1) & 0x3
        m = nib & 1
        sign = -1.0 if s else 1.0
        if e == 0:
            return sign * (0.5 if m else 0.0)
        exp2 = e - 1
        base = float(1 << exp2) if exp2 >= 0 else 1.0 / float(1 << (-exp2))
        return sign * base * (1.0 + 0.5 * m)

    @classmethod
    def _razer_codebook_fp4_canonical(cls, special_is_negative: int, special_is_8: int) -> torch.Tensor:
        idx = (int(special_is_negative) << 1) | int(special_is_8)
        sv_bits = cls._RAZER_SPECIAL_U32[idx]
        sv_val = cls._u32_to_f32(sv_bits)
        table = []
        for x in range(16):
            if x == 0x0:
                table.append(sv_val)
            elif x == 0x8:
                table.append(0.0)
            else:
                table.append(cls._fp4_e2m1_nibble_to_float(x))
        return torch.tensor(table, dtype=torch.float32)

    @staticmethod
    def _pack8_nibbles_batched(nibs: torch.Tensor) -> torch.Tensor:
        device = nibs.device
        shifts = torch.arange(8, dtype=torch.int64, device=device) * 4
        packed = ((nibs.to(torch.int64) & 0xF) << shifts).sum(dim=1)
        return packed.to(torch.uint32)

    @staticmethod
    def _encode_scale_with_flags_fp16(scales_f16: torch.Tensor, choose_8: torch.Tensor) -> torch.Tensor:
        bits_u16 = scales_f16.view(torch.uint16)
        bits_i32 = bits_u16.to(torch.int32)
        bit14 = 0x4000
        assert ((bits_i32 & bit14) == 0).all().item(), "fp16 scale bit14 (0x4000) must be 0 before encoding"
        flag_i32 = ((choose_8.to(torch.int32) & 1) << 14)
        bits_i32 = bits_i32 | flag_i32
        return bits_i32.to(torch.uint16).view(torch.float16)

    @staticmethod
    def _g16_scale_table(device: torch.device) -> torch.Tensor:
        payload = torch.arange(64, dtype=torch.uint8)
        table = payload.view(torch.float8_e4m3fn).to(torch.float32) * 16.0
        return table.to(device=device)

    @classmethod
    def _quantize_g16_scales_to_payload(cls, scales_fp32: torch.Tensor):
        table = cls._g16_scale_table(scales_fp32.device)
        diffs = torch.abs(scales_fp32.unsqueeze(1) - table.unsqueeze(0))
        idx = torch.argmin(diffs, dim=1)
        return table[idx], idx.to(torch.uint8)

    @classmethod
    @torch.no_grad()
    def _quantize_groups_razer_4bit_batched(cls, groups: torch.Tensor):
        num_groups, group_size = groups.shape
        assert group_size in (16, 128)
        device = groups.device
        x = groups.to(torch.float32)

        best_indices = torch.zeros((num_groups, group_size), dtype=torch.int32, device=device)
        best_scales = torch.zeros((num_groups,), dtype=torch.float32, device=device)
        best_special_is_8 = torch.zeros((num_groups,), dtype=torch.int32, device=device)
        best_special_is_negative = torch.zeros((num_groups,), dtype=torch.int32, device=device)
        best_mse = torch.full((num_groups,), float('inf'), dtype=torch.float32, device=device)

        zero_mask = torch.all(x == 0, dim=1)
        if zero_mask.any():
            cb0 = cls._razer_codebook_fp4_canonical(0, 0).to(device)
            zero_idx = torch.argmin(torch.abs(cb0)).item()
            best_indices[zero_mask] = zero_idx
            best_scales[zero_mask] = 1.0
            best_special_is_8[zero_mask] = 0
            best_special_is_negative[zero_mask] = 0
            best_mse[zero_mask] = 0.0

        nonzero_mask = ~zero_mask
        if not nonzero_mask.any():
            return best_indices, best_scales, best_special_is_negative, best_special_is_8, best_mse

        X = x[nonzero_mask]
        global_idx = torch.where(nonzero_mask)[0]

        for special_is_negative in (0, 1):
            for special_is_8 in (0, 1):
                cb = cls._razer_codebook_fp4_canonical(special_is_negative, special_is_8).to(device)
                max_vals = torch.max(torch.abs(X), dim=1)[0]
                max_cb = torch.max(torch.abs(cb)).item()
                init_s = max_vals / max(max_cb, 1e-8)

                d1 = torch.abs(X.unsqueeze(2) - init_s[:, None, None] * cb[None, None, :])
                idx1 = torch.argmin(d1, dim=2)

                codes = cb[idx1]
                num = (X * codes).sum(dim=1)
                den = (codes * codes).sum(dim=1)
                valid = den > 1e-12
                ref_s = torch.where(valid, num / den, init_s).abs().clamp_min(1e-8)

                d2 = torch.abs(X.unsqueeze(2) - ref_s[:, None, None] * cb[None, None, :])
                idx2 = torch.argmin(d2, dim=2)
                recon = ref_s[:, None] * cb[idx2]
                mse = ((X - recon) ** 2).mean(dim=1)

                better = mse < best_mse[nonzero_mask]
                if better.any():
                    sel = global_idx[better]
                    best_indices[sel] = idx2[better].to(torch.int32)
                    best_scales[sel] = ref_s[better]
                    best_special_is_8[sel] = int(special_is_8)
                    best_special_is_negative[sel] = int(special_is_negative)
                    best_mse[sel] = mse[better]

        return best_indices, best_scales, best_special_is_negative, best_special_is_8, best_mse

    @classmethod
    @torch.no_grad()
    def _assign_indices_for_fixed_scales(
        cls,
        groups: torch.Tensor,
        scales: torch.Tensor,
        special_is_negative: torch.Tensor,
        special_is_8: torch.Tensor,
    ) -> torch.Tensor:
        num_groups, group_size = groups.shape
        device = groups.device
        x = groups.to(torch.float32)
        out = torch.empty((num_groups, group_size), dtype=torch.int32, device=device)

        for neg in (0, 1):
            for mag8 in (0, 1):
                mask = (special_is_negative == neg) & (special_is_8 == mag8)
                if not mask.any():
                    continue
                cb = cls._razer_codebook_fp4_canonical(neg, mag8).to(device)
                x_sel = x[mask]
                s_sel = scales[mask]
                d = torch.abs(x_sel.unsqueeze(2) - s_sel[:, None, None] * cb[None, None, :])
                out[mask] = torch.argmin(d, dim=2).to(torch.int32)
        return out

    @torch.no_grad()
    def create_quantized_state_dict(self, use_cuda=True):
        device = torch.device("cuda" if use_cuda else "cpu")
        sd = self.mod.state_dict()
        out_sd = dict(sd)

        for fqn, m in tqdm(list(self.mod.named_modules()), desc="RaZeR quant (FP4)"):
            if not _is_layers_linear_fqn(fqn, m):
                continue

            W = m.weight.detach().to(torch.float32).to(device)
            N, K = W.shape
            assert K % 128 == 0 and K % 8 == 0, f"{fqn}: K must be multiple of 128 and 8"

            k_packed = K // 8
            batch_size = min(16 if self.groupsize == 16 else 32, N)

            if self.groupsize == 128:
                G = K // 128
                qB = torch.empty((N, k_packed), dtype=torch.int32, device=device)
                scaling_f16 = torch.empty((N, G), dtype=torch.float16, device=device)

                for bs in range(0, N, batch_size):
                    be = min(bs + batch_size, N)
                    rows = W[bs:be]
                    B = be - bs

                    groups = rows.reshape(B * G, 128)
                    idx, scl, special_is_negative, special_is_8, _ = self._quantize_groups_razer_4bit_batched(groups)

                    idx = idx.reshape(B, G, 128)
                    scl = scl.reshape(B, G)
                    special_is_negative = special_is_negative.reshape(B, G).to(torch.int32)
                    special_is_8 = special_is_8.reshape(B, G).to(torch.int32)

                    neg = special_is_negative.to(torch.bool)
                    choose_8 = special_is_8

                    if neg.any():
                        flip_mask = neg.unsqueeze(-1).expand(-1, -1, 128)
                        idx = idx.clone()
                        do_flip = flip_mask & (idx != 0) & (idx != 8)
                        idx[do_flip] = idx[do_flip] ^ 8

                    scales = scl.to(torch.float16)
                    scales = torch.where(neg, -scales, scales)
                    scales = self._encode_scale_with_flags_fp16(scales, choose_8)
                    scaling_f16[bs:be] = scales

                    idx_words = idx.reshape(B, G, 16, 8).reshape(-1, 8)
                    packed_words = self._pack8_nibbles_batched(idx_words)
                    qB[bs:be] = packed_words.reshape(B, G * 16)

                out_sd[f"{fqn}.razer_qB"] = qB.t().contiguous().cpu()
                out_sd[f"{fqn}.razer_scaling"] = scaling_f16.t().contiguous().cpu()
            else:
                G16 = K // 16
                G8 = K // 128
                qB = torch.empty((N, k_packed), dtype=torch.int32, device=device)
                scaling_packed = torch.empty((N, G8, 2), dtype=torch.int32, device=device)

                for bs in range(0, N, batch_size):
                    be = min(bs + batch_size, N)
                    rows = W[bs:be]
                    B = be - bs

                    groups = rows.reshape(B * G16, 16)
                    idx, scl, special_is_negative, special_is_8, _ = self._quantize_groups_razer_4bit_batched(groups)
                    q_scales, scale_payload = self._quantize_g16_scales_to_payload(scl)
                    idx = self._assign_indices_for_fixed_scales(groups, q_scales, special_is_negative, special_is_8)

                    idx = idx.reshape(B, G16, 16)
                    scale_payload = scale_payload.reshape(B, G16)
                    special_is_negative = special_is_negative.reshape(B, G16).to(torch.uint8)
                    special_is_8 = special_is_8.reshape(B, G16).to(torch.uint8)

                    scale_bytes = scale_payload | ((special_is_8 & 1) << 6) | ((special_is_negative & 1) << 7)

                    idx_words = idx.reshape(B, G16, 2, 8).reshape(-1, 8)
                    packed_words = self._pack8_nibbles_batched(idx_words)
                    qB[bs:be] = packed_words.reshape(B, G16 * 2).to(torch.int32)

                    shifts = torch.arange(4, dtype=torch.int64, device=device) * 8
                    scale_bytes = scale_bytes.reshape(B, G8, 8)
                    word0 = ((scale_bytes[:, :, :4].to(torch.int64) & 0xFF) << shifts).sum(dim=-1)
                    word1 = ((scale_bytes[:, :, 4:].to(torch.int64) & 0xFF) << shifts).sum(dim=-1)
                    scaling_packed[bs:be, :, 0] = word0.to(torch.int32)
                    scaling_packed[bs:be, :, 1] = word1.to(torch.int32)

                out_sd[f"{fqn}.razer_qB"] = qB.t().contiguous().cpu()
                out_sd[f"{fqn}.razer_scaling"] = scaling_packed.permute(1, 0, 2).contiguous().cpu()

            if f"{fqn}.weight" in out_sd:
                del out_sd[f"{fqn}.weight"]

        return out_sd

    def convert_for_runtime(self):
        import custom_linears
        root = _layers_root(self.mod)
        for name, child in root.named_children():
            if isinstance(child, nn.Linear):
                setattr(root, name, custom_linears.RaZeRLinear(
                    child.in_features, child.out_features, self.bits, self.groupsize, self.fp_dtype, padding=True
                ))
            else:
                WeightOnlyRaZeRQuantHandler(child, self.bits, self.groupsize, self.fp_dtype).convert_for_runtime()
        return self.mod


def quantize(
    checkpoint_path: Path,
    mode: str,
    groupsize: int = 128,
    label: str = "",
    fp_dtype: str = "fp16",
    use_cuda: bool = True,
) -> Path:
    assert checkpoint_path.is_file(), checkpoint_path
    if mode not in ("razer", "marlinrazer", "marlinfp4"):
        raise ValueError("mode must be one of: razer, marlinrazer, marlinfp4")

    with torch.device("meta"):
        model = Transformer.from_name(checkpoint_path.parent.name)

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)
    model = model.to(dtype=torch.bfloat16, device="cpu")

    if mode == "razer":
        print("Quantizing model weights for RaZeR FP4(E2M1) weight-only quantization")
        if groupsize == 16:
            print("Using experimental g16 scale packing. Runtime uses the kernel for M==1 and dequant+GEMM for larger M.")
        quant_handler = WeightOnlyRaZeRQuantHandler(model, 4, groupsize, fp_dtype)
        suffix = f"{label}razer.g{groupsize}.{fp_dtype}.pth"
    elif mode == "marlinrazer":
        print("Quantizing model weights for Marlin-RaZeR FP16xFP4(E2M1) with negative-zero remap")
        quant_handler = WeightOnlyMarlinRaZeRQuantHandler(model, groupsize, fp_dtype)
        suffix = f"{label}marlinrazer.g{groupsize}.{fp_dtype}.pth"
    else:
        print("Quantizing model weights for Marlin-FP4 FP16xFP4(E2M1)")
        quant_handler = WeightOnlyMarlinFP4QuantHandler(model, groupsize, fp_dtype)
        suffix = f"{label}marlinfp4.g{groupsize}.{fp_dtype}.pth"

    quantized_state_dict = quant_handler.create_quantized_state_dict(use_cuda=use_cuda)
    output_path = checkpoint_path.parent / checkpoint_path.name.replace(".pth", suffix)
    print(f"Saving quantized checkpoint to {output_path}")
    torch.save(quantized_state_dict, output_path)
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare weight-only checkpoints for the public RaZeR inference artifact")
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--mode", choices=["razer", "marlinrazer", "marlinfp4"], required=True)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--fp_dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()
    quantize(
        checkpoint_path=args.checkpoint_path,
        mode=args.mode,
        groupsize=args.groupsize,
        label=args.label,
        fp_dtype=args.fp_dtype,
        use_cuda=args.use_cuda,
    )


if __name__ == "__main__":
    _main()
