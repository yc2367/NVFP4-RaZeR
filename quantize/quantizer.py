import torch
from typing import Optional
from functools import partial
from .quant_config import QuantConfig
from .utils import quant_scale



@torch.no_grad()
def quant_nf4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        4-bit Normal-Float quantization.
    """

    quant_value  = sorted(
        [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, 
        -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
    )
    mid_value    = [
        (quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)
    ]

    orig_shape   = w_fp.shape 
    w_fp_new     = w_fp.reshape(-1, groupsize).to(torch.float32)

    rmax         = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    qmax         = max([abs(x) for x in quant_value])
    scale_fp     = (rmax / qmax).clamp_(min=1e-7)
    w_scaled     = w_fp_new / scale_fp

    w_q = torch.zeros_like(w_scaled)
    for i, data in enumerate(quant_value):
        if i == 0:
            w_q += torch.where(
                w_scaled <= mid_value[i], 
                data, 0
            )
        elif i == len(quant_value) - 1:
            w_q += torch.where(
                w_scaled > mid_value[i - 1], 
                data, 0
            )
        else:
            w_q += torch.where(
                (mid_value[i - 1] < w_scaled) & (w_scaled <= mid_value[i]), 
                data, 0
            )

    w_dq = w_q * scale_fp 

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_hf4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        HiFloat4 quantization. Following the implementation https://arxiv.org/pdf/2602.11287
    """
    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, 4)

    # Block Maximum
    max_4  = w_fp_new.abs().amax(dim=-1, keepdim=True)
    max_8  = max_4.reshape(-1, 2).amax(dim=-1, keepdim=True).repeat(1, 2).view(-1, 1)
    max_64 = max_4.reshape(-1, 16).amax(dim=-1, keepdim=True).repeat(1, 16).view(-1, 1)

    # Block Scale
    scale_64 = max_64 / 7
    scale_64_max, scale_64_min = 2**15 * 1.5, 2**(-48)
    scale_64 = quant_scale(
        scale_64.clamp(min=scale_64_min, max=scale_64_max),
        exp_bits=6, man_bits=2, exp_min=-48
    )
    scale_8  = torch.where(
        max_8 / scale_64 >= 4, 
        2, 1
    )
    scale_4  = torch.where(
        max_64 / (scale_64 * scale_8) >= 2, 
        2, 1
    )

    # Block Quantized Element
    w_scaled = w_fp_new / (scale_64 * scale_8 * scale_4)
    w_q      = (w_scaled * 4).clamp(min=-7, max=7).round() / 4
    w_dq     = w_q * (scale_64 * scale_8 * scale_4)

    return w_dq.view(orig_shape)


@torch.no_grad()
def quant_mxfp4_orig(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        Original MXFP4 quantization.
    """
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

    FP4_EXP_BITS = 2
    FP4_MAN_BITS = 1
    FP4_EMAX     = 2
    FP4_MAX      = 6.0

    orig_shape = w_fp.shape 
    w_fp_new = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    shared_exp = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(
        torch.log2(
            shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype)
        )
    )

    # Offset the max exponent by the largest representable exponent
    # in the element data format
    shared_exp = (shared_exp - FP4_EMAX).clamp(min=-FP32_EXPONENT_BIAS+1, max=FP32_EXPONENT_BIAS)
    w_s        = w_fp_new / (2**shared_exp)

    # FP4 Quantization
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q         = w_m * (2**private_exp) / (2**FP4_MAN_BITS)
    w_q         = torch.clamp(w_q, min=-FP4_MAX, max=FP4_MAX)

    # De-Quantization
    w_dq        = w_q * (2**shared_exp)

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        Better MXFP4 quantization.
    """
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

    FP4_MAN_BITS = 1
    FP4_MAX      = 6.0

    orig_shape = w_fp.shape 
    w_fp_new = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    scale_fp32 = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) / FP4_MAX
    shared_exp = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-FP32_EXPONENT_BIAS+1, max=FP32_EXPONENT_BIAS)
    w_s        = w_fp_new / (2**shared_exp) 
    
    # FP4 Quantization
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    # De-Quantization
    w_dq = w_q * (2**shared_exp)

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxif4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        MXFP4 and MXINT4 quantization, where a block is quantized to FP4 / INT4 with lower quantization error.
    """
    # FP4 quantization values
    quant_value_fp4 = sorted([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mid_value_fp4   = [
        (quant_value_fp4[i] + quant_value_fp4[i + 1]) / 2 for i in range(len(quant_value_fp4) - 1)
    ]
  
    SCALE_EMAX      = 63
    FP32_MIN_NORMAL = 2 ** (-126)
    FP4_MAN_BITS    = 1
    FP4_MAX         = 6.0
    INT4_MAX        = 7.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    block_max  = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True)

    ########## MXFP4 quantization ##########
    block_scale_fp4     = block_max / FP4_MAX
    block_exp_fp4       = torch.ceil(
        torch.log2(
            block_scale_fp4 + FP32_MIN_NORMAL * (block_scale_fp4 == 0).type(block_max.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q_fp4   = 2**block_exp_fp4
    w_s_fp4             = w_fp_new / block_scale_q_fp4

    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s_fp4) + (w_s_fp4 == 0).type(w_s_fp4.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m_fp4     = w_s_fp4 / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m_fp4     = torch.sign(w_m_fp4) * torch.floor(torch.abs(w_m_fp4) + 0.5)
    w_q_fp4     = w_m_fp4 * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## MXINT4 quantization ##########
    block_scale_int4    = block_max / INT4_MAX
    block_exp_int4      = torch.ceil(
        torch.log2(
            block_scale_int4 + FP32_MIN_NORMAL * (block_scale_int4 == 0).type(block_max.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q_int4  = 2**block_exp_int4
    w_s_int4            = w_fp_new / block_scale_q_int4
    w_q_int4            = w_s_int4.clamp(min=-7, max=7).round()

    ########## Select between MXFP4 and MXINT4 quantization ##########
    quant_error_fp4   = ((w_q_fp4 * block_scale_q_fp4 - w_fp_new) ** 2).sum(dim=-1)
    quant_error_int4  = ((w_q_int4 * block_scale_q_int4 - w_fp_new) ** 2).sum(dim=-1)
    select_fp4        = (quant_error_fp4 < quant_error_int4)[:, None]

    w_q  = torch.where(
        select_fp4,
        w_q_fp4,
        w_q_int4,
    )
    block_scale_q = torch.where(
        select_fp4,
        block_scale_q_fp4,
        block_scale_q_int4,
    )
    # Dequantization
    w_dq = w_q * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4_razer(w_fp, n_bits: int=4, groupsize: Optional[int]=None, is_act: bool=False):
    """
        MXFP4-RaZeR quantization.
    """
    SCALE_EMAX      = 31
    FP32_MIN_NORMAL = 2**(-126)
    FP4_EXP_BITS    = 2
    FP4_MAN_BITS    = 1
    FP4_EMAX        = 2
    FP4_MAX         = 6.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    scale_fp32     = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) / FP4_MAX
    shared_exp     = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_scale_q  = 2**shared_exp
    w_s            = w_fp_new / block_scale_q
    
    ########## Normal FP4 quantization ##########
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4     = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Special Value ##########
    if is_act:
        special_value_list = [-5.0, 5.0]
    else:
        special_value_list = [-5.0, 5.0, -3.5, 3.5]

    error     = torch.full([w_fp_new.shape[0]], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in special_value_list:
        w_q_razer_tmp = torch.where(
            (w_s - w_q_fp4).abs() < (w_s - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - w_s).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_mxfp4_razer_new(w_fp, n_bits: int=4, groupsize: Optional[int]=None, is_act: bool=False):
    """
        MXFP4-RaZeR quantization.
    """
    SCALE_EMAX      = 31
    FP32_MIN_NORMAL = 2**(-126)
    FP4_EXP_BITS    = 2
    FP4_MAN_BITS    = 1
    FP4_EMAX        = 2
    FP4_MAX         = 6.0

    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
    
    block_max      = torch.amax(w_fp_new.abs(), dim=-1, keepdim=True) 
    scale_fp32     = block_max / FP4_MAX
    shared_exp     = torch.ceil(
        torch.log2(
            scale_fp32 + FP32_MIN_NORMAL * (scale_fp32 == 0).type(scale_fp32.dtype)
        )
    ).clamp(min=-SCALE_EMAX, max=SCALE_EMAX)
    block_max_s    = block_max / 2**shared_exp
    upscale_mask   = block_max_s.lt(3.5).squeeze() # If the scaled block maximum is smaller than 3.5, then upscale the whole block by 2
    shared_exp[upscale_mask] -= 1
    block_scale_q  = 2**shared_exp
    w_s            = w_fp_new / block_scale_q
    
    ########## Normal FP4 quantization ##########
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_s) + (w_s == 0).type(w_s.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m         = w_s / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m         = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4     = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Special Value ##########
    if is_act:
        special_value_list = [-5.0, 5.0]
    else:
        special_value_list = [-5.0, 5.0, -7.0, 7.0]

    error     = torch.full([w_fp_new.shape[0]], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in special_value_list:
        w_q_razer_tmp = torch.where(
            (w_s - w_q_fp4).abs() < (w_s - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - w_s).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4 quantization. 
    """
    FP4_MAX       = 6.0
    FP4_MAN_BITS  = 1

    orig_shape    = w_fp.shape 
    w_fp_new      = w_fp.reshape(-1, groupsize).to(torch.float32)

    global_qmax   = FP4_MAX * 448
    global_scale  = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled      = w_fp_new / global_scale
    block_max     = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_q = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled      = w_scaled / block_scale_q

    #################### FP4 Quantization ####################
    private_exp   = torch.floor(
        torch.log2(
            torch.abs(w_scaled) + (w_scaled == 0).type(w_scaled.dtype)
        )
    )
    private_exp   = private_exp.clamp(min=0)
    w_m           = w_scaled / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m           = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q           = w_m * (2**private_exp) / (2**FP4_MAN_BITS)
    w_dq          = w_q * block_scale_q * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_4over6(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4 4over6 quantization. Following the implementation of FourOverSix (https://arxiv.org/pdf/2512.02010)
    """
    quant_value = sorted([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mid_value = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]

    orig_shape = w_fp.shape 
    w_fp_new = w_fp.reshape(-1, groupsize).to(torch.float32)

    qmax_6       = 6.0
    qmax_4       = 4.0
    global_qmax  = qmax_6 * 448
    global_scale = w_fp_new.abs().amax() / global_qmax

    w_scaled       = w_fp_new / global_scale
    w_dq_sign      = w_scaled.sign()
    block_scale    = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_6  = (block_scale / qmax_6).clamp(
        max=torch.finfo(torch.float8_e4m3fn).max,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    block_scale_4  = (block_scale / qmax_4).clamp(
        max=torch.finfo(torch.float8_e4m3fn).max,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled_6     = (w_scaled / block_scale_6).abs()
    w_scaled_4     = (w_scaled / block_scale_4).abs()

    w_q_6 = torch.zeros_like(w_scaled)
    for i in range(len(quant_value)):
        data = quant_value[i]
        if i == 0:
            w_q_6 += torch.where(w_scaled_6 <= mid_value[i], data, 0)
        elif i == len(quant_value) - 1:
            w_q_6 += torch.where(w_scaled_6 > mid_value[i - 1], data, 0)
        else:
            w_q_6 += torch.where((mid_value[i - 1] < w_scaled_6) & (w_scaled_6 <= mid_value[i]), data, 0)
    w_q_6 = w_q_6 * w_dq_sign

    w_q_4 = torch.zeros_like(w_scaled)
    for i in range(len(quant_value)):
        data = quant_value[i]
        if i == 0:
            w_q_4 += torch.where(w_scaled_4 <= mid_value[i], data, 0)
        elif i == len(quant_value) - 1:
            w_q_4 += torch.where(w_scaled_4 > mid_value[i - 1], data, 0)
        else:
            w_q_4 += torch.where((mid_value[i - 1] < w_scaled_4) & (w_scaled_4 <= mid_value[i]), data, 0)
    w_q_4 = w_q_4 * w_dq_sign

    quant_error_6 = ((w_q_6*block_scale_6 - w_scaled) ** 2).sum(dim=-1)
    quant_error_4 = ((w_q_4*block_scale_4 - w_scaled) ** 2).sum(dim=-1)
    select_4      = (quant_error_4 < quant_error_6)[:, None]

    w_q = torch.where(
        select_4,
        w_q_4,
        w_q_6,
    )
    block_scale = torch.where(
        select_4,
        block_scale_4,
        block_scale_6,
    )

    w_dq = w_q * block_scale * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvif4(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVIF4 quantization. Following the implementation of (https://arxiv.org/abs/2603.28765)
    """
    FP4_MAX      = 6.0
    INT4_MAX     = 7.0
    FP4_MAN_BITS = 1

    #################### Reshape Tensor ####################
    orig_shape = w_fp.shape 
    w_fp_new   = w_fp.reshape(-1, groupsize).to(torch.float32)
 
    #################### Global Scale ####################
    global_qmax  = FP4_MAX * 448
    global_scale = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled         = w_fp_new / global_scale
    block_max        = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_fp4  = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    block_scale_int4 = (block_max / INT4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled_fp4     = w_scaled / block_scale_fp4
    w_scaled_int4    = w_scaled / block_scale_int4

    #################### FP4 Quantization ####################
    private_exp = torch.floor(
        torch.log2(
            torch.abs(w_scaled_fp4) + (w_scaled_fp4 == 0).type(w_scaled_fp4.dtype)
        )
    )
    private_exp = private_exp.clamp(min=0)
    w_m_fp4     = w_scaled_fp4 / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m_fp4     = torch.sign(w_m_fp4) * torch.floor(torch.abs(w_m_fp4) + 0.5)
    w_q_fp4     = w_m_fp4 * (2**private_exp) / (2**FP4_MAN_BITS)

    #################### INT4 Quantization ####################
    w_q_int4    = w_scaled_int4.clamp(min=-7, max=7).round()

    #################### Select Optimal Data Type ####################
    quant_error_fp4  = ((w_q_fp4 * block_scale_fp4 - w_scaled) ** 2).sum(dim=-1)
    quant_error_int4 = ((w_q_int4 * block_scale_int4 - w_scaled) ** 2).sum(dim=-1)
    select_int4      = (quant_error_int4 < quant_error_fp4)[:, None]
    w_q              = torch.where(
        select_int4,
        w_q_int4,
        w_q_fp4,
    )
    block_scale   = torch.where(
        select_int4,
        block_scale_int4,
        block_scale_fp4,
    )

    w_dq = w_q * block_scale * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_razer_e3m3(w_fp, n_bits: int=4, groupsize: Optional[int]=None, outlier: float=8.0):
    """
        NVFP4-RaZeR quantization.
    """

    inlier  = 5.0
    datatype_list = [
        [inlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [-inlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [outlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        [-outlier, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    ]

    #################### Reshape Tensor ####################
    orig_shape   = w_fp.shape 
    w_fp_new     = w_fp.view(-1, groupsize).to(torch.float32)
    num_group    = w_fp_new.shape[0]

    #################### Global Scale ####################
    global_qmax  = 6.0 * 28
    global_scale = w_fp_new.abs().amax() / global_qmax
    #################### Block Maximum ####################
    w_scaled     = w_fp_new / global_scale
    block_max    = w_scaled.abs().amax(dim=-1, keepdim=True)

    ############### Optimal Data Type Search ###############
    w_q          = torch.zeros_like(w_fp_new)
    block_scale  = torch.zeros(num_group, 1, dtype=w_fp_new.dtype, device=w_fp_new.device)
    quant_error  = torch.full([num_group], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    # Iterate through data types
    for quant_value in datatype_list:
        quant_value     = sorted(quant_value)
        mid_value       = [(quant_value[i] + quant_value[i + 1]) / 2 for i in range(len(quant_value) - 1)]
        qmax_tmp        = abs(max(quant_value, key=abs))
        block_scale_tmp = (block_max / qmax_tmp).clamp(
            max=28,
            min=2**(-5)
        ).to(torch.float8_e4m3fn).to(w_fp_new.dtype)
        w_scaled_tmp    = w_scaled / block_scale_tmp

        # Fake Quantization
        w_q_tmp = torch.zeros_like(w_scaled_tmp)
        for i, data in enumerate(quant_value):
            if i == 0:
                w_q_tmp += torch.where(w_scaled_tmp <= mid_value[i], data, 0)
            elif i == len(quant_value) - 1:
                w_q_tmp += torch.where(w_scaled_tmp > mid_value[i - 1], data, 0)
            else:
                w_q_tmp += torch.where((mid_value[i - 1] < w_scaled_tmp) & (w_scaled_tmp <= mid_value[i]), data, 0)

        # Quantization Error
        quant_error_tmp = (w_q_tmp*block_scale_tmp - w_scaled).pow(2).mean(dim=-1)
        # Update Data Type if Smaller Quantization Error
        mask_update               = torch.lt(quant_error_tmp, quant_error)
        w_q[mask_update]          = w_q_tmp[mask_update]
        block_scale[mask_update]  = block_scale_tmp[mask_update]
        quant_error[mask_update]  = quant_error_tmp[mask_update]

    w_dq = w_q * block_scale * global_scale
    return w_dq.view(orig_shape).to(torch.bfloat16)


@torch.no_grad()
def quant_nvfp4_razer_e4m3(w_fp, n_bits: int=4, groupsize: Optional[int]=None):
    """
        NVFP4-RaZeR quantization.
    """
    FP4_MAX      = 6.0
    FP4_MAN_BITS = 1

    orig_shape     = w_fp.shape 
    w_fp_new       = w_fp.reshape(-1, groupsize).to(torch.float32)
    num_group      = w_fp_new.shape[0]

    inlier         = 5.0
    global_qmax    = FP4_MAX * 448
    global_scale   = w_fp_new.abs().amax() / global_qmax

    ############### Block Scale Quantization ###############
    w_scaled      = w_fp_new / global_scale
    block_max     = w_scaled.abs().amax(dim=-1, keepdim=True)
    block_scale_q = (block_max / FP4_MAX).clamp(
        max=448,
        min=2**(-9)
    ).to(torch.float8_e4m3fn).to(w_scaled.dtype)
    w_scaled      = w_scaled / block_scale_q

    #################### FP4 Quantization ####################
    private_exp   = torch.floor(
        torch.log2(
            torch.abs(w_scaled) + (w_scaled == 0).type(w_scaled.dtype)
        )
    )
    private_exp   = private_exp.clamp(min=0)
    w_m           = w_scaled / (2**private_exp) * (2**FP4_MAN_BITS)
    w_m           = torch.sign(w_m) * torch.floor(torch.abs(w_m) + 0.5)
    w_q_fp4       = w_m * (2**private_exp) / (2**FP4_MAN_BITS)

    ########## Search for the Optimal RaZeR-FP4 Data Type ##########
    error     = torch.full([num_group], float('inf'), dtype=w_fp_new.dtype, device=w_fp_new.device)
    w_q_razer = torch.zeros_like(w_fp_new)
    for special_value in [-inlier, inlier]:
        # Handle special value
        w_q_razer_tmp = torch.where(
            (w_scaled - w_q_fp4).abs() < (w_scaled - special_value).abs(),
            w_q_fp4, special_value
        )
        # Dequantize and calculate error
        quant_error            = (w_q_razer_tmp - block_scale_q).pow(2).mean(-1)
        mask_update            = torch.lt(quant_error, error)
        error[mask_update]     = quant_error[mask_update]
        w_q_razer[mask_update] = w_q_razer_tmp[mask_update]
    ##################################################################

    w_dq = w_q_razer * block_scale_q * global_scale

    return w_dq.view(orig_shape).to(torch.bfloat16)


def quant_weight(model, quant_config: QuantConfig):
    n_bits       = quant_config.w_bits
    w_groupsize  = quant_config.w_groupsize
    w_dtype      = quant_config.w_dtype.lower()
    w_outlier    = quant_config.w_outlier

    print(f"Performing LLM weight quantization using Data Type:  {w_dtype}\n")

    if (n_bits >= 16) or (w_dtype is None) or (w_dtype in ["fp16", "fp32"]):
        return

    n_bits      = 4
    quant_func  = None
    if (w_dtype == "mxfp4_orig"):
        quant_func = quant_mxfp4_orig
    elif (w_dtype == "mxfp4"):
        quant_func  = quant_mxfp4
    elif (w_dtype == "mxif4"):
        quant_func  = quant_mxif4
    elif (w_dtype == "mxfp4_razer"):
        quant_func  = quant_mxfp4_razer
    elif (w_dtype == "mxfp4_razer_new"):
        quant_func  = quant_mxfp4_razer_new
    elif (w_dtype == "nf4"):
        quant_func = quant_nf4  
    elif (w_dtype == "hf4"):
        quant_func = quant_hf4
    elif (w_dtype == "nvfp4"):
        quant_func = quant_nvfp4
    elif (w_dtype == "nvfp4_4over6"):
        quant_func = quant_nvfp4_4over6
    elif (w_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (w_dtype == "nvfp4_razer_e3m3"):
        quant_func = partial(quant_nvfp4_razer_e3m3, outlier=w_outlier)
    elif (w_dtype == "nvfp4_razer_e4m3"):
        quant_func = quant_nvfp4_razer_e4m3
    else:
        raise ValueError(f"Unsupported Data Type: {w_dtype}")
    
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and ('head' not in n):
            m.weight.data = quant_func(m.weight.data, n_bits=n_bits, groupsize=w_groupsize)


def quant_act(act, quant_config: QuantConfig):
    n_bits       = quant_config.a_bits
    a_groupsize  = quant_config.a_groupsize
    a_dtype      = quant_config.a_dtype.lower()

    if (n_bits >= 16) or (a_dtype in ["fp16", "fp32"]):
        return act

    n_bits      = 4
    quant_func  = None

    if (a_dtype == "mxfp4"):
        quant_func = quant_mxfp4
    elif (a_dtype == "mxfp4_orig"):
        quant_func = quant_mxfp4_orig
    elif (a_dtype == "mxif4"):
        quant_func = quant_mxif4
    elif (a_dtype == "mxfp4_razer"):
        quant_func  = partial(quant_mxfp4_razer, is_act=True)
    elif (a_dtype == "mxfp4_razer_new"):
        quant_func  = partial(quant_mxfp4_razer_new, is_act=True)
    elif (a_dtype == "nf4"):
        quant_func = quant_nf4
    elif (a_dtype == "hf4"):
        quant_func = quant_hf4
    elif (a_dtype == "nvfp4"):
        quant_func = quant_nvfp4
    elif (a_dtype == "nvfp4_4over6"):
        quant_func = quant_nvfp4_4over6
    elif (a_dtype == "nvif4"):
        quant_func = quant_nvif4
    elif (a_dtype == "nvfp4_razer_e4m3"):
        quant_func = quant_nvfp4_razer_e4m3
    else:
        raise ValueError(f"Unsupported Data Type: {a_dtype}")
    
    return quant_func(act, n_bits=n_bits, groupsize=a_groupsize)
