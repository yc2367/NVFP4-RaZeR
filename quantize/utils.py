import torch


def get_scale_min_max(exp_bits, man_bits):
    exp_bias  = 2**(exp_bits-1) - 1
    exp_max   = 2**(exp_bits-1)
    exp_min   = -2**(exp_bits-1) + 2
    qmax      = 2**exp_max * (2 - 2**(-man_bits+1))
    qmin      = 2**(exp_min - man_bits)
    return qmax, qmin


def quant_scale(scale_fp, exp_bits, man_bits):
    exp_min    = -2**(exp_bits-1) + 2
    scale_sign = scale_fp.sign()
    assert (scale_sign == -1).any().logical_not(), "The scaling factor CANNOT be negative. Something is WRONG..."
    scale_exp  = (
        scale_fp + (scale_fp == 0).type(scale_fp.dtype)
    ).log2().floor().clamp_(min=exp_min)
    scale_man  = torch.round(
        scale_fp / 2**scale_exp * 2**man_bits
    ) / (2**man_bits)
    scale_dq   = scale_sign * 2**scale_exp * scale_man 

    return scale_dq

