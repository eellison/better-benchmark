"""cuTile port of pointwise_46ffe591b5ee: EfficientNet BN-inference + SiLU + constant_pad.

Per (b, c) row, apply BN affine (rounded to bf16), then SiLU (rounded to bf16),
then torch.constant_pad_nd([1,2,1,2]) via a torch epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_silu_kernel(
    running_mean_ptr,
    x_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    C: ct.Constant[int],
):
    row = ct.bid(0)
    channel = row - (row // C) * C

    x = ct.load(
        x_ptr,
        index=(row, 0),
        shape=(1, HW_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    mean = ct.astype(ct.load(running_mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(running_var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    invstd = ct.rsqrt(var + EPS)
    normalized = (x_f - mean) * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf16, ct.float32)

    # SiLU: x / (1 + exp(-x))
    exp_neg = ct.exp(-affine_f)
    silu = affine_f / (exp_neg + 1.0)
    silu_bf16 = ct.astype(silu, ct.bfloat16)

    # Mask OOB elements.
    cols = ct.arange(HW_PAD, dtype=ct.int32)
    valid = cols < HW
    valid_2d = ct.reshape(valid, (1, HW_PAD))
    zero_bf16 = ct.full(shape=(1, HW_PAD), fill_value=0.0, dtype=ct.bfloat16)
    out_masked = ct.where(valid_2d, silu_bf16, zero_bf16)

    ct.store(out_ptr, index=(row, 0), tile=out_masked)


def _launch(inputs, *, BLOCK_HW: int, BLOCK_C: int, PAD_BLOCK: int, num_warps: int):
    running_mean, x, running_var, weight, bias, _pad = inputs
    batch, channels, height, width = x.shape
    HW = height * width
    HW_PAD = _next_pow2(HW)

    rows = batch * channels
    x_nchw = x.contiguous().view(rows, HW)
    inner = torch.empty_strided(
        (rows, HW),
        (HW, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _bn_silu_kernel,
        (running_mean, x_nchw, running_var, weight, bias, inner, HW, HW_PAD, channels),
    )
    inner_nchw = inner.view(batch, channels, height, width)
    # Pad [1, 2, 1, 2] means (left=1, right=2, top=1, bottom=2)
    padded = torch.nn.functional.pad(inner_nchw, [1, 2, 1, 2], mode="constant", value=0.0)
    # Repro output stride matches channels-last:
    out_h = height + 3
    out_w = width + 3
    out = torch.empty_strided(
        (batch, channels, out_h, out_w),
        (channels * out_h * out_w, 1, out_w * channels, channels),
        device=x.device,
        dtype=torch.bfloat16,
    )
    out.copy_(padded)
    return out


@oracle_impl(hardware="B200", point="8162a5bc", BLOCK_HW=32, BLOCK_C=64, PAD_BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="51719261", BLOCK_HW=64, BLOCK_C=64, PAD_BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, PAD_BLOCK: int, num_warps: int):
    return _launch(
        inputs,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        PAD_BLOCK=PAD_BLOCK,
        num_warps=num_warps,
    )
