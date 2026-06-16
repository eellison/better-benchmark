"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete channels-last Inception BN-inference affine, explicit bf16 cast, NaN-preserving bf16 ReLU, and 3x3 stride-2 low-memory maxpool value output in one Triton stencil kernel while eliminating the unused int8 offset output, whereas Inductor materializes the full rounded ReLU tensor and lowers `_low_memory_max_pool_with_offsets` as a generic tuple-producing pooling region even though only the values escape; Inductor cannot do this today because scheduler fusion does not sink broadcast BN/ReLU producers through low-memory maxpool while also dead-code-eliminating unused offsets and preserving bf16 rounding boundaries and channels-last strides; the fix is SCHEDULER_FUSION: teach the pooling scheduler to absorb per-channel BN/ReLU producers and emit a value-only maxpool stencil when offsets are dead."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 64
H_IN = 147
W_IN = 147
H_OUT = 73
W_OUT = 73
OUT_HW = H_OUT * W_OUT
EPS = 0.001


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return tl.where(x != x, x, rounded.to(tl.float32, bitcast=True))


@triton.jit
def _bn_relu_maxpool_values_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS_: tl.constexpr,
    H_IN_: tl.constexpr,
    W_IN_: tl.constexpr,
    W_OUT_: tl.constexpr,
    OUT_HW_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    batch = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    s_offsets = tl.program_id(2) * BLOCK_S + tl.arange(0, BLOCK_S)

    c_mask = c_offsets < CHANNELS_
    s_mask = s_offsets < OUT_HW_
    out_h = s_offsets // W_OUT_
    out_w = s_offsets - out_h * W_OUT_

    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + EPS_)

    best = tl.full((BLOCK_S, BLOCK_C), -float("inf"), tl.float32)
    input_batch_base = batch * (CHANNELS_ * H_IN_ * W_IN_)

    for kh in tl.static_range(0, 3):
        in_h = out_h * 2 + kh
        for kw in tl.static_range(0, 3):
            in_w = out_w * 2 + kw
            offsets = (
                input_batch_base
                + in_h[:, None] * (W_IN_ * CHANNELS_)
                + in_w[:, None] * CHANNELS_
                + c_offsets[None, :]
            )
            mask = s_mask[:, None] & c_mask[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

            y = x - mean[None, :]
            y = y * inv_std[None, :]
            y = y * weight[None, :]
            y = y + bias[None, :]
            y = _round_bf16_to_fp32(y)
            y = tl.where((y > 0.0) | (y != y), y, 0.0)

            take = mask & ((y > best) | (y != y))
            best = tl.where(take, y, best)

    out_offsets = (
        batch * (CHANNELS_ * OUT_HW_)
        + s_offsets[:, None] * CHANNELS_
        + c_offsets[None, :]
    )
    tl.store(out_ptr + out_offsets, best, mask=s_mask[:, None] & c_mask[None, :])


# 5d3e4e56: bf16 Inception BN/ReLU -> 3x3 stride-2 maxpool values, channels-last NCHW.
@oracle_impl(hardware="B200", point="5d3e4e56", BLOCK_C=64, BLOCK_S=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_S: int, num_warps: int):
    mean, x, var, weight, bias = inputs[:5]
    out = torch.empty_strided(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        (CHANNELS * OUT_HW, 1, W_OUT * CHANNELS, CHANNELS),
        device=x.device,
        dtype=torch.bfloat16,
    )
    grid = (BATCH, triton.cdiv(CHANNELS, BLOCK_C), triton.cdiv(OUT_HW, BLOCK_S))
    _bn_relu_maxpool_values_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        CHANNELS_=CHANNELS,
        H_IN_=H_IN,
        W_IN_=W_IN,
        W_OUT_=W_OUT,
        OUT_HW_=OUT_HW,
        EPS_=EPS,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
