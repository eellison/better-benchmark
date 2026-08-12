"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete channels-last Inception BN-inference affine, explicit bf16 cast, NaN-preserving bf16 ReLU, 3x3 stride-2 low-memory maxpool value output, and padded 3x3 stride-1 avg_pool2d over the returned maxpool tensor, whereas Inductor currently materializes the full bf16 ReLU activation before generic maxpool and avgpool stencil regions; Inductor cannot do this today because scheduler fusion does not sink a broadcast affine producer through the low-memory maxpool stencil and keep the returned maxpool values resident for the sibling avgpool consumer while preserving bf16 rounding boundaries and channels-last strides; the fix is SCHEDULER_FUSION: teach the pooling scheduler to absorb per-channel BN/ReLU producers and emit the required returned pool plus avgpool outputs from a fused stencil plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 192
H_IN = 71
W_IN = 71
H_OUT = 35
W_OUT = 35
OUT_HW = H_OUT * W_OUT
EPS = 0.001


@triton.jit
def _round_bf16_to_fp32(x):
    bits = x.to(tl.uint32, bitcast=True)
    lsb = (bits >> 16) & 1
    rounded = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return tl.where(x != x, x, rounded.to(tl.float32, bitcast=True))


@triton.jit
def _bn_relu_maxpool_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    pool_ptr,
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
    tl.store(pool_ptr + out_offsets, best, mask=s_mask[:, None] & c_mask[None, :])


@triton.jit
def _avgpool_kernel(
    pool_ptr,
    out_ptr,
    CHANNELS_: tl.constexpr,
    H_OUT_: tl.constexpr,
    W_OUT_: tl.constexpr,
    OUT_HW_: tl.constexpr,
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
    batch_base = batch * (CHANNELS_ * OUT_HW_)

    acc = tl.zeros((BLOCK_S, BLOCK_C), tl.float32)
    for kh in tl.static_range(0, 3):
        in_h = out_h + kh - 1
        h_valid = (in_h >= 0) & (in_h < H_OUT_)
        for kw in tl.static_range(0, 3):
            in_w = out_w + kw - 1
            valid = s_mask & h_valid & (in_w >= 0) & (in_w < W_OUT_)
            offsets = (
                batch_base
                + (in_h[:, None] * W_OUT_ + in_w[:, None]) * CHANNELS_
                + c_offsets[None, :]
            )
            value = tl.load(
                pool_ptr + offsets,
                mask=valid[:, None] & c_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += value

    out_offsets = (
        batch_base
        + s_offsets[:, None] * CHANNELS_
        + c_offsets[None, :]
    )
    avg = _round_bf16_to_fp32(acc / 9.0)
    tl.store(out_ptr + out_offsets, avg, mask=s_mask[:, None] & c_mask[None, :])


# 760bed68: bf16 Inception BN/ReLU -> maxpool -> avgpool, channels-last NCHW.
@oracle_impl(hardware="B200", point="760bed68", BLOCK_C=64, BLOCK_S=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_S, num_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs[:5]
    out_shape = (BATCH, CHANNELS, H_OUT, W_OUT)
    out_stride = (CHANNELS * OUT_HW, 1, W_OUT * CHANNELS, CHANNELS)
    pool = torch.empty_strided(out_shape, out_stride, device=arg1_1.device, dtype=torch.bfloat16)
    avg = torch.empty_strided(out_shape, out_stride, device=arg1_1.device, dtype=torch.bfloat16)

    grid = (BATCH, triton.cdiv(CHANNELS, BLOCK_C), triton.cdiv(OUT_HW, BLOCK_S))
    _bn_relu_maxpool_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        pool,
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
    _avgpool_kernel[grid](
        pool,
        avg,
        CHANNELS_=CHANNELS,
        H_OUT_=H_OUT,
        W_OUT_=W_OUT,
        OUT_HW_=OUT_HW,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps,
        num_stages=3,
    )
    return pool, avg
