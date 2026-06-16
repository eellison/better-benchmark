"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 bf16 channel-LayerNorm training scope in one shape-specialized Triton kernel, including the channels-last NCHW input viewed as NHWC, bf16-to-fp32 channel statistics over dim=3 with correction=0, eps=1e-6 rsqrt, returned f32 mean and rsqrt side outputs, fp32 affine scale/bias, returned f32 channels-last NCHW view, and final bf16 channels-last NCHW cast, whereas Inductor lowers the permute/cast/var_mean/affine/permute/cast graph through generic normalization and pointwise/layout schedules; Inductor cannot do this full returned-output envelope today because its normalization scheduler does not fuse the channel-statistics reduction with both sibling side outputs and the dtype-aware dual f32/bf16 epilogue for this fixed C=80 channels-last layout; the fix is SCHEDULER_FUSION: add a guarded ConvNeXt channel-LayerNorm schedule that emits mean, invstd, f32 affine, and bf16 affine outputs directly from the row-reduction tile while preserving the channels-last output strides."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _convnext_channel_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    invstd_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_C: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    channels = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS
    channel_mask = channels < C

    spatial = rows % (H * W)
    n_idx = rows // (H * W)
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    offsets = (
        n_idx[:, None] * X_STRIDE_N
        + channels[None, :] * X_STRIDE_C
        + h_idx[:, None] * X_STRIDE_H
        + w_idx[:, None] * X_STRIDE_W
    )
    mask = row_mask[:, None] & channel_mask[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_for_sum = tl.where(channel_mask[None, :], x, 0.0)
    mean = tl.sum(x_for_sum, axis=1) / C
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(channel_mask[None, :], centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / C
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-6))

    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    out_offsets = (
        n_idx[:, None] * OUT_STRIDE_N
        + channels[None, :] * OUT_STRIDE_C
        + h_idx[:, None] * OUT_STRIDE_H
        + w_idx[:, None] * OUT_STRIDE_W
    )
    tl.store(mean_ptr + rows, mean, mask=row_mask)
    tl.store(invstd_ptr + rows, invstd, mask=row_mask)
    tl.store(out_f32_ptr + out_offsets, affine, mask=mask)
    tl.store(out_bf16_ptr + out_offsets, affine.to(tl.bfloat16), mask=mask)


# 32d9a8b7: (T([128,80,56,56], bf16, stride=(250880,1,4480,80)), T([80], f32), T([80], f32))
@oracle_impl(hardware="B200", point="32d9a8b7", BLOCK_R=8, BLOCK_C=128, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    rows = batch * height * width
    out_stride = (channels * height * width, 1, width * channels, channels)

    mean = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_f32 = torch.empty_strided(
        (batch, channels, height, width),
        out_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (batch, channels, height, width),
        out_stride,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _convnext_channel_layernorm_kernel[(triton.cdiv(rows, BLOCK_R),)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        invstd,
        out_f32,
        out_bf16,
        ROWS=rows,
        C=channels,
        H=height,
        W=width,
        X_STRIDE_N=int(arg0_1.stride(0)),
        X_STRIDE_C=int(arg0_1.stride(1)),
        X_STRIDE_H=int(arg0_1.stride(2)),
        X_STRIDE_W=int(arg0_1.stride(3)),
        OUT_STRIDE_N=out_stride[0],
        OUT_STRIDE_C=out_stride[1],
        OUT_STRIDE_H=out_stride[2],
        OUT_STRIDE_W=out_stride[3],
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean, invstd, out_f32, out_bf16
