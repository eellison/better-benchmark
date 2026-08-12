"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 bf16 channel-LayerNorm training scope in one shape-specialized Triton row kernel for both recorded points, including the channels-last NCHW residual add rounded to the returned contiguous NHWC `permute` output, the fp32 residual-add row tile feeding population `var_mean(..., dim=3, correction=0, keepdim=True)`, eps=1e-6 rsqrt side output, fp32 affine scale/bias epilogue, and final bf16 channels-last NCHW cast, whereas Inductor lowers the captured add/permute/cast/var_mean/affine/permute/cast graph through generic normalization and layout schedules; Inductor cannot fuse this exact returned-output envelope today because the norm-template scheduler does not keep the visible bf16 add producer, channel statistics, mean/rsqrt side-output stores, and final layout/cast epilogue in one ConvNeXt channel-LayerNorm plan while preserving the observable rounding boundaries; the fix is SCHEDULER_FUSION: add a guarded channels-last NCHW channel-LayerNorm schedule that emits the NHWC add, statistics, and rounded affine output directly."""

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
def _convnextv2_channel_layernorm_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
    add_nhwc_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_bf16_ptr,
    C: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    rows = row_ids[:, None]
    channels = tl.arange(0, BLOCK_C)[None, :]
    row_mask = row_ids < TOTAL_ROWS
    channel_mask = channels < C
    mask = row_mask[:, None] & channel_mask
    offsets = rows * C + channels

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = _f32_add(x0, x1)
    added = values.to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(add_nhwc_ptr + offsets, added, mask=mask)

    values_masked = tl.where(mask, values, 0.0)
    sum_values = tl.sum(values_masked, axis=1)[:, None]
    sum_squares = tl.sum(_f32_mul(values_masked, values_masked), axis=1)[:, None]
    mean = sum_values / C
    centered = _f32_sub(values, mean)
    variance = _f32_add(sum_squares / C, -_f32_mul(mean, mean))
    variance = tl.maximum(variance, 0.0)
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-6))

    channel_ids = tl.arange(0, BLOCK_C)
    channel_ids_mask = channel_ids < C
    weight = tl.load(weight_ptr + channel_ids, mask=channel_ids_mask, other=0.0).to(tl.float32)[None, :]
    bias = tl.load(bias_ptr + channel_ids, mask=channel_ids_mask, other=0.0).to(tl.float32)[None, :]
    rounded_centered = _f32_sub(added.to(tl.float32), mean)
    normalized = _f32_mul(rounded_centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    final = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(mean_ptr + row_ids[:, None], mean, mask=row_mask[:, None])
    tl.store(rsqrt_ptr + row_ids[:, None], invstd, mask=row_mask[:, None])
    tl.store(out_bf16_ptr + offsets, final, mask=mask)


def _channels_last_stride(batch: int, channels: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (channels * height * width, 1, channels * width, channels)


def _nhwc_stride(batch: int, height: int, width: int, channels: int) -> tuple[int, int, int, int]:
    return (height * width * channels, width * channels, channels, 1)


# f9c4eb2d: (T([128,320,14,14], bf16, stride=(62720,1,4480,320)), T([128,320,14,14], bf16, stride=(62720,1,4480,320)), T([320], f32), T([320], f32))
@oracle_impl(hardware="B200", point="f9c4eb2d", BLOCK_C=512, BLOCK_ROWS=4, num_warps=8, num_stages=3)
# e5e4b0b5: (T([128,160,28,28], bf16, stride=(125440,1,4480,160)), T([128,160,28,28], bf16, stride=(125440,1,4480,160)), T([160], f32), T([160], f32))
@oracle_impl(hardware="B200", point="e5e4b0b5", BLOCK_C=256, BLOCK_ROWS=16, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_ROWS: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    total_rows = batch * height * width

    add_nhwc = torch.empty_strided(
        (batch, height, width, channels),
        _nhwc_stride(batch, height, width, channels),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (batch, height, width, 1),
        (height * width, width, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (batch, channels, height, width),
        _channels_last_stride(batch, channels, height, width),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _convnextv2_channel_layernorm_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_nhwc,
        mean,
        rsqrt,
        out_bf16,
        C=channels,
        TOTAL_ROWS=total_rows,
        BLOCK_C=BLOCK_C,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_nhwc, mean, rsqrt, out_bf16
