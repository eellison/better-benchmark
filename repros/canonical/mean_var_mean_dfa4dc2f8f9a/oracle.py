"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 bf16 residual-add, 7x7 spatial mean side output, fp32 channel `var_mean(..., [3], correction=0)`, eps=1e-6 rsqrt side output, and affine bf16 result with a specialized spatial-reduction stage feeding a row-wise normalization epilogue while preserving the two visible bf16 cast boundaries, whereas Inductor lowers the returned pooled tensor, channel normalization, and affine epilogue as generic producer/consumer schedules; Inductor cannot do this today because its norm-template scheduler does not fuse a returned spatial reduction producer into a following channelwise layernorm plan while also emitting the intermediate mean and rsqrt outputs; the fix is SCHEDULER_FUSION: extend norm-template fusion to carry returned reduction side outputs and bf16 rounding boundaries through the full scope."""

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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _spatial_mean_kernel(
    x_ptr,
    residual_ptr,
    pooled_ptr,
    partials_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    residual_stride_n: tl.constexpr,
    residual_stride_c: tl.constexpr,
    residual_stride_h: tl.constexpr,
    residual_stride_w: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw < HEIGHT * WIDTH

    n = rows // CHANNELS
    channels = rows - n * CHANNELS
    h = hw // WIDTH
    w = hw - h * WIDTH
    mask = row_mask[:, None] & hw_mask[None, :]

    x_offsets = (
        n[:, None] * x_stride_n
        + channels[:, None] * x_stride_c
        + h[None, :] * x_stride_h
        + w[None, :] * x_stride_w
    )
    residual_offsets = (
        n[:, None] * residual_stride_n
        + channels[:, None] * residual_stride_c
        + h[None, :] * residual_stride_h
        + w[None, :] * residual_stride_w
    )
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = _f32_add(x, residual).to(tl.bfloat16)
    spatial_sum = tl.sum(tl.where(mask, add_bf16.to(tl.float32), 0.0), axis=1)
    pooled_bf16 = _f32_div(spatial_sum, 49.0).to(tl.bfloat16)
    tl.store(pooled_ptr + rows, pooled_bf16, mask=row_mask)

    pooled_f32 = pooled_bf16.to(tl.float32)
    valid_pooled = tl.where(row_mask, pooled_f32, 0.0)
    partial_sum = tl.sum(valid_pooled, axis=0)
    partial_sumsq = tl.sum(_f32_mul(valid_pooled, valid_pooled), axis=0)
    tl.store(partials_ptr + tl.program_id(0), partial_sum)
    tl.store(partials_ptr + TOTAL_BLOCKS + tl.program_id(0), partial_sumsq)


@triton.jit
def _layernorm_affine_kernel(
    pooled_ptr,
    partials_ptr,
    weight_ptr,
    bias_ptr,
    channel_mean_ptr,
    rsqrt_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    BLOCKS_PER_BATCH: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < CHANNELS
    partial_offsets = tl.arange(0, BLOCK_P)
    partial_mask = partial_offsets < BLOCKS_PER_BATCH
    partial_base = n * BLOCKS_PER_BATCH + partial_offsets
    partial_sum = tl.load(
        partials_ptr + partial_base,
        mask=partial_mask,
        other=0.0,
    ).to(tl.float32)
    partial_sumsq = tl.load(
        partials_ptr + TOTAL_BLOCKS + partial_base,
        mask=partial_mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(partial_sum, axis=0)
    total_sumsq = tl.sum(partial_sumsq, axis=0)

    pooled_f32 = tl.load(
        pooled_ptr + n * CHANNELS + cols,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    channel_mean = _f32_div(total, 640.0)
    tl.store(channel_mean_ptr + n, channel_mean)

    centered = tl.where(col_mask, _f32_add(pooled_f32, -channel_mean), 0.0)
    mean_square = _f32_mul(channel_mean, channel_mean)
    variance = _f32_add(_f32_div(total_sumsq, 640.0), -mean_square)
    variance = tl.maximum(variance, 0.0)
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-6))
    tl.store(rsqrt_ptr + n, invstd)

    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    normed = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normed, weight), bias)
    tl.store(out_ptr + n * CHANNELS + cols, affine.to(tl.bfloat16), mask=col_mask)


def _launch(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_C: int,
    spatial_warps: int,
    row_warps: int,
    num_stages: int,
):
    x, residual, weight, bias, *_shape_params = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    total_rows = batch * channels
    total_blocks = triton.cdiv(total_rows, BLOCK_ROWS)
    blocks_per_batch = triton.cdiv(channels, BLOCK_ROWS)

    pooled = torch.empty_strided(
        (batch, 1, 1, channels),
        (channels, channels, channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    channel_mean = torch.empty_strided(
        (batch, 1, 1, 1),
        (1, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (batch, 1, 1, 1),
        (1, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (batch, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (2, total_blocks),
        (total_blocks, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _spatial_mean_kernel[(total_blocks,)](
        x,
        residual,
        pooled,
        partials,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        residual_stride_n=residual.stride(0),
        residual_stride_c=residual.stride(1),
        residual_stride_h=residual.stride(2),
        residual_stride_w=residual.stride(3),
        TOTAL_ROWS=total_rows,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        TOTAL_BLOCKS=total_blocks,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=64,
        num_warps=spatial_warps,
        num_stages=num_stages,
    )
    _layernorm_affine_kernel[(batch,)](
        pooled,
        partials,
        weight,
        bias,
        channel_mean,
        rsqrt,
        out,
        CHANNELS=channels,
        TOTAL_BLOCKS=total_blocks,
        BLOCKS_PER_BATCH=blocks_per_batch,
        BLOCK_P=16,
        BLOCK_C=BLOCK_C,
        num_warps=row_warps,
        num_stages=num_stages,
    )
    return pooled, channel_mean, rsqrt, out


# c50453fe: (T([128,640,7,7], bf16, stride=(31360,1,4480,640)), T([128,640,7,7], bf16, stride=(31360,1,4480,640)), T([640], f32), T([640], f32))
@oracle_impl(
    hardware="B200",
    point="c50453fe",
    BLOCK_ROWS=64,
    BLOCK_C=1024,
    spatial_warps=2,
    row_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_ROWS: int,
    BLOCK_C: int,
    spatial_warps: int,
    row_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_C=BLOCK_C,
        spatial_warps=spatial_warps,
        row_warps=row_warps,
        num_stages=num_stages,
    )
