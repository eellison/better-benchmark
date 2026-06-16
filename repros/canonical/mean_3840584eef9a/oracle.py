"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GhostNet inference BatchNorm-affine plus spatial-mean scope in one channels-last Triton kernel, including fp32 mean/variance normalization with eps=1e-5, affine scale/bias, explicit bf16 activation rounding, the returned full bf16 activation, and the returned bf16 mean accumulated from that rounded activation, whereas Inductor lowers the affine materialization and downstream spatial mean through generic pointwise and reduction scheduling around a visible intermediate tensor; Inductor cannot do this today because its normalization/pointwise scheduler does not keep a returned bf16 producer and its immediate spatial-reduction consumer in one full-output channels-last plan while preserving the cast boundary; the fix is SCHEDULER_FUSION: teach norm-template scheduling to fuse inference BN affine stores with dependent spatial-mean reductions over the rounded activation."""

import torch
import triton
import triton.language as tl

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
def _bn_affine_spatial_mean_tile_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    spatial_mean_ptr,
    partial_ptr,
    CHANNELS: tl.constexpr,
    HW_SIZE: tl.constexpr,
    HW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    DIRECT_MEAN: tl.constexpr,
):
    n = tl.program_id(0)
    hw_block = tl.program_id(2)
    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)[:, None]
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    hw_mask = hw_offsets < HW_SIZE
    c_mask = c_offsets < CHANNELS
    mask = hw_mask & c_mask

    x_offsets = n * CHANNELS * HW_SIZE + hw_offsets * CHANNELS + c_offsets
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    var_eps = _f32_add(var, 1.0e-5)
    sqrt = tl.sqrt_rn(var_eps)
    recip = _f32_div(1.0, sqrt)
    scale = _f32_mul(recip, 1.0)
    normalized = _f32_mul(centered, scale)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    rounded = affine.to(tl.bfloat16)

    tl.store(out_ptr + x_offsets, rounded, mask=mask)

    rounded_f32 = tl.where(hw_mask, rounded.to(tl.float32), 0.0)
    total = tl.sum(rounded_f32, axis=0)
    c_1d = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_1d_mask = c_1d < CHANNELS
    if DIRECT_MEAN:
        denom = tl.full((BLOCK_C,), HW_SIZE, tl.float32)
        pooled = _f32_div(total, denom)
        tl.store(
            spatial_mean_ptr + n * CHANNELS + c_1d,
            pooled.to(tl.bfloat16),
            mask=c_1d_mask,
        )
    else:
        partial_offsets = (n * CHANNELS + c_1d) * HW_BLOCKS + hw_block
        tl.store(partial_ptr + partial_offsets, total, mask=c_1d_mask)


@triton.jit
def _finalize_spatial_mean_kernel(
    partial_ptr,
    spatial_mean_ptr,
    CHANNELS: tl.constexpr,
    HW_SIZE: tl.constexpr,
    HW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FINAL_BLOCK: tl.constexpr,
):
    n = tl.program_id(0)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    p_offsets = tl.arange(0, FINAL_BLOCK)[:, None]
    mask = (c_offsets < CHANNELS) & (p_offsets < HW_BLOCKS)
    partial_offsets = (n * CHANNELS + c_offsets) * HW_BLOCKS + p_offsets
    partial = tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(partial, axis=0)
    denom = tl.full((BLOCK_C,), HW_SIZE, tl.float32)
    pooled = _f32_div(total, denom)
    c_1d = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(
        spatial_mean_ptr + n * CHANNELS + c_1d,
        pooled.to(tl.bfloat16),
        mask=c_1d < CHANNELS,
    )


def _launch(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_HW: int,
    FINAL_BLOCK: int,
    num_warps: int,
    final_warps: int,
):
    mean, x, var, weight, bias = inputs
    n_size = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw_size = height * width

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    spatial_mean = torch.empty_strided(
        (n_size, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    hw_blocks = triton.cdiv(hw_size, BLOCK_HW)
    if hw_blocks == 1:
        partial = spatial_mean
    else:
        partial = torch.empty_strided(
            (n_size, channels, hw_blocks),
            (channels * hw_blocks, hw_blocks, 1),
            device=x.device,
            dtype=torch.float32,
        )

    _bn_affine_spatial_mean_tile_kernel[
        (n_size, triton.cdiv(channels, BLOCK_C), hw_blocks)
    ](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        spatial_mean,
        partial,
        CHANNELS=channels,
        HW_SIZE=hw_size,
        HW_BLOCKS=hw_blocks,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        DIRECT_MEAN=(hw_blocks == 1),
        num_warps=num_warps,
        num_stages=3,
    )
    if hw_blocks != 1:
        _finalize_spatial_mean_kernel[(n_size, triton.cdiv(channels, BLOCK_C))](
            partial,
            spatial_mean,
            CHANNELS=channels,
            HW_SIZE=hw_size,
            HW_BLOCKS=hw_blocks,
            BLOCK_C=BLOCK_C,
            FINAL_BLOCK=FINAL_BLOCK,
            num_warps=final_warps,
            num_stages=3,
        )
    return out, spatial_mean


# 57e42e70: (T([672], bf16), T([512,672,7,7], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="57e42e70", BLOCK_C=128, BLOCK_HW=64, FINAL_BLOCK=1, num_warps=4, final_warps=1)
# bddd3dfb: (T([72], bf16), T([512,72,28,28], bf16, channels-last), ...)
@oracle_impl(hardware="B200", point="bddd3dfb", BLOCK_C=128, BLOCK_HW=64, FINAL_BLOCK=16, num_warps=4, final_warps=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_HW: int,
    FINAL_BLOCK: int,
    num_warps: int,
    final_warps: int,
):
    return _launch(
        inputs,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        FINAL_BLOCK=FINAL_BLOCK,
        num_warps=num_warps,
        final_warps=final_warps,
    )
