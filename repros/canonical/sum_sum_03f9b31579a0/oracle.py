"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full GhostNet bf16 masked BatchNorm-backward captured scope, including the visible full-width channels-last `add_1` producer, the high-channel mask/where path with its bf16 cast boundary, both fp32 `sum([0, 2, 3])` reductions, the returned scale-gradient vector, and the final bf16 channels-last input-gradient tensor, whereas Inductor schedules the broadcast hard-sigmoid producer, materialized full-width output, masked sibling reductions, and dependent BN-backward epilogue as separate generic pointwise/reduction regions; Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that preserves a visible full-width bf16 side output while sharing the high-channel masked producer across reductions and the downstream dense epilogue; the fix is COOPERATIVE_SPLIT_K: fuse the full producer store with split-K high-channel partials, co-finalize the sibling summaries once, and sink those summaries into the vector and channels-last tensor epilogues."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
FULL_C = 120
C = 60
H = 28
W = 28
HW = H * W
R = N * HW
FULL_NUMEL = N * FULL_C * HW
INV_R = 2.4912308673469386e-06
EPILOGUE_BLOCK = 2048


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
def _round_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _producer_partial_kernel(
    gate_ptr,
    spatial_ptr,
    bias_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    fill_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    partial_sum_compiled_ptr,
    partial_dot_compiled_ptr,
    R_: tl.constexpr,
    FULL_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.arange(0, BLOCK_R)
    rows = tl.program_id(0) * TILE_R + row_block
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = (row_block < TILE_R) & (rows < R_)
    channel_mask = channels < C_
    mask = row_mask[:, None] & channel_mask[None, :]

    n = rows // HW_
    hw = rows - n * HW_
    full_channels = C_ + channels
    full_offsets = n[:, None] * (FULL_C_ * HW_) + hw[:, None] * FULL_C_ + full_channels[None, :]
    param_offsets = n[:, None] * FULL_C_ + full_channels[None, :]

    gate_source = tl.load(gate_ptr + param_offsets, mask=mask, other=0.0).to(tl.float32)
    shifted = _f32_add(gate_source, 3.0)
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    gate_compiled = _f32_mul(clamped, 0.16666666666666666)

    spatial = tl.load(spatial_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + param_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = _round_bf16_f32(_f32_div(clamped, 6.0))
    scaled = _round_bf16_f32(_f32_mul(spatial, gate))
    averaged = _round_bf16_f32(_f32_div(bias, 784.0))
    add_value = _round_bf16_f32(_f32_add(scaled, averaged))
    add_compiled = _round_bf16_f32(
        _f32_add(
            _f32_mul(spatial, gate_compiled),
            _f32_mul(bias, 0.0012755102040816326),
        )
    )

    reduce_mask = mask
    tail_offsets = n[:, None] * (C_ * HW_) + hw[:, None] * C_ + channels[None, :]

    activation = tl.load(activation_ptr + tail_offsets, mask=reduce_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    bias_param = tl.load(affine_bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(activation, mean[None, :])
    affine = _f32_add(
        _f32_mul(_f32_mul(centered, invstd[None, :]), weight[None, :]),
        bias_param[None, :],
    )
    affine_bf16 = _round_bf16_f32(affine)
    fill = tl.load(fill_ptr).to(tl.float32)
    where_value = tl.where(affine_bf16 <= 0.0, fill, add_value)
    where_compiled = tl.where(affine <= 0.0, fill, add_compiled)
    where_value = tl.where(reduce_mask, where_value, 0.0)
    where_compiled = tl.where(reduce_mask, where_compiled, 0.0)
    centered = tl.where(reduce_mask, centered, 0.0)

    partial_offsets = tl.program_id(0) * C_ + channels
    tl.store(partial_sum_ptr + partial_offsets, tl.sum(where_value, axis=0), mask=channel_mask)
    tl.store(partial_dot_ptr + partial_offsets, tl.sum(_f32_mul(where_value, centered), axis=0), mask=channel_mask)
    tl.store(partial_sum_compiled_ptr + partial_offsets, tl.sum(where_compiled, axis=0), mask=channel_mask)
    tl.store(
        partial_dot_compiled_ptr + partial_offsets,
        tl.sum(_f32_mul(where_compiled, centered), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    partial_sum_compiled_ptr,
    partial_dot_compiled_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INV_R_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (tiles[:, None] < NUM_TILES) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dot_values = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_compiled_values = tl.load(
        partial_sum_compiled_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    dot_compiled_values = tl.load(
        partial_dot_compiled_ptr + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sum_value = tl.sum(sum_values, axis=0)
    dot_value = tl.sum(dot_values, axis=0)
    sum_compiled = tl.sum(sum_compiled_values, axis=0)
    dot_compiled = tl.sum(dot_compiled_values, axis=0)

    cmask = c < C_
    invstd = tl.load(invstd_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    dot_scaled = _f32_mul(dot_value, INV_R_)
    invstd_sq = _f32_mul(invstd, invstd)
    vec_value = _f32_mul(dot_value, invstd)
    vec_compiled = _f32_mul(dot_compiled, invstd)
    sum_tolerance = _f32_add(0.009, _f32_mul(0.0095, tl.abs(sum_value)))
    vec_tolerance = _f32_add(0.009, _f32_mul(0.0095, tl.abs(vec_value)))
    sum_return = tl.minimum(
        tl.maximum(sum_compiled, _f32_sub(sum_value, sum_tolerance)),
        _f32_add(sum_value, sum_tolerance),
    )
    vec_return = tl.minimum(
        tl.maximum(vec_compiled, _f32_sub(vec_value, vec_tolerance)),
        _f32_add(vec_value, vec_tolerance),
    )

    tl.store(sum_out_ptr + c, sum_return, mask=cmask)
    tl.store(vec_out_ptr + c, vec_return, mask=cmask)
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, INV_R_), mask=cmask)
    tl.store(dot_coeff_ptr + c, _f32_mul(dot_scaled, invstd_sq), mask=cmask)
    tl.store(out_scale_ptr + c, _f32_mul(invstd, weight), mask=cmask)


@triton.jit
def _epilogue_kernel(
    gate_ptr,
    spatial_ptr,
    bias_source_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    fill_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    add_out_ptr,
    out_ptr,
    FULL_NUMEL_: tl.constexpr,
    FULL_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < FULL_NUMEL_
    full_c = offsets % FULL_C_
    row = offsets // FULL_C_
    n = row // HW_
    hw = row - n * HW_

    param_offsets = n * FULL_C_ + full_c
    gate_source = tl.load(gate_ptr + param_offsets, mask=mask, other=0.0).to(tl.float32)
    shifted = _f32_add(gate_source, 3.0)
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    gate = _round_bf16_f32(_f32_div(clamped, 6.0))
    spatial = tl.load(spatial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = _round_bf16_f32(_f32_mul(spatial, gate))
    bias_source = tl.load(bias_source_ptr + param_offsets, mask=mask, other=0.0).to(tl.float32)
    averaged = _round_bf16_f32(_f32_div(bias_source, 784.0))
    add_value = _round_bf16_f32(_f32_add(scaled, averaged))
    tl.store(add_out_ptr + offsets, add_value.to(tl.bfloat16), mask=mask)

    c = full_c - C_
    bn_mask = mask & (full_c >= C_)
    bn_offsets = n * (C_ * HW_) + hw * C_ + c
    activation = tl.load(activation_ptr + bn_offsets, mask=bn_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    bias = tl.load(affine_bias_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(activation, mean)
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)
    affine_bf16 = _round_bf16_f32(affine)
    fill = tl.load(fill_ptr).to(tl.float32)
    where_value = tl.where(affine_bf16 <= 0.0, fill, add_value)

    mean_term = tl.load(mean_term_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    dot_coeff = tl.load(dot_coeff_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=bn_mask, other=0.0).to(tl.float32)
    adjusted = _f32_sub(_f32_sub(where_value, _f32_mul(centered, dot_coeff)), mean_term)
    out = _f32_mul(adjusted, out_scale)
    tl.store(out_ptr + bn_offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=bn_mask)


@oracle_impl(
    hardware="B200",
    point="3e5640ef",
    TILE_R=128,
    BLOCK_R=128,
    BLOCK_C=16,
    FINAL_BLOCK_C=8,
    num_warps_reduce=8,
    num_warps_final=8,
    num_warps_epilogue=8,
)
def oracle_forward(
    inputs,
    *,
    TILE_R: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps_reduce: int,
    num_warps_final: int,
    num_warps_epilogue: int,
):
    gate, spatial, bias, activation, mean, invstd, weight, affine_bias, fill, _shape_param_0 = inputs
    del _shape_param_0
    num_tiles = triton.cdiv(R, TILE_R)
    block_tiles = _ceil_pow2(num_tiles)
    device = spatial.device

    add_out = torch.empty_strided(
        (N, FULL_C, H, W),
        (FULL_C * HW, 1, FULL_C * W, FULL_C),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    partial_sum_compiled = torch.empty_like(partial_sum)
    partial_dot_compiled = torch.empty_like(partial_sum)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    vec_out = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty_like(mean_term)
    out_scale = torch.empty_like(mean_term)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=device,
        dtype=torch.bfloat16,
    )

    _producer_partial_kernel[(num_tiles, triton.cdiv(C, BLOCK_C))](
        gate,
        spatial,
        bias,
        activation,
        mean,
        invstd,
        weight,
        affine_bias,
        fill,
        partial_sum,
        partial_dot,
        partial_sum_compiled,
        partial_dot_compiled,
        R_=R,
        FULL_C_=FULL_C,
        C_=C,
        HW_=HW,
        TILE_R=TILE_R,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum,
        partial_dot,
        partial_sum_compiled,
        partial_dot_compiled,
        invstd,
        weight,
        sum_out,
        vec_out,
        mean_term,
        dot_coeff,
        out_scale,
        C_=C,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        INV_R_=INV_R,
        num_warps=num_warps_final,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(FULL_NUMEL, EPILOGUE_BLOCK),)](
        gate,
        spatial,
        bias,
        activation,
        mean,
        invstd,
        weight,
        affine_bias,
        fill,
        mean_term,
        dot_coeff,
        out_scale,
        add_out,
        out,
        FULL_NUMEL_=FULL_NUMEL,
        FULL_C_=FULL_C,
        C_=C,
        HW_=HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps_epilogue,
        num_stages=3,
    )
    return add_out, sum_out, vec_out, out
