"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BatchNorm-backward return scope by preserving the high-half channels-last slice, the bf16 ReLU-gate cast boundary, the shared masked producer, both fp32 `sum([0, 2, 3])` reductions, the returned scale-gradient vector, and the final bf16 channels-last dense gradient tensor; whereas Inductor schedules the slice/gate producer, sibling channel reductions, and reduction-dependent dense epilogue as separate generic pointwise/reduction regions. Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K multi-output BN-backward template that coordinates a bf16 gated producer, compatible per-channel summaries, vector side output, and dense epilogue while preserving channels-last strides and cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded split-K channel-reduction lowering that co-finalizes the summaries once and sinks the vector and dense epilogues into the same full-scope plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
FULL_C = 960
C = 480
H = 7
W = 7
HW = H * W
R = N * HW
OUT_NUMEL = N * C * HW
INV_R = 3.985969387755102e-05
SLICE_START = 480
EPILOGUE_BLOCK = 1024


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
def _partial_reduce_kernel(
    slice_base_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    fill_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    R_: tl.constexpr,
    FULL_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (rows[:, None] < R_) & (channels[None, :] < C_)

    n = rows // HW_
    hw = rows - n * HW_
    h = hw // W_
    w = hw - h * W_
    source_offsets = (
        n[:, None] * (FULL_C_ * HW_)
        + h[:, None] * (W_ * FULL_C_)
        + w[:, None] * FULL_C_
        + SLICE_START_
        + channels[None, :]
    )
    compact_offsets = (
        n[:, None] * (C_ * HW_)
        + h[:, None] * (W_ * C_)
        + w[:, None] * C_
        + channels[None, :]
    )

    source = tl.load(slice_base_ptr + source_offsets, mask=active, other=0.0)
    x = tl.load(bn_input_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    bias = tl.load(affine_bias_ptr + channels, mask=channels < C_, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr)

    centered = _f32_sub(x, mean[None, :])
    affine = _f32_add(
        _f32_mul(_f32_mul(centered, invstd[None, :]), weight[None, :]),
        bias[None, :],
    )
    affine_bf16 = _round_bf16_f32(affine)
    selected_bf16 = tl.where(affine_bf16 <= 0.0, fill, source)
    selected = selected_bf16.to(tl.float32)
    selected = tl.where(active, selected, 0.0)
    centered = tl.where(active, centered, 0.0)

    partial_offsets = tl.program_id(0) * C_ + channels
    channel_mask = channels < C_
    tl.store(partial_sum_ptr + partial_offsets, tl.sum(selected, axis=0), mask=channel_mask)
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(_f32_mul(selected, centered), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
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
    sum_value = tl.sum(sum_values, axis=0)
    dot_value = tl.sum(dot_values, axis=0)

    cmask = c < C_
    invstd = tl.load(invstd_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + c, mask=cmask, other=0.0).to(tl.float32)
    dot_scaled = _f32_mul(dot_value, INV_R_)
    invstd_sq = _f32_mul(invstd, invstd)

    tl.store(sum_out_ptr + c, sum_value, mask=cmask)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd), mask=cmask)
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, INV_R_), mask=cmask)
    tl.store(dot_coeff_ptr + c, _f32_mul(dot_scaled, invstd_sq), mask=cmask)
    tl.store(out_scale_ptr + c, _f32_mul(invstd, weight), mask=cmask)


@triton.jit
def _epilogue_kernel(
    slice_base_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    fill_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    dense_out_ptr,
    OUT_NUMEL_: tl.constexpr,
    FULL_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < OUT_NUMEL_
    c = linear % C_
    row = linear // C_
    n = row // HW_
    hw = row - n * HW_
    h = hw // W_
    w = hw - h * W_

    source_offsets = n * (FULL_C_ * HW_) + h * (W_ * FULL_C_) + w * FULL_C_ + SLICE_START_ + c
    source = tl.load(slice_base_ptr + source_offsets, mask=active, other=0.0)
    x = tl.load(bn_input_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(affine_bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr)

    centered = _f32_sub(x, mean)
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)
    affine_bf16 = _round_bf16_f32(affine)
    selected = tl.where(affine_bf16 <= 0.0, fill, source).to(tl.float32)

    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_coeff = tl.load(dot_coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    adjusted = _f32_sub(_f32_sub(selected, _f32_mul(centered, dot_coeff)), mean_term)
    out = _f32_mul(adjusted, out_scale)
    tl.store(dense_out_ptr + linear, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


@oracle_impl(
    hardware="B200",
    point="04b8f85c",
    BLOCK_R=512,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    slice_base, bn_input, mean, invstd, weight, bias, fill = inputs
    num_tiles = triton.cdiv(R, BLOCK_R)
    block_tiles = _ceil_pow2(num_tiles)
    device = bn_input.device

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_out = torch.empty_like(sum_out)
    scale_grad = torch.empty_like(sum_out)
    mean_term = torch.empty_like(sum_out)
    dot_coeff = torch.empty_like(sum_out)
    out_scale = torch.empty_like(sum_out)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    _partial_reduce_kernel[(num_tiles, triton.cdiv(C, BLOCK_C))](
        slice_base,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        partial_sum,
        partial_dot,
        R_=R,
        FULL_C_=FULL_C,
        C_=C,
        HW_=HW,
        W_=W,
        SLICE_START_=SLICE_START,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scale_grad,
        mean_term,
        dot_coeff,
        out_scale,
        C_=C,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        INV_R_=INV_R,
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(OUT_NUMEL, EPILOGUE_BLOCK),)](
        slice_base,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        mean_term,
        dot_coeff,
        out_scale,
        dense_out,
        OUT_NUMEL_=OUT_NUMEL,
        FULL_C_=FULL_C,
        C_=C,
        HW_=HW,
        W_=W,
        SLICE_START_=SLICE_START,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out
