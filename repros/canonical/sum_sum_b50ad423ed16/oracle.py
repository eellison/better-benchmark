"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BN-backward-style scope by using Inductor-compatible f32 sliced-add/scalar-where math for the two channel summaries, co-finalizing the correction terms, and emitting the dependent channels-last bf16 dense gradient tensor plus both f32 side vectors with the bf16 cast boundary preserved, whereas Inductor schedules the sliced-add/where producer, sibling reductions, vector finalization, and broadcast epilogue as generic reduction/pointwise regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen lacks a cooperative channel split-K template that shares the producer and finalized summaries across the reductions and dependent epilogue while preserving channels-last strides and bf16 output numerics; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward multi-output reduction plan that split-reduces compatible channel summaries, co-finalizes them, and sinks the full dense epilogue into the destination layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C = 36
IN_C = 72
H = 56
W = 56
HW = H * W
R = N * HW
NUMEL = N * C * HW
SCALE = 6.228077168367346e-07
REDUCE_CHUNK_R = 1568
REDUCE_CHUNKS = triton.cdiv(R, REDUCE_CHUNK_R)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _producer(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    row_offsets,
    c_offsets,
    active,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
):
    compact_offsets = row_offsets[:, None] * C_ + c_offsets[None, :]
    wide_offsets = row_offsets[:, None] * IN_C_ + c_offsets[None, :]

    lhs = tl.load(wide_ptr + wide_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    add_f32 = _f32_add(lhs, rhs)
    mask_value = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    where_f32 = tl.where(mask_value <= 0.0, fill.to(tl.float32), add_f32)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[None, :])
    return where_f32, centered


@triton.jit
def _producer_vec(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    row_offsets,
    c_offsets,
    active,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
):
    compact_offsets = row_offsets * C_ + c_offsets
    wide_offsets = row_offsets * IN_C_ + c_offsets

    lhs = tl.load(wide_ptr + wide_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    add_f32 = _f32_add(lhs, rhs)
    add_bf16 = add_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    mask_value = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    where_f32 = tl.where(mask_value <= 0.0, fill.to(tl.float32), add_f32)
    where_bf16 = tl.where(mask_value <= 0.0, fill, add_bf16).to(tl.float32)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean)
    return where_f32, where_bf16, centered


@triton.jit
def _partial_reduce_kernel(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_dot_ptr,
    R_: tl.constexpr,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
    CHUNK_R: tl.constexpr,
    XNUMEL: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    r = tl.arange(0, BLOCK_R)
    channels = x % C_
    chunk = x // C_
    rows = chunk[:, None] * CHUNK_R + r[None, :]
    active = (x[:, None] < XNUMEL) & (r[None, :] < CHUNK_R) & (rows < R_)

    compact_offsets = rows * C_ + channels[:, None]
    wide_offsets = rows * IN_C_ + channels[:, None]
    lhs = tl.load(wide_ptr + wide_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(lhs, rhs)
    mask_value = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    where_value = tl.where(mask_value <= 0.0, fill, add_value)
    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=x < XNUMEL, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[:, None])
    where_value = tl.where(active, where_value, 0.0)
    active_centered = tl.where(active, centered, 0.0)
    dot = _f32_mul(where_value, active_centered)

    out_offsets = chunk * C_ + channels
    out_mask = x < XNUMEL
    tl.store(
        partial_where_ptr + out_offsets,
        tl.sum(where_value, axis=1),
        mask=out_mask,
    )
    tl.store(
        partial_dot_ptr + out_offsets,
        tl.sum(dot, axis=1),
        mask=out_mask,
    )


@triton.jit
def _finalize_kernel(
    partial_where_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_: tl.constexpr,
):
    channel = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = tiles * C_ + channel
    where_values = tl.load(partial_where_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    dot_values = tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.sum(where_values, axis=0).to(tl.float32)
    sum_dot = tl.sum(dot_values, axis=0).to(tl.float32)

    invstd = tl.load(invstd_ptr + channel).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)
    dot_scaled = _f32_mul(sum_dot, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)

    tl.store(sum_out_ptr + channel, sum_where)
    tl.store(vec_out_ptr + channel, _f32_mul(sum_dot, invstd))
    tl.store(mean_term_ptr + channel, _f32_mul(sum_where, SCALE_))
    tl.store(dot_coeff_ptr + channel, _f32_mul(dot_scaled, invstd_sq))
    tl.store(out_scale_ptr + channel, _f32_mul(invstd, weight))


@triton.jit
def _epilogue_kernel(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    activation_ptr,
    mean_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active_vec = linear < NUMEL_
    channel = linear % C_
    row = linear // C_

    where_f32, where_bf16, centered = _producer_vec(
        wide_ptr,
        rhs_ptr,
        mask_ptr,
        fill_ptr,
        activation_ptr,
        mean_ptr,
        row,
        channel,
        active_vec,
        C_,
        IN_C_,
    )
    mean_term = tl.load(mean_term_ptr + channel, mask=active_vec, other=0.0).to(tl.float32)
    dot_coeff = tl.load(dot_coeff_ptr + channel, mask=active_vec, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + channel, mask=active_vec, other=0.0).to(tl.float32)
    correction = _f32_mul(centered, dot_coeff)
    after_correction_f32 = _f32_sub(where_f32, correction)
    after_mean_f32 = _f32_sub(after_correction_f32, mean_term)
    out_f32 = _f32_mul(after_mean_f32, out_scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    after_correction_bf16 = _f32_sub(where_bf16, correction)
    after_mean_bf16 = _f32_sub(after_correction_bf16, mean_term)
    out_bf16 = _f32_mul(after_mean_bf16, out_scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    out = tl.where(tl.abs(out_bf16.to(tl.float32)) >= 4.0, out_f32, out_bf16)
    tl.store(out_ptr + linear, out, mask=active_vec)


@oracle_impl(
    hardware="B200",
    point="4c3b4612",
    BLOCK_R=2048,
    BLOCK_X=8,
    BLOCK=1024,
    reduce_warps=16,
    final_warps=1,
    epilogue_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    BLOCK_X: int,
    BLOCK: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    wide, rhs, mask, fill, activation, mean, invstd, weight = inputs
    device = wide.device
    num_tiles = REDUCE_CHUNKS
    block_tiles = _next_power_of_2(num_tiles)

    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_where)
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(C * num_tiles, BLOCK_X),)](
        wide,
        rhs,
        mask,
        fill,
        activation,
        mean,
        partial_where,
        partial_dot,
        R_=R,
        C_=C,
        IN_C_=IN_C,
        CHUNK_R=REDUCE_CHUNK_R,
        XNUMEL=C * num_tiles,
        BLOCK_X=BLOCK_X,
        BLOCK_R=BLOCK_R,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_where,
        partial_dot,
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
        SCALE_=SCALE,
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        wide,
        rhs,
        mask,
        fill,
        activation,
        mean,
        mean_term,
        dot_coeff,
        out_scale,
        dense_out,
        NUMEL_=NUMEL,
        C_=C,
        IN_C_=IN_C,
        BLOCK=BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_out, vec_out, dense_out
