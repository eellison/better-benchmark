"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BN-backward-style scope by using Inductor's fused bf16-input f32 sliced-add/scalar-where producer, cooperatively reducing the two compatible per-channel summaries across batch/spatial chunks, co-finalizing the mean and variance correction terms, and emitting the dependent channels-last bf16 dense gradient tensor plus both f32 side vectors; Inductor schedules the producer, sibling reductions, vector finalization, and broadcast epilogue as generic reduction/pointwise regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen lacks a cooperative channel split-K template that shares the producer and finalized summaries across the reductions and dependent epilogue while preserving the output stride and final bf16 store; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward multi-output reduction plan that split-reduces compatible channel summaries, co-finalizes them, and sinks the full dense epilogue into the destination layout."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C = 120
IN_C = 240
H = 28
W = 28
HW = H * W
R = N * HW
NUMEL = N * C * HW
REDUCE_SCALE = 2.4912308673469386e-06
REDUCE_CHUNK_R = 256
REDUCE_CHUNKS = triton.cdiv(R, REDUCE_CHUNK_R)
REDUCE_TILE_R = 32
CHANNEL_BLOCK = 128
FINAL_BLOCK_CHUNKS = 2048
EPILOGUE_BLOCK = 2048


@triton.jit
def _f32_add(a, b):
    return a + b


@triton.jit
def _f32_sub(a, b):
    return a - b


@triton.jit
def _f32_mul(a, b):
    return a * b


@triton.jit
def _partial_dual_reduce_kernel(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    centered_src_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_prod_ptr,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
    R_: tl.constexpr,
    CHUNK_R_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    chunk = tl.program_id(0)
    rows_in_tile = tl.arange(0, BLOCK_R)
    channels = tl.arange(0, BLOCK_C)
    channel_mask = channels < C_
    acc_where = tl.full((BLOCK_C,), 0.0, tl.float32)
    acc_prod = tl.full((BLOCK_C,), 0.0, tl.float32)
    fill_value = tl.load(fill_ptr).to(tl.float32)
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

    for base in tl.range(0, CHUNK_R_, BLOCK_R):
        rows = chunk * CHUNK_R_ + base + rows_in_tile
        active = (rows[:, None] < R_) & channel_mask[None, :]
        narrow_offset = rows[:, None] * C_ + channels[None, :]
        wide_offset = rows[:, None] * IN_C_ + channels[None, :]

        lhs = tl.load(wide_ptr + wide_offset, mask=active, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + narrow_offset, mask=active, other=0.0).to(tl.float32)
        add_value = _f32_add(lhs, rhs)
        mask_value = tl.load(mask_ptr + narrow_offset, mask=active, other=0.0)
        where_value = tl.where(mask_value <= 0.0, fill_value, add_value)

        src = tl.load(centered_src_ptr + narrow_offset, mask=active, other=0.0).to(
            tl.float32
        )
        centered = _f32_sub(src, mean[None, :])
        prod = _f32_mul(where_value, centered)
        acc_where = _f32_add(
            acc_where,
            tl.sum(tl.where(active, where_value, 0.0), axis=0),
        )
        acc_prod = _f32_add(
            acc_prod,
            tl.sum(tl.where(active, prod, 0.0), axis=0),
        )

    partial_offset = chunk * C_ + channels
    tl.store(partial_where_ptr + partial_offset, acc_where, mask=channel_mask)
    tl.store(partial_prod_ptr + partial_offset, acc_prod, mask=channel_mask)


@triton.jit
def _finalize_kernel(
    partial_where_ptr,
    partial_prod_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_ptr,
    out_vec_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    C_: tl.constexpr,
    NUM_CHUNKS_: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
):
    channel = tl.program_id(0)
    chunks = tl.arange(0, BLOCK_CHUNKS)
    mask = chunks < NUM_CHUNKS_
    offsets = chunks * C_ + channel
    where_values = tl.load(partial_where_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    prod_values = tl.load(partial_prod_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    sum_where = tl.sum(where_values, axis=0).to(tl.float32)
    sum_prod = tl.sum(prod_values, axis=0).to(tl.float32)
    invstd = tl.load(invstd_ptr + channel).to(tl.float32)
    weight = tl.load(weight_ptr + channel).to(tl.float32)

    mean_term = _f32_mul(sum_where, REDUCE_SCALE_)
    prod_scaled = _f32_mul(sum_prod, REDUCE_SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    prod_coeff = _f32_mul(prod_scaled, invstd_sq)
    scale = _f32_mul(invstd, weight)

    tl.store(sum_where_ptr + channel, sum_where)
    tl.store(out_vec_ptr + channel, _f32_mul(sum_prod, invstd))
    tl.store(mean_term_ptr + channel, mean_term)
    tl.store(prod_coeff_ptr + channel, prod_coeff)
    tl.store(output_scale_ptr + channel, scale)


@triton.jit
def _epilogue_kernel(
    wide_ptr,
    rhs_ptr,
    mask_ptr,
    fill_ptr,
    centered_src_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    IN_C_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = xindex < NUMEL_
    channel = xindex % C_
    row = xindex // C_
    wide_offset = row * IN_C_ + channel

    lhs = tl.load(wide_ptr + wide_offset, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + xindex, mask=active, other=0.0).to(tl.float32)
    add_value = _f32_add(lhs, rhs).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    mask_value = tl.load(mask_ptr + xindex, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    where_value = tl.where(mask_value <= 0.0, fill_value, add_value).to(tl.float32)

    src = tl.load(centered_src_ptr + xindex, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + channel, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + channel, mask=active, other=0.0).to(tl.float32)
    prod_coeff = tl.load(prod_coeff_ptr + channel, mask=active, other=0.0).to(
        tl.float32
    )
    output_scale = tl.load(output_scale_ptr + channel, mask=active, other=0.0).to(
        tl.float32
    )

    centered = _f32_sub(src, mean)
    correction = _f32_mul(centered, prod_coeff)
    after_variance = _f32_sub(where_value, correction)
    after_mean = _f32_sub(after_variance, mean_term)
    grad = _f32_mul(after_mean, output_scale)
    tl.store(
        out_ptr + xindex,
        grad.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


# 11c59fce: GhostNet train bf16 NHWC BN-backward fragment.
@oracle_impl(hardware="B200", point="11c59fce")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
    ) = inputs

    device = arg0_1.device
    partial_where = torch.empty(
        (REDUCE_CHUNKS, C),
        device=device,
        dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_where)
    sum_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_vec = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_like(sum_where)
    prod_coeff = torch.empty_like(sum_where)
    output_scale = torch.empty_like(sum_where)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )

    _partial_dual_reduce_kernel[(REDUCE_CHUNKS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partial_where,
        partial_prod,
        C_=C,
        IN_C_=IN_C,
        R_=R,
        CHUNK_R_=REDUCE_CHUNK_R,
        BLOCK_R=REDUCE_TILE_R,
        BLOCK_C=CHANNEL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_where,
        partial_prod,
        arg6_1,
        arg7_1,
        sum_where,
        out_vec,
        mean_term,
        prod_coeff,
        output_scale,
        C_=C,
        NUM_CHUNKS_=REDUCE_CHUNKS,
        BLOCK_CHUNKS=FINAL_BLOCK_CHUNKS,
        REDUCE_SCALE_=REDUCE_SCALE,
        num_warps=4,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        NUMEL_=NUMEL,
        C_=C,
        IN_C_=IN_C,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return sum_where, out_vec, out
