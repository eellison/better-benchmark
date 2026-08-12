"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail by split-K reducing the masked `where(arg1 <= 0, arg2, arg3)` producer into both returned per-channel f32 summaries, writing the full returned bf16 dense gradient, and forming the returned last-32-channel residual add with the required bf16 cast boundary. Inductor currently schedules the sibling reductions, reduction-dependent dense BN epilogue, and final residual slice add as separate generic regions around replayed or materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope split-K multi-output plan that shares the conditional bf16 producer and keeps finalized channel summaries available to both dense and slice-limited consumers while preserving bf16/f32 cast boundaries. The fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward split-K lowering that co-finalizes compatible channel reductions and sinks the vector, dense, and static residual-slice epilogues into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 224
H = 56
W = 56
HW = H * W
K_TOTAL = N * HW
SCALE = 7.971938775510203e-05
SLICE_START = 192
SLICE_C = 32


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _nchw_offsets(c, k_offsets, C_: tl.constexpr, HW_: tl.constexpr):
    n = k_offsets // HW_
    spatial = k_offsets - n * HW_
    return n * C_ * HW_ + c * HW_ + spatial


@triton.jit
def _partial_reduce_kernel(
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    CHUNK_K: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    lanes = tl.arange(0, R_BLOCK)
    fill = tl.load(fill_ptr).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    acc_dot = tl.full([R_BLOCK], 0.0, tl.float32)
    acc_sum = tl.full([R_BLOCK], 0.0, tl.float32)
    for r_offset in tl.range(0, CHUNK_K, R_BLOCK):
        local = r_offset + lanes
        k_offsets = block * CHUNK_K + local
        active = (local < CHUNK_K) & (k_offsets < K_TOTAL_)
        offsets = _nchw_offsets(c, k_offsets, C_, HW_)

        mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0)
        source = tl.load(source_ptr + offsets, mask=active, other=0.0)
        grad = tl.where(mask_value <= 0.0, fill, source).to(tl.float32)

        centered_source = tl.load(
            centered_source_ptr + offsets, mask=active, other=0.0
        ).to(tl.float32)
        centered = _f32_sub(centered_source, mean)
        prod = _f32_mul(grad, centered)
        acc_dot = tl.where(active, _f32_add(acc_dot, prod), acc_dot)
        acc_sum = tl.where(active, _f32_add(acc_sum, grad), acc_sum)

    tl.store(partial_sum_ptr + block * C_ + c, tl.sum(acc_sum, axis=0))
    tl.store(partial_dot_ptr + block * C_ + c, tl.sum(acc_dot, axis=0))


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scale_grad_ptr,
    C_: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    c = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_BLOCKS)
    active = blocks < NUM_BLOCKS
    offsets = blocks * C_ + c
    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    dense_out_ptr,
    add_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    SCALE_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    RESIDUAL_C_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_K)
    k_offsets = block * CHUNK_K + lanes
    active = (lanes < CHUNK_K) & (k_offsets < K_TOTAL_)
    offsets = _nchw_offsets(c, k_offsets, C_, HW_)

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    grad = tl.where(mask_value <= 0.0, fill, source).to(tl.float32)

    centered_source = tl.load(
        centered_source_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    sum_value = tl.load(sum_ptr + c).to(tl.float32)
    dot_value = tl.load(dot_ptr + c).to(tl.float32)

    centered = _f32_sub(centered_source, mean)
    dot_scaled = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    corrected = _f32_sub(grad, _f32_mul(centered, variance_term))
    mean_term = _f32_mul(sum_value, SCALE_)
    centered_grad = _f32_sub(corrected, mean_term)
    output_scale = _f32_mul(invstd, weight)
    dense_bf16 = _f32_mul(centered_grad, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)

    in_slice = c >= SLICE_START_
    n = k_offsets // HW_
    spatial = k_offsets - n * HW_
    slice_c = c - SLICE_START_
    add_mask = active & in_slice
    residual_offsets = n * (RESIDUAL_C_ * HW_) + c * HW_ + spatial
    add_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + spatial
    residual = tl.load(residual_ptr + residual_offsets, mask=add_mask, other=0.0)
    add_value = _bf16_add(residual, dense_bf16)
    tl.store(add_out_ptr + add_offsets, add_value, mask=add_mask)


@oracle_impl(
    hardware="B200",
    point="e6a7fabb",
    BLOCK_K=8192,
    CHUNK_K=6272,
    R_BLOCK=2048,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_K: int, CHUNK_K: int, R_BLOCK: int, num_warps: int):
    (
        residual,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
    ) = inputs
    device = source.device
    num_blocks = triton.cdiv(K_TOTAL, CHUNK_K)
    block_blocks = _ceil_pow2(num_blocks)

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    grid = (C, num_blocks)
    _partial_reduce_kernel[grid](
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        partial_sum,
        partial_dot,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        CHUNK_K=CHUNK_K,
        R_BLOCK=R_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        invstd,
        sum_out,
        dot_tmp,
        scale_grad,
        C_=C,
        NUM_BLOCKS=num_blocks,
        BLOCK_BLOCKS=block_blocks,
        num_warps=8,
        num_stages=3,
    )
    _epilogue_kernel[grid](
        residual,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
        sum_out,
        dot_tmp,
        dense_out,
        add_out,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        SCALE_=SCALE,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        RESIDUAL_C_=256,
        BLOCK_K=BLOCK_K,
        CHUNK_K=CHUNK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, add_out
