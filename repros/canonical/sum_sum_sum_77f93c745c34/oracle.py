"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Inception bf16 scatter/expand-fed six-branch batch-norm-backward scope by direct-gathering each branch's `arg0[n, source_channel] / 64` producer, applying the captured affine/ReLU mask boundary, reducing each channel over `(N,H,W)`, and writing the returned scalar zero, fp32 summary vectors, and channels-last bf16 gradient tensors, whereas Inductor materializes the as_strided_scatter/expand/channel-slice producer and schedules the sibling masked reductions and dependent BN-backward epilogues as generic regions; Inductor cannot do this today because its scheduler/codegen has no scatter-gather reduction template that recognizes the structured as_strided_scatter/expand source and shares finalized per-channel summaries with multiple branch epilogues; the fix is SCATTER_REDUCE: teach Inductor to canonicalize this scatter/expand/div source into direct indexed loads and fuse the compatible masked channel reductions with their full-tensor epilogues for the complete return tuple."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
H = 8
W = 8
HW = H * W
K_TOTAL = N * HW
MM_C = 2048
SCALE = 0.0001220703125


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _branch_partial_kernel(
    source_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C_: tl.constexpr,
    SRC_OFFSET_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    MM_C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c = c_vec[:, None]
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    active = (c_vec[:, None] < C_) & (k < K_TOTAL_)
    n = k // HW_
    hw = k - n * HW_
    offset = n * (C_ * HW_) + hw * C_ + c
    source = tl.load(source_ptr + n * MM_C_ + SRC_OFFSET_ + c, mask=active, other=0.0).to(tl.float32) * 0.015625
    x = tl.load(x_ptr + offset, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    centered = x - mean
    affine = centered * invstd * weight + bias
    where_value = tl.where(affine <= 0.0, 0.0, source).to(tl.float32)
    where_active = tl.where(active, where_value, 0.0)
    centered_active = tl.where(active, centered, 0.0)
    tl.store(partial_sum_ptr + tile * C_ + c_vec, tl.sum(where_active, axis=1), mask=c_vec < C_)
    tl.store(partial_dot_ptr + tile * C_ + c_vec, tl.sum(where_active * centered_active, axis=1), mask=c_vec < C_)


@triton.jit
def _branch_finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    zero_ptr,
    sum_out_ptr,
    dot_out_ptr,
    mean_term_ptr,
    coeff_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_: tl.constexpr,
    STORE_ZERO_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)[None, :]
    mask = c_vec < C_
    offsets = tiles * C_ + c_vec[:, None]
    tile_mask = tiles < NUM_TILES_
    sum_value = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask[:, None] & tile_mask, other=0.0).to(tl.float32), axis=1)
    dot_value = tl.sum(tl.load(partial_dot_ptr + offsets, mask=mask[:, None] & tile_mask, other=0.0).to(tl.float32), axis=1)
    invstd = tl.load(invstd_ptr + c_vec, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_vec, mask=mask, other=0.0).to(tl.float32)
    if STORE_ZERO_:
        tl.store(zero_ptr, 0.0, mask=tl.program_id(0) == 0)
    tl.store(sum_out_ptr + c_vec, sum_value, mask=mask)
    tl.store(dot_out_ptr + c_vec, dot_value * invstd, mask=mask)
    tl.store(mean_term_ptr + c_vec, sum_value * SCALE_, mask=mask)
    tl.store(coeff_ptr + c_vec, (dot_value * SCALE_) * (invstd * invstd), mask=mask)
    tl.store(out_scale_ptr + c_vec, invstd * weight, mask=mask)


@triton.jit
def _branch_epilogue_kernel(
    source_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    mean_term_ptr,
    coeff_ptr,
    out_scale_ptr,
    grad_ptr,
    C_: tl.constexpr,
    SRC_OFFSET_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    MM_C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c = c_vec[:, None]
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
    active = (c_vec[:, None] < C_) & (k < K_TOTAL_)
    n = k // HW_
    hw = k - n * HW_
    offset = n * (C_ * HW_) + hw * C_ + c
    source = tl.load(source_ptr + n * MM_C_ + SRC_OFFSET_ + c, mask=active, other=0.0).to(tl.float32) * 0.015625
    x = tl.load(x_ptr + offset, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    coeff = tl.load(coeff_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=c_vec[:, None] < C_, other=0.0).to(tl.float32)
    centered = x - mean
    affine = centered * invstd * weight + bias
    where_value = tl.where(affine <= 0.0, 0.0, source).to(tl.float32)
    grad = (where_value - centered * coeff - mean_term) * out_scale
    tl.store(grad_ptr + offset, grad, mask=active)


def _empty_cl(shape, device):
    n, c, h, w = shape
    return torch.empty_strided(shape, (c * h * w, 1, w * c, c), device=device, dtype=torch.bfloat16)


def _run_branch(
    source,
    x,
    mean,
    invstd,
    weight,
    bias,
    zero,
    c,
    src_offset,
    store_zero,
    *,
    block_c,
    block_k,
    finalize_block_c,
    num_warps,
    num_stages,
):
    device = source.device
    num_tiles = triton.cdiv(K_TOTAL, block_k)
    block_tiles = _next_power_of_2(num_tiles)
    partial_sum = torch.empty((num_tiles, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, c), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_out = torch.empty((c,), device=device, dtype=torch.float32)
    mean_term = torch.empty((c,), device=device, dtype=torch.float32)
    coeff = torch.empty((c,), device=device, dtype=torch.float32)
    out_scale = torch.empty((c,), device=device, dtype=torch.float32)
    grad = _empty_cl((N, c, H, W), device)

    grid = (triton.cdiv(c, block_c), num_tiles)
    _branch_partial_kernel[grid](
        source,
        x,
        mean,
        invstd,
        weight,
        bias,
        partial_sum,
        partial_dot,
        C_=c,
        SRC_OFFSET_=src_offset,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        MM_C_=MM_C,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _branch_finalize_kernel[(triton.cdiv(c, finalize_block_c),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        zero,
        sum_out,
        dot_out,
        mean_term,
        coeff,
        out_scale,
        C_=c,
        NUM_TILES_=num_tiles,
        BLOCK_TILES=block_tiles,
        SCALE_=SCALE,
        STORE_ZERO_=store_zero,
        BLOCK_C=finalize_block_c,
        num_warps=1,
        num_stages=1,
    )
    _branch_epilogue_kernel[grid](
        source,
        x,
        mean,
        invstd,
        weight,
        bias,
        mean_term,
        coeff,
        out_scale,
        grad,
        C_=c,
        SRC_OFFSET_=src_offset,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        MM_C_=MM_C,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return sum_out, dot_out, grad


# 1de9bf8b: Inception six-branch scatter/expand masked BN-backward, bf16 channels-last.
@oracle_impl(
    hardware="B200",
    point="1de9bf8b",
    block_c=32,
    block_k=256,
    finalize_block_c=16,
    num_warps=4,
    num_stages=4,
)
def oracle_forward(
    inputs,
    *,
    block_c: int,
    block_k: int,
    finalize_block_c: int,
    num_warps: int,
    num_stages: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        arg29_1,
        arg30_1,
        *_shape_params,
    ) = inputs
    zero = torch.empty((), device=arg0_1.device, dtype=torch.bfloat16)
    kwargs = {
        "block_c": block_c,
        "block_k": block_k,
        "finalize_block_c": finalize_block_c,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    sum_a, dot_a, grad_a = _run_branch(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, zero, 192, 1856, True, **kwargs)
    sum_b, dot_b, grad_b = _run_branch(arg0_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, zero, 384, 1472, False, **kwargs)
    sum_c, dot_c, grad_c = _run_branch(arg0_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, zero, 384, 1088, False, **kwargs)
    sum_d, dot_d, grad_d = _run_branch(arg0_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, zero, 384, 704, False, **kwargs)
    sum_e, dot_e, grad_e = _run_branch(arg0_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, zero, 384, 320, False, **kwargs)
    sum_f, dot_f, grad_f = _run_branch(arg0_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, zero, 320, 0, False, **kwargs)

    return (
        zero,
        sum_a,
        dot_a,
        grad_a,
        sum_b,
        dot_b,
        grad_b,
        sum_c,
        dot_c,
        grad_c,
        sum_d,
        dot_d,
        grad_d,
        sum_e,
        dot_e,
        grad_e,
        sum_f,
        dot_f,
        grad_f,
    )
