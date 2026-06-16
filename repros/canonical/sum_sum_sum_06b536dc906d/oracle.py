"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Inception scatter-gather reduction scope by constructing the low-memory max-pool-backward scatter-add source from `arg0[:, 512:1280]`/`arg1`, preserving the bf16 round after the scatter and after each following add with `arg2` and `arg3`, then sharing that source across the four sibling BN/ReLU-backward channel reductions and dense epilogues; Inductor currently materializes the scatter-add/view/channels-last clone/add chain and schedules the four masked reductions plus dependent tensor epilogues as generic fragments, because its scheduler cannot sink a structured max-pool scatter-add producer with bf16 cast boundaries into multiple channel reductions while reusing their summaries; the fix is SCATTER_REDUCE: add guarded max-pool scatter producer fusion that preserves dtype boundaries and feeds the shared reduction summaries into the per-element epilogue kernels."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
CTOTAL = 768
BRANCH_C = 192
H = 17
W = 17
HW = H * W
SPATIAL = N * HW


@triton.jit
def _scatter_candidate(arg0_ptr, arg1_ptr, n, c, h, w, oh, ow, active):
    kh = h - oh * 2
    kw = w - ow * 2
    valid = active & (oh >= 0) & (oh < 8) & (ow >= 0) & (ow < 8)
    valid = valid & (kh >= 0) & (kh < 3) & (kw >= 0) & (kw < 3)
    oh_safe = tl.where(valid, oh, 0)
    ow_safe = tl.where(valid, ow, 0)
    local = kh * 3 + kw
    index_offsets = n * (768 * 64) + oh_safe * (8 * 768) + ow_safe * 768 + c
    source_offsets = n * (1280 * 64) + oh_safe * (8 * 1280) + ow_safe * 1280 + 512 + c
    index = tl.load(arg1_ptr + index_offsets, mask=valid, other=-1).to(tl.int32)
    matched = valid & (index == local)
    return tl.load(arg0_ptr + source_offsets, mask=matched, other=0.0).to(tl.float32)


@triton.jit
def _load_scatter_value(arg0_ptr, arg1_ptr, n, c, h, w, active):
    oh_hi = h // 2
    oh_lo = oh_hi - 1
    ow_hi = w // 2
    ow_lo = ow_hi - 1

    acc = _scatter_candidate(arg0_ptr, arg1_ptr, n, c, h, w, oh_lo, ow_lo, active)
    acc += _scatter_candidate(arg0_ptr, arg1_ptr, n, c, h, w, oh_lo, ow_hi, active)
    acc += _scatter_candidate(arg0_ptr, arg1_ptr, n, c, h, w, oh_hi, ow_lo, active)
    return acc + _scatter_candidate(arg0_ptr, arg1_ptr, n, c, h, w, oh_hi, ow_hi, active)


@triton.jit
def _load_added_source(arg0_ptr, arg1_ptr, arg2_ptr, arg3_ptr, n, c, h, w, active):
    offsets = n * (768 * 289) + h * (17 * 768) + w * 768 + c
    scatter = _load_scatter_value(arg0_ptr, arg1_ptr, n, c, h, w, active)
    add2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    add3 = tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32)

    scatter_bf16 = scatter.to(tl.bfloat16).to(tl.float32)
    first_add = (scatter_bf16 + add2).to(tl.bfloat16).to(tl.float32)
    return (first_add + add3).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _partial_branch_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    full_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    partial_sum0_ptr,
    partial_sum1_ptr,
    num_tiles: tl.constexpr,
    BASE: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < 192
    k_mask = k < 36992
    n = k // 289
    spatial = k - n * 289
    h = spatial // 17
    w = spatial - h * 17
    active = k_mask[:, None] & c_mask[None, :]

    c_mat = c[None, :]
    global_c = c + BASE
    global_c_mat = global_c[None, :]
    n_mat = n[:, None]
    h_mat = h[:, None]
    w_mat = w[:, None]

    source = _load_added_source(
        arg0_ptr, arg1_ptr, arg2_ptr, arg3_ptr, n_mat, global_c_mat, h_mat, w_mat, active
    )
    x_offsets = n_mat * (192 * 289) + h_mat * (17 * 192) + w_mat * 192 + c_mat
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    relu_input = (centered * invstd[None, :] * gamma[None, :] + beta[None, :]).to(
        tl.bfloat16
    ).to(tl.float32)
    fill = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(relu_input <= 0.0, fill, source)
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)

    sum0 = tl.sum(where_value, axis=0)
    sum1 = tl.sum(where_value * centered, axis=0)
    partial_offsets = SLOT * (192 * num_tiles) + c * num_tiles + tl.program_id(1)
    tl.store(partial_sum0_ptr + partial_offsets, sum0, mask=c_mask)
    tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)


@triton.jit
def _finalize_branch_kernel(
    partial_sum0_ptr,
    partial_sum1_ptr,
    invstd_ptr,
    total0_ptr,
    total1_ptr,
    out_sum_ptr,
    out_vec_ptr,
    num_tiles: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < 192
    tile_mask = tiles < num_tiles
    offsets = SLOT * (192 * num_tiles) + tiles[:, None] + c[None, :] * num_tiles
    active = tile_mask[:, None] & c_mask[None, :]

    partial0 = tl.load(partial_sum0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial1 = tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    total0 = tl.sum(partial0, axis=0)
    total1 = tl.sum(partial1, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    vec = total1 * invstd

    total_offsets = SLOT * 192 + c
    tl.store(total0_ptr + total_offsets, total0, mask=c_mask)
    tl.store(total1_ptr + total_offsets, total1, mask=c_mask)
    tl.store(out_sum_ptr + c, total0, mask=c_mask)
    tl.store(out_vec_ptr + c, vec, mask=c_mask)


@triton.jit
def _epilogue_branch_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    full_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    total0_ptr,
    total1_ptr,
    out_ptr,
    BASE: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < 192
    k_mask = k < 36992
    n = k // 289
    spatial = k - n * 289
    h = spatial // 17
    w = spatial - h * 17
    active = k_mask[:, None] & c_mask[None, :]

    c_mat = c[None, :]
    global_c = c + BASE
    global_c_mat = global_c[None, :]
    n_mat = n[:, None]
    h_mat = h[:, None]
    w_mat = w[:, None]

    source = _load_added_source(
        arg0_ptr, arg1_ptr, arg2_ptr, arg3_ptr, n_mat, global_c_mat, h_mat, w_mat, active
    )
    x_offsets = n_mat * (192 * 289) + h_mat * (17 * 192) + w_mat * 192 + c_mat
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    total_offsets = SLOT * 192 + c
    total0 = tl.load(total0_ptr + total_offsets, mask=c_mask, other=0.0).to(tl.float32)
    total1 = tl.load(total1_ptr + total_offsets, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    relu_input = (centered * invstd[None, :] * gamma[None, :] + beta[None, :]).to(
        tl.bfloat16
    ).to(tl.float32)
    fill = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(relu_input <= 0.0, fill, source)

    mean_term = total0 * 2.703287197231834e-05
    variance_term = total1 * 2.703287197231834e-05 * invstd * invstd
    scale = invstd * gamma
    out = (where_value - centered * variance_term[None, :] - mean_term[None, :]) * scale[
        None, :
    ]
    out_offsets = n_mat * (192 * 289) + h_mat * (17 * 192) + w_mat * 192 + c_mat
    tl.store(out_ptr + out_offsets, out, mask=active)


def _empty_channels_last(device):
    return torch.empty_strided(
        (N, BRANCH_C, H, W),
        (BRANCH_C * HW, 1, W * BRANCH_C, BRANCH_C),
        device=device,
        dtype=torch.bfloat16,
    )


@oracle_impl(
    hardware="B200",
    point="0acf4a9d",
    block_c=8,
    block_k=512,
    final_block_c=8,
    epilogue_block_k=128,
)
def oracle_forward(
    inputs,
    *,
    block_c: int,
    block_k: int,
    final_block_c: int,
    epilogue_block_k: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        *_shape_params,
    ) = inputs
    del _shape_params

    device = arg0.device
    num_tiles = triton.cdiv(SPATIAL, block_k)
    partial_sum0 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    partial_sum1 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    sum0 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)
    sum1 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)

    sum_high = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    vec_high = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    out_high = _empty_channels_last(device)
    sum_wide = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    vec_wide = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    out_wide = _empty_channels_last(device)
    sum_mid = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    vec_mid = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    out_mid = _empty_channels_last(device)
    sum_low = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    vec_low = torch.empty((BRANCH_C,), device=device, dtype=torch.float32)
    out_low = _empty_channels_last(device)

    branches = (
        (576, 0, arg4, arg5, arg6, arg7, arg8, sum_high, vec_high, out_high),
        (384, 1, arg10, arg11, arg12, arg13, arg14, sum_wide, vec_wide, out_wide),
        (192, 2, arg15, arg16, arg17, arg18, arg19, sum_mid, vec_mid, out_mid),
        (0, 3, arg20, arg21, arg22, arg23, arg24, sum_low, vec_low, out_low),
    )

    for base, slot, x, mean, invstd, gamma, beta, _out_sum, _out_vec, _out in branches:
        _partial_branch_kernel[(triton.cdiv(BRANCH_C, block_c), num_tiles)](
            arg0,
            arg1,
            arg2,
            arg3,
            arg9,
            x,
            mean,
            invstd,
            gamma,
            beta,
            partial_sum0,
            partial_sum1,
            num_tiles=num_tiles,
            BASE=base,
            SLOT=slot,
            BLOCK_C=block_c,
            BLOCK_K=block_k,
            num_warps=8,
        )

    for _base, slot, _x, _mean, invstd, _gamma, _beta, out_sum, out_vec, _out in branches:
        _finalize_branch_kernel[(triton.cdiv(BRANCH_C, final_block_c),)](
            partial_sum0,
            partial_sum1,
            invstd,
            sum0,
            sum1,
            out_sum,
            out_vec,
            num_tiles=num_tiles,
            SLOT=slot,
            BLOCK_C=final_block_c,
            BLOCK_TILES=triton.next_power_of_2(num_tiles),
            num_warps=8,
        )

    for base, slot, x, mean, invstd, gamma, beta, _out_sum, _out_vec, out in branches:
        _epilogue_branch_kernel[
            (triton.cdiv(BRANCH_C, block_c), triton.cdiv(SPATIAL, epilogue_block_k))
        ](
            arg0,
            arg1,
            arg2,
            arg3,
            arg9,
            x,
            mean,
            invstd,
            gamma,
            beta,
            sum0,
            sum1,
            out,
            BASE=base,
            SLOT=slot,
            BLOCK_C=block_c,
            BLOCK_K=epilogue_block_k,
            num_warps=8,
        )

    return (
        sum_high,
        vec_high,
        out_high,
        sum_wide,
        vec_wide,
        out_wide,
        sum_mid,
        vec_mid,
        out_mid,
        sum_low,
        vec_low,
        out_low,
    )
