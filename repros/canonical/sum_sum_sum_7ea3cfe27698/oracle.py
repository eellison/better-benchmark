"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Adv-Inception bf16 channels-last avg-pool-add fanout returned by Repro.forward, including the full `add_2` tensor, both ReLU-gated BN-backward channel-sum pairs, both scale-gradient vectors, and both dense tensor epilogues; Inductor currently lowers the exact channels-last avg_pool2d_backward, three bf16 adds, channel slices, two ReLU masks, four `sum([0, 2, 3])` reductions, and dependent broadcast epilogues as generic materialized schedules; Inductor cannot do this today because its scheduler has no cooperative multi-output channel-reduction template that shares a structured pooled-add source across disjoint slices while preserving bf16 rounding boundaries and full returned output scope; the fix is COOPERATIVE_SPLIT_K: teach Inductor to form a tiled split-K plan for these sibling channel reductions and sink their finalized scalars into the tensor epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C_TOTAL = 1280
H = 8
W = 8
HW = 64
TOTAL_SPATIAL = N * HW
REDUCE_SCALE = 0.0001220703125
NUMEL_TOTAL = N * C_TOTAL * HW


@triton.jit
def _round_bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _pooled_add_kernel(
    grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    out_ptr,
    NUMEL: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL

    c = offsets % C_TOTAL_
    spatial = offsets // C_TOTAL_
    w = spatial % W_
    h = (spatial // W_) % H_
    n = spatial // HW_
    base_n = n * (C_TOTAL_ * HW_)

    pool = tl.zeros((BLOCK,), dtype=tl.float32)
    for dh in tl.static_range(-2, 1):
        ih = h + dh
        h_ok = mask & (ih >= 0)
        for dw in tl.static_range(-2, 1):
            iw = w + dw
            active = h_ok & (iw >= 0)
            neighbor_offsets = base_n + ih * (W_ * C_TOTAL_) + iw * C_TOTAL_ + c
            term = tl.load(grad_ptr + neighbor_offsets, mask=active, other=0.0).to(tl.float32)
            pool += _round_bf16(term * 0.1111111111111111)

    value = _round_bf16(pool)
    value = _round_bf16(value + tl.load(add0_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    value = _round_bf16(value + tl.load(add1_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    value = _round_bf16(value + tl.load(add2_ptr + offsets, mask=mask, other=0.0).to(tl.float32))
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _branch_partial_kernel(
    add2_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    partial_sum_ptr,
    partial_xhat_ptr,
    SOURCE_OFFSET: tl.constexpr,
    C_BRANCH: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    C_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    k = tl.program_id(1) * K_BLOCK + tl.arange(0, K_BLOCK)
    c_mask = c < C_BRANCH
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    branch_offsets = n[:, None] * (C_BRANCH * HW_) + hw[:, None] * C_BRANCH + c[None, :]
    add_offsets = n[:, None] * (C_TOTAL_ * HW_) + hw[:, None] * C_TOTAL_ + (SOURCE_OFFSET + c[None, :])
    active = k_mask[:, None] & c_mask[None, :]

    x = tl.load(activation_ptr + branch_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[None, :]
    affine = centered * invstd[None, :]
    affine = affine * gamma[None, :]
    affine = affine + beta[None, :]
    relu_mask = _round_bf16(affine) <= 0.0

    source = tl.load(add2_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(relu_mask, full, source)
    where_value = tl.where(active, where_value, 0.0)

    sum_value = tl.sum(where_value, axis=0)
    sum_xhat = tl.sum(where_value * centered, axis=0)
    partial_offsets = tl.program_id(1) * C_BRANCH + c
    tl.store(partial_sum_ptr + partial_offsets, sum_value, mask=c_mask)
    tl.store(partial_xhat_ptr + partial_offsets, sum_xhat, mask=c_mask)


@triton.jit
def _branch_finalize_kernel(
    partial_sum_ptr,
    partial_xhat_ptr,
    invstd_ptr,
    sum_out_ptr,
    xhat_out_ptr,
    vector_out_ptr,
    C_BRANCH: tl.constexpr,
    NUM_TILES: tl.constexpr,
    C_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    tiles = tl.arange(0, T_BLOCK)
    c_mask = c < C_BRANCH
    tile_mask = tiles < NUM_TILES
    offsets = tiles[:, None] * C_BRANCH + c[None, :]

    values = tl.load(partial_sum_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    xhat_values = tl.load(partial_xhat_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    sum_value = tl.sum(values, axis=0)
    sum_xhat = tl.sum(xhat_values, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value, mask=c_mask)
    tl.store(xhat_out_ptr + c, sum_xhat, mask=c_mask)
    tl.store(vector_out_ptr + c, sum_xhat * invstd, mask=c_mask)


@triton.jit
def _branch_epilogue_kernel(
    add2_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    sum_ptr,
    xhat_sum_ptr,
    out_ptr,
    SOURCE_OFFSET: tl.constexpr,
    C_BRANCH: tl.constexpr,
    C_TOTAL_: tl.constexpr,
    HW_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < NUMEL

    c = offsets % C_BRANCH
    spatial = offsets // C_BRANCH
    hw = spatial % HW_
    n = spatial // HW_
    add_offsets = n * (C_TOTAL_ * HW_) + hw * C_TOTAL_ + (SOURCE_OFFSET + c)

    x = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=active, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * invstd
    affine = affine * gamma
    affine = affine + beta
    relu_mask = _round_bf16(affine) <= 0.0

    source = tl.load(add2_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(relu_mask, full, source)

    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_xhat = tl.load(xhat_sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = sum_value * REDUCE_SCALE_
    invstd_sq = invstd * invstd
    variance_term = (sum_xhat * REDUCE_SCALE_) * invstd_sq
    affine_scale = invstd * gamma
    out = where_value - centered * variance_term
    out = out - mean_term
    out = out * affine_scale
    tl.store(out_ptr + offsets, out, mask=active)


def _empty_cl(device, channels):
    return torch.empty_strided(
        (N, channels, H, W),
        (channels * HW, 1, channels * W, channels),
        device=device,
        dtype=torch.bfloat16,
    )


def _run_branch(
    add2,
    activation,
    mean,
    invstd,
    gamma,
    beta,
    full,
    source_offset,
    channels,
    *,
    C_BLOCK,
    K_BLOCK,
    ELEM_BLOCK,
):
    device = add2.device
    num_tiles = triton.cdiv(TOTAL_SPATIAL, K_BLOCK)
    partial_sum = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)
    partial_xhat = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)
    sum_out = torch.empty((channels,), device=device, dtype=torch.float32)
    xhat_sum = torch.empty((channels,), device=device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=device, dtype=torch.float32)
    tensor_out = _empty_cl(device, channels)

    _branch_partial_kernel[(triton.cdiv(channels, C_BLOCK), num_tiles)](
        add2,
        activation,
        mean,
        invstd,
        gamma,
        beta,
        full,
        partial_sum,
        partial_xhat,
        SOURCE_OFFSET=source_offset,
        C_BRANCH=channels,
        C_TOTAL_=C_TOTAL,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        C_BLOCK=C_BLOCK,
        K_BLOCK=K_BLOCK,
        num_warps=8,
    )
    _branch_finalize_kernel[(triton.cdiv(channels, C_BLOCK),)](
        partial_sum,
        partial_xhat,
        invstd,
        sum_out,
        xhat_sum,
        vector_out,
        C_BRANCH=channels,
        NUM_TILES=num_tiles,
        C_BLOCK=C_BLOCK,
        T_BLOCK=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    _branch_epilogue_kernel[(triton.cdiv(N * channels * HW, ELEM_BLOCK),)](
        add2,
        activation,
        mean,
        invstd,
        gamma,
        beta,
        full,
        sum_out,
        xhat_sum,
        tensor_out,
        SOURCE_OFFSET=source_offset,
        C_BRANCH=channels,
        C_TOTAL_=C_TOTAL,
        HW_=HW,
        REDUCE_SCALE_=REDUCE_SCALE,
        NUMEL=N * channels * HW,
        BLOCK=ELEM_BLOCK,
        num_warps=4,
    )
    return sum_out, vector_out, tensor_out


@oracle_impl(hardware="B200", point="71ddf97f", ADD_BLOCK=256, C_BLOCK=16, K_BLOCK=512, ELEM_BLOCK=256)
def oracle_forward(inputs, *, ADD_BLOCK, C_BLOCK, K_BLOCK, ELEM_BLOCK):
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
    ) = inputs

    add2 = _empty_cl(arg0_1.device, C_TOTAL)
    _pooled_add_kernel[(triton.cdiv(NUMEL_TOTAL, ADD_BLOCK),)](
        arg0_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add2,
        NUMEL=NUMEL_TOTAL,
        C_TOTAL_=C_TOTAL,
        H_=H,
        W_=W,
        HW_=HW,
        BLOCK=ADD_BLOCK,
        num_warps=4,
    )

    sum_1, mul_10, out_192 = _run_branch(
        add2,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        320,
        192,
        C_BLOCK=C_BLOCK,
        K_BLOCK=K_BLOCK,
        ELEM_BLOCK=ELEM_BLOCK,
    )
    sum_3, mul_21, out_320 = _run_branch(
        add2,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg10_1,
        0,
        320,
        C_BLOCK=C_BLOCK,
        K_BLOCK=K_BLOCK,
        ELEM_BLOCK=ELEM_BLOCK,
    )
    return add2, sum_1, mul_10, out_192, sum_3, mul_21, out_320
