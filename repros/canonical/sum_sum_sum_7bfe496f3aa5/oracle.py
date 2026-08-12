"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Adv-Inception bf16 channels-last avg-pool-backward/add fanout and the four dependent BN/ReLU-backward branches, preserving ATen's channels-last `avg_pool2d_backward` numerics, the bf16 rounding after each add, the four disjoint channel-slice reductions, and the final bf16 dense epilogues; Inductor currently materializes the pooled/add producer, slices, masks, sibling channel reductions, and reduction-dependent epilogues as separate generic kernels; Inductor cannot do this today because its scheduler/codegen has no cooperative multi-output split-K reduction template that keeps the shared rounded producer live across disjoint BN-backward channel slices while feeding finalized channel summaries into full-tensor epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to coordinate one shared producer with split-NHW channel reductions and dependent dense epilogues for all branch slices."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C_TOTAL = 256
H = 35
W = 35
HW = H * W
SPATIAL = N * HW
NUMEL_TOTAL = N * C_TOTAL * HW
REDUCE_SCALE = 6.3775510204081635e-06


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
def _round_bf16(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _pool_contribution(arg0_ptr, base, mask):
    value = tl.load(arg0_ptr + base, mask=mask, other=0.0).to(tl.float32)
    return _round_bf16(_f32_mul(value, 0.1111111111111111))


@triton.jit
def _pool_add_chain_kernel(
    arg0_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    add2_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < 40140800
    spatial_flat = offsets // 256
    spatial = spatial_flat % 1225
    h = spatial // 35
    w = spatial - h * 35
    base = offsets

    value = tl.zeros((BLOCK,), dtype=tl.float32)
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (2 * 35 * 256) - (2 * 256), active & (h > 1) & (w > 1)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (2 * 35 * 256) - 256, active & (h > 1) & (w > 0)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (2 * 35 * 256), active & (h > 1)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (35 * 256) - (2 * 256), active & (h > 0) & (w > 1)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (35 * 256) - 256, active & (h > 0) & (w > 0)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (35 * 256), active & (h > 0)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - (2 * 256), active & (w > 1)),
    )
    value = _f32_add(
        value,
        _pool_contribution(arg0_ptr, base - 256, active & (w > 0)),
    )
    value = _f32_add(value, _pool_contribution(arg0_ptr, base, active))
    value = _round_bf16(value)

    value = _round_bf16(
        _f32_add(value, tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    )
    value = _round_bf16(
        _f32_add(value, tl.load(arg3_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    )
    value = _round_bf16(
        _f32_add(value, tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32))
    )
    tl.store(add2_ptr + offsets, value, mask=active)


@triton.jit
def _partial_branch_kernel(
    add2_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    partial_sum0_ptr,
    partial_sum1_ptr,
    SOURCE_OFFSET: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C
    k_mask = k < 156800
    n = k // 1225
    spatial = k - n * 1225
    active = k_mask[:, None] & c_mask[None, :]

    c_mat = c[None, :]
    x_offsets = n[:, None] * (C * 1225) + spatial[:, None] * C + c_mat
    add_offsets = (
        n[:, None] * (256 * 1225)
        + spatial[:, None] * 256
        + SOURCE_OFFSET
        + c_mat
    )

    source = tl.load(add2_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    affine = _f32_mul(centered, invstd[None, :])
    affine = _f32_mul(affine, gamma[None, :])
    affine = _f32_add(affine, beta[None, :])
    fill = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(_round_bf16(affine) <= 0.0, fill, source)
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)

    partial_offsets = tl.program_id(1) * C + c
    tl.store(partial_sum0_ptr + partial_offsets, tl.sum(where_value, axis=0), mask=c_mask)
    tl.store(
        partial_sum1_ptr + partial_offsets,
        tl.sum(_f32_mul(where_value, centered), axis=0),
        mask=c_mask,
    )


@triton.jit
def _finalize_branch_kernel(
    partial_sum0_ptr,
    partial_sum1_ptr,
    invstd_ptr,
    sum0_ptr,
    sum1_ptr,
    vec_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < C
    tile_mask = tiles < NUM_TILES
    active = tile_mask[:, None] & c_mask[None, :]
    offsets = tiles[:, None] * C + c[None, :]

    partial0 = tl.load(partial_sum0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial1 = tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    total0 = tl.sum(partial0, axis=0)
    total1 = tl.sum(partial1, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum0_ptr + c, total0, mask=c_mask)
    tl.store(sum1_ptr + c, total1, mask=c_mask)
    tl.store(vec_ptr + c, _f32_mul(total1, invstd), mask=c_mask)


@triton.jit
def _epilogue_branch_kernel(
    add2_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    sum0_ptr,
    sum1_ptr,
    out_ptr,
    SOURCE_OFFSET: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < (128 * C * 1225)
    c = offsets % C
    spatial_flat = offsets // C
    n = spatial_flat // 1225
    spatial = spatial_flat - n * 1225
    add_offsets = n * (256 * 1225) + spatial * 256 + SOURCE_OFFSET + c

    source = tl.load(add2_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=active, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=active, other=0.0).to(tl.float32)
    total0 = tl.load(sum0_ptr + c, mask=active, other=0.0).to(tl.float32)
    total1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    affine = _f32_mul(centered, invstd)
    affine = _f32_mul(affine, gamma)
    affine = _f32_add(affine, beta)
    fill = tl.load(full_ptr).to(tl.float32)
    where_value = tl.where(_round_bf16(affine) <= 0.0, fill, source)

    mean_term = _f32_mul(total0, 0.0000063775510204081635)
    variance_term = _f32_mul(
        _f32_mul(total1, 0.0000063775510204081635),
        _f32_mul(invstd, invstd),
    )
    scale = _f32_mul(invstd, gamma)
    out = _f32_sub(where_value, _f32_mul(centered, variance_term))
    out = _f32_sub(out, mean_term)
    out = _f32_mul(out, scale)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


def _empty_channels_last(device, channels: int) -> torch.Tensor:
    return torch.empty_strided(
        (N, channels, H, W),
        (channels * HW, 1, W * channels, channels),
        device=device,
        dtype=torch.bfloat16,
    )


def _run_branch(
    add2,
    x,
    mean,
    invstd,
    gamma,
    beta,
    full,
    *,
    source_offset: int,
    channels: int,
    c_block: int,
    k_block: int,
    final_c_block: int,
    elem_block: int,
):
    device = add2.device
    num_tiles = triton.cdiv(SPATIAL, k_block)
    partial_sum0 = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)
    partial_sum1 = torch.empty((num_tiles, channels), device=device, dtype=torch.float32)
    sum0 = torch.empty((channels,), device=device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=device, dtype=torch.float32)
    vec = torch.empty((channels,), device=device, dtype=torch.float32)
    out = _empty_channels_last(device, channels)

    _partial_branch_kernel[(triton.cdiv(channels, c_block), num_tiles)](
        add2,
        x,
        mean,
        invstd,
        gamma,
        beta,
        full,
        partial_sum0,
        partial_sum1,
        SOURCE_OFFSET=source_offset,
        C=channels,
        BLOCK_C=c_block,
        BLOCK_K=k_block,
        num_warps=8,
    )
    _finalize_branch_kernel[(triton.cdiv(channels, final_c_block),)](
        partial_sum0,
        partial_sum1,
        invstd,
        sum0,
        sum1,
        vec,
        C=channels,
        NUM_TILES=num_tiles,
        BLOCK_C=final_c_block,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=8,
    )
    _epilogue_branch_kernel[(triton.cdiv(N * channels * HW, elem_block),)](
        add2,
        x,
        mean,
        invstd,
        gamma,
        beta,
        full,
        sum0,
        sum1,
        out,
        SOURCE_OFFSET=source_offset,
        C=channels,
        BLOCK=elem_block,
        num_warps=4,
    )
    return sum0, vec, out


@oracle_impl(
    hardware="B200",
    point="32cbe7ad",
    ADD_BLOCK=1024,
    C_BLOCK=16,
    K_BLOCK=512,
    FINAL_C_BLOCK=16,
    ELEM_BLOCK=512,
)
def oracle_forward(
    inputs,
    *,
    ADD_BLOCK: int,
    C_BLOCK: int,
    K_BLOCK: int,
    FINAL_C_BLOCK: int,
    ELEM_BLOCK: int,
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
        arg25,
    ) = inputs

    del arg1
    add2 = _empty_channels_last(arg0.device, C_TOTAL)
    _pool_add_chain_kernel[(triton.cdiv(NUMEL_TOTAL, ADD_BLOCK),)](
        arg0,
        arg2,
        arg3,
        arg4,
        add2,
        BLOCK=ADD_BLOCK,
        num_warps=4,
    )

    high_sum, high_vec, high_out = _run_branch(
        add2,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        source_offset=224,
        channels=32,
        c_block=C_BLOCK,
        k_block=K_BLOCK,
        final_c_block=FINAL_C_BLOCK,
        elem_block=ELEM_BLOCK,
    )
    wide_sum, wide_vec, wide_out = _run_branch(
        add2,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg10,
        source_offset=128,
        channels=96,
        c_block=C_BLOCK,
        k_block=K_BLOCK,
        final_c_block=FINAL_C_BLOCK,
        elem_block=ELEM_BLOCK,
    )
    mid_sum, mid_vec, mid_out = _run_branch(
        add2,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg10,
        source_offset=64,
        channels=64,
        c_block=C_BLOCK,
        k_block=K_BLOCK,
        final_c_block=FINAL_C_BLOCK,
        elem_block=ELEM_BLOCK,
    )
    low_sum, low_vec, low_out = _run_branch(
        add2,
        arg21,
        arg22,
        arg23,
        arg24,
        arg25,
        arg10,
        source_offset=0,
        channels=64,
        c_block=C_BLOCK,
        k_block=K_BLOCK,
        final_c_block=FINAL_C_BLOCK,
        elem_block=ELEM_BLOCK,
    )

    return (
        high_sum,
        high_vec,
        high_out,
        wide_sum,
        wide_vec,
        wide_out,
        mid_sum,
        mid_vec,
        mid_out,
        low_sum,
        low_vec,
        low_out,
    )
