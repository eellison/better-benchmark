"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 Adv-Inception avg-pool-backward plus six-branch BatchNorm-backward tail returned by Repro.forward, including the 3x3 count-including-pad pool-gradient reconstruction, three bf16 residual adds, fixed channel slices, bf16 affine-ReLU mask rounding, all twelve channel reductions, six scale-gradient vectors, and six channels-last bf16 tensor-gradient epilogues, whereas Inductor lowers the decomposed avg_pool2d_backward, add fanout, slice/mask producers, duplicated reductions, and dependent tensor epilogues as generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative multi-output reduction template that shares a structured pooled-add producer across disjoint channel slices while feeding sibling channel sums and full tensor epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to keep the pooled-add source virtual, split or persist the shared `(N,H,W)` reduction domain by channel slice, and finalize every returned vector and tensor output from the same coordinated accumulators while preserving bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
FULL_C = 2048
H = 8
W = 8
HW = H * W
TOTAL_SPATIAL = N * HW
AVG_SCALE = 1.0 / 9.0
REDUCE_SCALE = 1.0 / TOTAL_SPATIAL


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


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
def _pooled_add_source(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    base,
    h,
    w,
    active,
    H_: tl.constexpr,
    W_: tl.constexpr,
    FULL_C_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
):
    h_stride: tl.constexpr = W_ * FULL_C_
    w_stride: tl.constexpr = FULL_C_
    prev_h1 = h > 0
    prev_h2 = h > 1
    prev_w1 = w > 0
    prev_w2 = w > 1

    pool = _bf16(tl.load(pool_grad_ptr + base - 2 * h_stride - 2 * w_stride, mask=active & prev_h2 & prev_w2, other=0.0).to(tl.float32) / 9.0)
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - 2 * h_stride - w_stride, mask=active & prev_h2 & prev_w1, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - 2 * h_stride, mask=active & prev_h2, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - h_stride - 2 * w_stride, mask=active & prev_h1 & prev_w2, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - h_stride - w_stride, mask=active & prev_h1 & prev_w1, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - h_stride, mask=active & prev_h1, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - 2 * w_stride, mask=active & prev_w2, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base - w_stride, mask=active & prev_w1, other=0.0).to(tl.float32) / 9.0))
    pool = _f32_add(pool, _bf16(tl.load(pool_grad_ptr + base, mask=active, other=0.0).to(tl.float32) / 9.0))
    pool = _bf16(pool)

    add_0 = _bf16(_f32_add(pool, tl.load(add0_ptr + base, mask=active, other=0.0).to(tl.float32)))
    add_1 = _bf16(_f32_add(add_0, tl.load(add1_ptr + base, mask=active, other=0.0).to(tl.float32)))
    return _bf16(_f32_add(add_1, tl.load(add2_ptr + base, mask=active, other=0.0).to(tl.float32)))

@triton.jit
def _branch_partial_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    source_offset: tl.constexpr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    FULL_C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_
    active = k_mask[:, None] & c_mask[None, :]
    branch_offsets = (
        n[:, None] * (C_ * HW_)
        + c[None, :]
        + h[:, None] * (W_ * C_)
        + w[:, None] * C_
    )
    source_base = (
        n[:, None] * (FULL_C_ * HW_)
        + source_offset
        + c[None, :]
        + h[:, None] * (W_ * FULL_C_)
        + w[:, None] * FULL_C_
    )

    x = tl.load(x_ptr + branch_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    source = _pooled_add_source(
        pool_grad_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        source_base,
        h[:, None],
        w[:, None],
        active,
        H_=H_,
        W_=W_,
        FULL_C_=FULL_C_,
        AVG_SCALE_=AVG_SCALE_,
    )

    centered = _f32_sub(x, mean[None, :])
    relu_input = _bf16(_f32_add(_f32_mul(_f32_mul(centered, invstd[None, :]), gamma[None, :]), beta[None, :]))
    full = tl.load(full_ptr).to(tl.float32)
    where_self = tl.where(relu_input <= 0.0, full, source)
    where_self = tl.where(active, where_self, 0.0)
    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(_f32_mul(where_self, centered), axis=0)
    partial_offsets = c * num_tiles + tl.program_id(1)
    tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)
    tl.store(partial_sum2_ptr + partial_offsets, sum2, mask=c_mask)


@triton.jit
def _branch_finalize_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    c_mask = c < C_
    tile_mask = tiles < num_tiles
    offsets = tiles[:, None] + c[None, :] * num_tiles
    values1 = tl.load(partial_sum1_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    values2 = tl.load(partial_sum2_ptr + offsets, mask=tile_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    sum1 = tl.sum(values1, axis=0)
    sum2 = tl.sum(values2, axis=0)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(sum1_ptr + c, sum1, mask=c_mask)
    tl.store(sum2_ptr + c, sum2, mask=c_mask)
    tl.store(vector_out_ptr + c, _f32_mul(sum2, invstd), mask=c_mask)


@triton.jit
def _branch_epilogue_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    full_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    source_offset: tl.constexpr,
    numel: tl.constexpr,
    C_: tl.constexpr,
    FULL_C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    AVG_SCALE_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < numel
    c = offsets % C_
    w = (offsets // C_) % W_
    h = (offsets // (C_ * W_)) % H_
    n = offsets // (C_ * HW_)
    source_base = (
        n * (FULL_C_ * HW_)
        + source_offset
        + c
        + h * (W_ * FULL_C_)
        + w * FULL_C_
    )
    source = _pooled_add_source(
        pool_grad_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        source_base,
        h,
        w,
        active,
        H_=H_,
        W_=W_,
        FULL_C_=FULL_C_,
        AVG_SCALE_=AVG_SCALE_,
    )

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c, mask=active, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)
    relu_input = _bf16(_f32_add(_f32_mul(_f32_mul(centered, invstd), gamma), beta))
    full = tl.load(full_ptr).to(tl.float32)
    where_self = tl.where(relu_input <= 0.0, full, source)
    variance_term = _f32_mul(_f32_mul(sum2, REDUCE_SCALE_), _f32_mul(invstd, invstd))
    mean_term = _f32_mul(sum1, REDUCE_SCALE_)
    out = _f32_mul(
        _f32_sub(_f32_sub(where_self, _f32_mul(centered, variance_term)), mean_term),
        _f32_mul(invstd, gamma),
    )
    tl.store(out_ptr + offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


def _empty_cl(shape, channel_stride, device):
    return torch.empty_strided(
        shape,
        (shape[1] * H * W, channel_stride, W * shape[1], shape[1]),
        device=device,
        dtype=torch.bfloat16,
    )


def _run_branch(
    pool_grad,
    add0,
    add1,
    add2,
    x,
    mean,
    invstd,
    gamma,
    beta,
    full,
    source_offset,
    channels,
    block_c,
    block_k,
    block_elems,
):
    device = pool_grad.device
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    partial_sum2 = torch.empty((channels, num_tiles), device=device, dtype=torch.float32)
    sum1 = torch.empty((channels,), device=device, dtype=torch.float32)
    sum2 = torch.empty((channels,), device=device, dtype=torch.float32)
    vector_out = torch.empty((channels,), device=device, dtype=torch.float32)
    out = _empty_cl((N, channels, H, W), 1, device)

    _branch_partial_kernel[(triton.cdiv(channels, block_c), num_tiles)](
        pool_grad,
        add0,
        add1,
        add2,
        x,
        mean,
        invstd,
        gamma,
        beta,
        full,
        partial_sum1,
        partial_sum2,
        source_offset=source_offset,
        num_tiles=num_tiles,
        C_=channels,
        FULL_C_=FULL_C,
        H_=H,
        W_=W,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        AVG_SCALE_=AVG_SCALE,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=8,
    )
    _branch_finalize_kernel[(triton.cdiv(channels, block_c),)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        C_=channels,
        BLOCK_C=block_c,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    _branch_epilogue_kernel[(triton.cdiv(N * channels * HW, block_elems),)](
        pool_grad,
        add0,
        add1,
        add2,
        x,
        mean,
        invstd,
        gamma,
        beta,
        full,
        sum1,
        sum2,
        out,
        source_offset=source_offset,
        numel=N * channels * HW,
        C_=channels,
        FULL_C_=FULL_C,
        H_=H,
        W_=W,
        HW_=HW,
        AVG_SCALE_=AVG_SCALE,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )
    return sum1, vector_out, out


@oracle_impl(hardware="B200", point="32065934", BLOCK_C=16, BLOCK_K=512, BLOCK_ELEMS=256)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_K: int, BLOCK_ELEMS: int):
    (
        pool_grad,
        _pool_self,
        add0,
        add1,
        add2,
        x_a,
        mean_a,
        invstd_a,
        gamma_a,
        beta_a,
        full,
        x_b,
        mean_b,
        invstd_b,
        gamma_b,
        beta_b,
        x_c,
        mean_c,
        invstd_c,
        gamma_c,
        beta_c,
        x_d,
        mean_d,
        invstd_d,
        gamma_d,
        beta_d,
        x_e,
        mean_e,
        invstd_e,
        gamma_e,
        beta_e,
        x_f,
        mean_f,
        invstd_f,
        gamma_f,
        beta_f,
    ) = inputs
    sum_a, vec_a, out_a = _run_branch(
        pool_grad, add0, add1, add2, x_a, mean_a, invstd_a, gamma_a, beta_a,
        full, 1856, 192, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )
    sum_b, vec_b, out_b = _run_branch(
        pool_grad, add0, add1, add2, x_b, mean_b, invstd_b, gamma_b, beta_b,
        full, 1472, 384, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )
    sum_c, vec_c, out_c = _run_branch(
        pool_grad, add0, add1, add2, x_c, mean_c, invstd_c, gamma_c, beta_c,
        full, 1088, 384, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )
    sum_d, vec_d, out_d = _run_branch(
        pool_grad, add0, add1, add2, x_d, mean_d, invstd_d, gamma_d, beta_d,
        full, 704, 384, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )
    sum_e, vec_e, out_e = _run_branch(
        pool_grad, add0, add1, add2, x_e, mean_e, invstd_e, gamma_e, beta_e,
        full, 320, 384, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )
    sum_f, vec_f, out_f = _run_branch(
        pool_grad, add0, add1, add2, x_f, mean_f, invstd_f, gamma_f, beta_f,
        full, 0, 320, BLOCK_C, BLOCK_K, BLOCK_ELEMS,
    )

    return (
        sum_a,
        vec_a,
        out_a,
        sum_b,
        vec_b,
        out_b,
        sum_c,
        vec_c,
        out_c,
        sum_d,
        vec_d,
        out_d,
        sum_e,
        vec_e,
        out_e,
        sum_f,
        vec_f,
        out_f,
    )
