"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete four-branch Inception bf16 avg-pool-backward plus residual-add source, ReLU-gated BN-backward channel reductions, vector outputs, and four channels-last bf16 gradient tensors with shared split reductions and fused epilogues; whereas Inductor schedules the pool-backward/add/slice producer, per-branch masks, sibling channel reductions, and dependent gradient tensors as generic materialized regions; Inductor cannot do this today because scheduler fusion does not coordinate a shared structured pool-backward producer with multiple gated BN-backward reductions and their dependent side-output tensors while preserving bf16 cast boundaries and channels-last strides; the fix is SCHEDULER_FUSION: add a multi-output Inception BN-backward schedule that shares the pool/add producer, split-reduces per-channel summaries, and emits dependent bf16 epilogues directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 192
GC = 768
H = 17
W = 17
HW = 289
K_TOTAL = 36992
SCALE = 2.703287197231834e-05


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _source_reduce_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    branch0_ptr,
    mean0_ptr,
    inv0_ptr,
    weight0_ptr,
    bias0_ptr,
    branch1_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    bias1_ptr,
    branch2_ptr,
    mean2_ptr,
    inv2_ptr,
    weight2_ptr,
    bias2_ptr,
    branch3_ptr,
    mean3_ptr,
    inv3_ptr,
    weight3_ptr,
    bias3_ptr,
    scalar_ptr,
    upstream_ptr,
    partials_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    branch = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < 192
    k_mask = k < 36992
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // 289
    hw = k - n * 289
    h = hw // 17
    w = hw - h * 17
    gc = (3 - branch) * 192 + c

    gbase = n[None, :] * 221952 + gc[:, None] + h[None, :] * 13056 + w[None, :] * 768
    pooled = tl.full((BLOCK_C, BLOCK_K), 0.0, tl.float32)
    for dh in tl.static_range(0, 3):
        hh = h - dh
        h_ok = hh >= 0
        for dw in tl.static_range(0, 3):
            ww = w - dw
            w_ok = ww >= 0
            poff = n[None, :] * 221952 + gc[:, None] + hh[None, :] * 13056 + ww[None, :] * 768
            grad = tl.load(pool_grad_ptr + poff, mask=mask & h_ok[None, :] & w_ok[None, :], other=0.0).to(tl.float32)
            pooled += _round_bf16_to_f32(grad * (1.0 / 9.0))
    g = _round_bf16_to_f32(pooled)
    g = _round_bf16_to_f32(g + tl.load(add0_ptr + gbase, mask=mask, other=0.0).to(tl.float32))
    g = _round_bf16_to_f32(g + tl.load(add1_ptr + gbase, mask=mask, other=0.0).to(tl.float32))
    g = _round_bf16_to_f32(g + tl.load(add2_ptr + gbase, mask=mask, other=0.0).to(tl.float32))

    local = n[None, :] * 55488 + c[:, None] + h[None, :] * 3264 + w[None, :] * 192
    branch_ptr = tl.where(branch == 0, branch0_ptr, tl.where(branch == 1, branch1_ptr, tl.where(branch == 2, branch2_ptr, branch3_ptr)))
    mean_ptr = tl.where(branch == 0, mean0_ptr, tl.where(branch == 1, mean1_ptr, tl.where(branch == 2, mean2_ptr, mean3_ptr)))
    inv_ptr = tl.where(branch == 0, inv0_ptr, tl.where(branch == 1, inv1_ptr, tl.where(branch == 2, inv2_ptr, inv3_ptr)))
    weight_ptr = tl.where(branch == 0, weight0_ptr, tl.where(branch == 1, weight1_ptr, tl.where(branch == 2, weight2_ptr, weight3_ptr)))
    bias_ptr = tl.where(branch == 0, bias0_ptr, tl.where(branch == 1, bias1_ptr, tl.where(branch == 2, bias2_ptr, bias3_ptr)))

    x = tl.load(branch_ptr + local, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    centered = x - mean[:, None]
    bn = _round_bf16_to_f32(centered * inv[:, None] * weight[:, None] + bias[:, None])
    use_grad = (bn > 0.0) | (bn != bn)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    where_val = tl.where(use_grad, g, scalar)

    upoff = branch * 7102464 + local
    tl.store(upstream_ptr + upoff, where_val, mask=mask)

    pbase = (tl.program_id(2) * 1536) + (branch * 384) + c
    tl.store(
        partials_ptr + pbase,
        tl.sum(tl.where(mask, where_val, 0.0), axis=1),
        mask=c_mask,
    )
    tl.store(
        partials_ptr + pbase + 192,
        tl.sum(tl.where(mask, where_val * centered, 0.0), axis=1),
        mask=c_mask,
    )


@triton.jit
def _finalize_branch_kernel(
    partials_ptr,
    stats_ptr,
    out_sum_ptr,
    out_scaled_ptr,
    inv_ptr,
    BRANCH: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    c_mask = c < 192
    mask = (groups[:, None] < NUM_GROUPS) & c_mask[None, :]
    offsets = groups[:, None] * 1536 + BRANCH * 384 + c[None, :]

    sum_where = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_centered = tl.load(partials_ptr + offsets + 192, mask=mask, other=0.0).to(tl.float32)
    sw = tl.sum(sum_where, axis=0)
    sc = tl.sum(sum_centered, axis=0)
    inv = tl.load(inv_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(stats_ptr + BRANCH * 384 + c, sw, mask=c_mask)
    tl.store(stats_ptr + (BRANCH * 2 + 1) * 192 + c, sc, mask=c_mask)
    tl.store(out_sum_ptr + c, sw, mask=c_mask)
    tl.store(out_scaled_ptr + c, sc * inv, mask=c_mask)


@triton.jit
def _grad_epilogue_kernel(
    upstream_ptr,
    branch0_ptr,
    mean0_ptr,
    inv0_ptr,
    weight0_ptr,
    out0_ptr,
    branch1_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    out1_ptr,
    branch2_ptr,
    mean2_ptr,
    inv2_ptr,
    weight2_ptr,
    out2_ptr,
    branch3_ptr,
    mean3_ptr,
    inv3_ptr,
    weight3_ptr,
    out3_ptr,
    scalar_ptr,
    stats_ptr,
    BLOCK_ELEMS: tl.constexpr,
):
    branch = tl.program_id(0)
    linear = tl.program_id(1) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < 7102464
    c = linear % 192
    tmp = linear // 192
    w = tmp % 17
    tmp = tmp // 17
    h = tmp % 17
    n = tmp // 17
    local = n * 55488 + c + h * 3264 + w * 192

    branch_ptr = tl.where(branch == 0, branch0_ptr, tl.where(branch == 1, branch1_ptr, tl.where(branch == 2, branch2_ptr, branch3_ptr)))
    mean_ptr = tl.where(branch == 0, mean0_ptr, tl.where(branch == 1, mean1_ptr, tl.where(branch == 2, mean2_ptr, mean3_ptr)))
    inv_ptr = tl.where(branch == 0, inv0_ptr, tl.where(branch == 1, inv1_ptr, tl.where(branch == 2, inv2_ptr, inv3_ptr)))
    weight_ptr = tl.where(branch == 0, weight0_ptr, tl.where(branch == 1, weight1_ptr, tl.where(branch == 2, weight2_ptr, weight3_ptr)))
    out_ptr = tl.where(branch == 0, out0_ptr, tl.where(branch == 1, out1_ptr, tl.where(branch == 2, out2_ptr, out3_ptr)))

    x = tl.load(branch_ptr + local, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    g = tl.load(upstream_ptr + branch * 7102464 + local, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(stats_ptr + branch * 384 + c, mask=active, other=0.0).to(tl.float32)
    sum_centered = tl.load(stats_ptr + (branch * 2 + 1) * 192 + c, mask=active, other=0.0).to(tl.float32)

    centered = x - mean
    mean_term = sum_where * 2.703287197231834e-05
    variance_term = sum_centered * 2.703287197231834e-05 * inv * inv
    grad = (g - centered * variance_term - mean_term) * (inv * weight)
    tl.store(out_ptr + local, grad, mask=active)


# 9eb75afe: channels-last Inception four-branch BN-backward tail
@oracle_impl(hardware="B200", point="9eb75afe", BLOCK_C=8, BLOCK_K=256, FINAL_BLOCK_C=8, BLOCK_ELEMS=256, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_K: int,
    FINAL_BLOCK_C: int,
    BLOCK_ELEMS: int,
    num_warps: int,
):
    (
        pool_grad,
        _pool_input,
        add0,
        add1,
        add2,
        branch0,
        mean0,
        inv0,
        weight0,
        _bias0,
        scalar,
        branch1,
        mean1,
        inv1,
        weight1,
        _bias1,
        branch2,
        mean2,
        inv2,
        weight2,
        _bias2,
        branch3,
        mean3,
        inv3,
        weight3,
        _bias3,
    ) = inputs

    out_sum0 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_scaled0 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_grad0 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C), device=pool_grad.device, dtype=torch.bfloat16
    )
    out_sum1 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_scaled1 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_grad1 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C), device=pool_grad.device, dtype=torch.bfloat16
    )
    out_sum2 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_scaled2 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_grad2 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C), device=pool_grad.device, dtype=torch.bfloat16
    )
    out_sum3 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_scaled3 = torch.empty((C,), device=pool_grad.device, dtype=torch.float32)
    out_grad3 = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C), device=pool_grad.device, dtype=torch.bfloat16
    )

    upstream = torch.empty_strided(
        (4, N, C, H, W),
        (N * C * HW, C * HW, 1, W * C, C),
        device=pool_grad.device,
        dtype=torch.bfloat16,
    )
    num_groups = triton.cdiv(K_TOTAL, BLOCK_K)
    partials = torch.empty_strided(
        (num_groups, 4, 2, C),
        (4 * 2 * C, 2 * C, C, 1),
        device=pool_grad.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(
        (4, 2, C),
        (2 * C, C, 1),
        device=pool_grad.device,
        dtype=torch.float32,
    )

    _source_reduce_kernel[(4, triton.cdiv(C, BLOCK_C), num_groups)](
        pool_grad,
        add0,
        add1,
        add2,
        branch0,
        mean0,
        inv0,
        weight0,
        _bias0,
        branch1,
        mean1,
        inv1,
        weight1,
        _bias1,
        branch2,
        mean2,
        inv2,
        weight2,
        _bias2,
        branch3,
        mean3,
        inv3,
        weight3,
        _bias3,
        scalar,
        upstream,
        partials,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_branch_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partials, stats, out_sum0, out_scaled0, inv0, BRANCH=0, NUM_GROUPS=num_groups, GROUP_BLOCK=group_block, BLOCK_C=FINAL_BLOCK_C, num_warps=8
    )
    _finalize_branch_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partials, stats, out_sum1, out_scaled1, inv1, BRANCH=1, NUM_GROUPS=num_groups, GROUP_BLOCK=group_block, BLOCK_C=FINAL_BLOCK_C, num_warps=8
    )
    _finalize_branch_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partials, stats, out_sum2, out_scaled2, inv2, BRANCH=2, NUM_GROUPS=num_groups, GROUP_BLOCK=group_block, BLOCK_C=FINAL_BLOCK_C, num_warps=8
    )
    _finalize_branch_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partials, stats, out_sum3, out_scaled3, inv3, BRANCH=3, NUM_GROUPS=num_groups, GROUP_BLOCK=group_block, BLOCK_C=FINAL_BLOCK_C, num_warps=8
    )

    _grad_epilogue_kernel[(4, triton.cdiv(N * C * HW, BLOCK_ELEMS))](
        upstream,
        branch0,
        mean0,
        inv0,
        weight0,
        out_grad0,
        branch1,
        mean1,
        inv1,
        weight1,
        out_grad1,
        branch2,
        mean2,
        inv2,
        weight2,
        out_grad2,
        branch3,
        mean3,
        inv3,
        weight3,
        out_grad3,
        scalar,
        stats,
        BLOCK_ELEMS=BLOCK_ELEMS,
        num_warps=num_warps,
    )

    return (
        out_sum0,
        out_scaled0,
        out_grad0,
        out_sum1,
        out_scaled1,
        out_grad1,
        out_sum2,
        out_scaled2,
        out_grad2,
        out_sum3,
        out_scaled3,
        out_grad3,
    )
