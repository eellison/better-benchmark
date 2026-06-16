"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DenseNet bf16 sliced-residual BatchNorm-backward, constant-center low-memory max-pool scatter, ReLU-gated second BatchNorm-backward reductions, and returned bf16 dense input-gradient tensor by reverse-gathering the structured scatter destinations instead of materializing the f32 scatter_add buffer, whereas Inductor expands `_low_memory_max_pool_offsets_to_indices`, materializes the scatter, then schedules both BN-backward reduction/epilogue regions as generic tensor work; Inductor cannot do this today because scheduler/codegen treats the max-pool-backward offsets plus scatter_add as an opaque producer rather than a structured scatter-reduce source that can feed dependent reductions and dense side outputs while preserving bf16 cast boundaries; the fix is SCATTER_REDUCE: add a guarded max-pool-backward scatter-reduce lowering that fuses the sliced residual producer, gate, channel summaries, and dense BN-backward epilogue into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _first_partials_kernel(
    arg6_ptr,
    scalar_ptr,
    arg8_ptr,
    arg9_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C
    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)
    acc_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_dot = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + rows_base
        row_mask = rows < R
        n = rows // (H * W)
        rem = rows - n * (H * W)
        h = rem // W
        w = rem - h * W
        active = row_mask[:, None] & col_mask[None, :]
        offsets = n[:, None] * (C * H * W) + cols[None, :] * (H * W) + h[:, None] * W + w[:, None]

        mask_src = tl.load(arg6_ptr + offsets, mask=active, other=0.0)
        grad_src = tl.load(arg8_ptr + offsets, mask=active, other=0.0)
        selected = tl.where(mask_src <= 0.0, scalar, grad_src).to(tl.float32)
        centered = tl.load(arg9_ptr + offsets, mask=active, other=0.0).to(tl.float32) - mean[None, :]
        selected = tl.where(active, selected, 0.0)
        centered = tl.where(active, centered, 0.0)
        acc_sum += tl.sum(selected, axis=0)
        acc_dot += tl.sum(selected * centered, axis=0)

    out_offsets = group * C + cols
    tl.store(partial_sum_ptr + out_offsets, acc_sum, mask=col_mask)
    tl.store(partial_dot_ptr + out_offsets, acc_dot, mask=col_mask)


@triton.jit
def _final_first_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    inv_ptr,
    sum_ptr,
    dot_ptr,
    dot_inv_ptr,
    NUM_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_GROUPS)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < C)
    offsets = groups[:, None] * C + cols[None, :]

    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dots = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total_sum = tl.sum(sums, axis=0)
    total_dot = tl.sum(dots, axis=0)
    col_mask = cols < C
    inv = tl.load(inv_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(sum_ptr + cols, total_sum, mask=col_mask)
    tl.store(dot_ptr + cols, total_dot, mask=col_mask)
    tl.store(dot_inv_ptr + cols, total_dot * inv, mask=col_mask)


@triton.jit
def _second_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    scalar_ptr,
    arg8_ptr,
    arg9_ptr,
    mean1_ptr,
    inv1_ptr,
    weight1_ptr,
    first_sum_ptr,
    first_dot_ptr,
    arg14_ptr,
    mean2_ptr,
    inv2_ptr,
    weight2_ptr,
    bias2_ptr,
    selected_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    H1: tl.constexpr,
    W1: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C

    mean1 = tl.load(mean1_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    inv1 = tl.load(inv1_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    first_sum = tl.load(first_sum_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    first_dot = tl.load(first_dot_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    mean2 = tl.load(mean2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    inv2 = tl.load(inv2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)

    acc_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_dot = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + rows_base
        row_mask = rows < R
        n = rows // (H2 * W2)
        rem = rows - n * (H2 * W2)
        oh = rem // W2
        ow = rem - oh * W2
        active = row_mask[:, None] & col_mask[None, :]

        out_offsets = n[:, None] * (C * H2 * W2) + cols[None, :] * (H2 * W2) + oh[:, None] * W2 + ow[:, None]
        x2 = tl.load(arg14_ptr + out_offsets, mask=active, other=0.0).to(tl.float32)
        centered2 = x2 - mean2[None, :]
        affine = centered2 * inv2[None, :]
        affine = affine * weight2[None, :] + bias2[None, :]
        take_scalar = affine.to(tl.bfloat16, fp_downcast_rounding="rtne") <= 0.0

        ph = oh // 2
        pw = ow // 2
        has_source = active & ((oh % 2) == 0)[:, None] & ((ow % 2) == 0)[:, None]
        src_offsets64 = n[:, None] * (C * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]

        m0 = tl.load(arg6_ptr + src_offsets64, mask=has_source, other=0.0)
        g0 = tl.load(arg8_ptr + src_offsets64, mask=has_source, other=0.0)
        selected1 = tl.where(m0 <= 0.0, scalar, g0).to(tl.float32)
        centered1 = tl.load(arg9_ptr + src_offsets64, mask=has_source, other=0.0).to(tl.float32) - mean1[None, :]
        coef1 = (first_dot * 7.971938775510203e-05) * (inv1 * inv1)
        grad1 = selected1 - centered1 * coef1[None, :]
        grad1 = grad1 - (first_sum * 7.971938775510203e-05)[None, :]
        grad1 = grad1 * (inv1 * weight1)[None, :]
        grad1_bf16 = grad1.to(tl.bfloat16, fp_downcast_rounding="rtne")

        off0 = n[:, None] * (256 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        off1 = n[:, None] * (224 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        off2 = n[:, None] * (192 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        off3 = n[:, None] * (160 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        off4 = n[:, None] * (128 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        off5 = n[:, None] * (96 * H1 * W1) + cols[None, :] * (H1 * W1) + ph[:, None] * W1 + pw[:, None]
        residual = tl.load(arg0_ptr + off0, mask=has_source, other=0.0)
        residual = (residual + tl.load(arg1_ptr + off1, mask=has_source, other=0.0)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        residual = (residual + tl.load(arg2_ptr + off2, mask=has_source, other=0.0)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        residual = (residual + tl.load(arg3_ptr + off3, mask=has_source, other=0.0)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        residual = (residual + tl.load(arg4_ptr + off4, mask=has_source, other=0.0)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        residual = (residual + tl.load(arg5_ptr + off5, mask=has_source, other=0.0)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        scatter_bf16 = (residual + grad1_bf16).to(tl.bfloat16, fp_downcast_rounding="rtne")
        scatter_bf16 = tl.where(has_source, scatter_bf16, tl.full((BLOCK_R, BLOCK_C), 0.0, tl.bfloat16))

        selected2_bf16 = tl.where(take_scalar, scalar, scatter_bf16)
        tl.store(selected_ptr + out_offsets, selected2_bf16, mask=active)
        selected2 = selected2_bf16.to(tl.float32)
        selected2 = tl.where(active, selected2, 0.0)
        centered2 = tl.where(active, centered2, 0.0)
        acc_sum += tl.sum(selected2, axis=0)
        acc_dot += tl.sum(selected2 * centered2, axis=0)

    partial_offsets = group * C + cols
    tl.store(partial_sum_ptr + partial_offsets, acc_sum, mask=col_mask)
    tl.store(partial_dot_ptr + partial_offsets, acc_dot, mask=col_mask)


@triton.jit
def _final_second_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    inv_ptr,
    sum_ptr,
    dot_ptr,
    dot_inv_ptr,
    NUM_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_GROUPS)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < C)
    offsets = groups[:, None] * C + cols[None, :]
    sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dots = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total_sum = tl.sum(sums, axis=0)
    total_dot = tl.sum(dots, axis=0)
    col_mask = cols < C
    inv = tl.load(inv_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(sum_ptr + cols, total_sum, mask=col_mask)
    tl.store(dot_ptr + cols, total_dot, mask=col_mask)
    tl.store(dot_inv_ptr + cols, total_dot * inv, mask=col_mask)


@triton.jit
def _dense_output_kernel(
    selected_ptr,
    arg14_ptr,
    mean2_ptr,
    inv2_ptr,
    weight2_ptr,
    second_sum_ptr,
    second_dot_ptr,
    out_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    H2: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r_block = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < R
    col_mask = cols < C
    active = row_mask[:, None] & col_mask[None, :]
    n = rows // (H2 * W2)
    rem = rows - n * (H2 * W2)
    oh = rem // W2
    ow = rem - oh * W2
    out_offsets = n[:, None] * (C * H2 * W2) + cols[None, :] * (H2 * W2) + oh[:, None] * W2 + ow[:, None]

    mean2 = tl.load(mean2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    inv2 = tl.load(inv2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    second_sum = tl.load(second_sum_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    second_dot = tl.load(second_dot_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    x2 = tl.load(arg14_ptr + out_offsets, mask=active, other=0.0).to(tl.float32)
    centered2 = x2 - mean2[None, :]
    selected2 = tl.load(selected_ptr + out_offsets, mask=active, other=0.0).to(tl.float32)
    coef2 = (second_dot * 1.992984693877551e-05) * (inv2 * inv2)
    out = selected2 - centered2 * coef2[None, :]
    out = out - (second_sum * 1.992984693877551e-05)[None, :]
    out = out * (inv2 * weight2)[None, :]
    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


@oracle_impl(
    hardware="B200",
    point="0b922608",
    GROUP_R1=128,
    REDUCE_BLOCK_R1=64,
    GROUP_R2=256,
    REDUCE_BLOCK_R2=64,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    OUT_BLOCK_R=256,
    num_warps_reduce=8,
    num_warps_output=8,
)
def oracle_forward(
    inputs,
    *,
    GROUP_R1: int,
    REDUCE_BLOCK_R1: int,
    GROUP_R2: int,
    REDUCE_BLOCK_R2: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    OUT_BLOCK_R: int,
    num_warps_reduce: int,
    num_warps_output: int,
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
        _arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        *_,
    ) = inputs
    n = int(arg6_1.shape[0])
    c = int(arg6_1.shape[1])
    h1 = int(arg6_1.shape[2])
    w1 = int(arg6_1.shape[3])
    h2 = int(arg14_1.shape[2])
    w2 = int(arg14_1.shape[3])
    r1 = n * h1 * w1
    r2 = n * h2 * w2

    groups1 = triton.cdiv(r1, GROUP_R1)
    partial1_sum = torch.empty((groups1, c), device=arg6_1.device, dtype=torch.float32)
    partial1_dot = torch.empty((groups1, c), device=arg6_1.device, dtype=torch.float32)
    sum1 = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)
    dot1 = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)
    dot1_inv = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)

    _first_partials_kernel[(groups1, triton.cdiv(c, BLOCK_C))](
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        partial1_sum,
        partial1_dot,
        R=r1,
        C=c,
        H=h1,
        W=w1,
        GROUP_R=GROUP_R1,
        BLOCK_R=REDUCE_BLOCK_R1,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _final_first_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial1_sum,
        partial1_dot,
        arg11_1,
        sum1,
        dot1,
        dot1_inv,
        NUM_GROUPS=groups1,
        C=c,
        BLOCK_GROUPS=_ceil_pow2(groups1),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    groups2 = triton.cdiv(r2, GROUP_R2)
    partial2_sum = torch.empty((groups2, c), device=arg6_1.device, dtype=torch.float32)
    partial2_dot = torch.empty((groups2, c), device=arg6_1.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)
    dot2 = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)
    dot2_inv = torch.empty((c,), device=arg6_1.device, dtype=torch.float32)
    selected2 = torch.empty_strided(
        tuple(arg14_1.shape),
        tuple(int(s) for s in arg14_1.stride()),
        device=arg14_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        tuple(arg14_1.shape),
        tuple(int(s) for s in arg14_1.stride()),
        device=arg14_1.device,
        dtype=torch.bfloat16,
    )

    _second_partials_kernel[(groups2, triton.cdiv(c, BLOCK_C))](
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
        sum1,
        dot1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        selected2,
        partial2_sum,
        partial2_dot,
        R=r2,
        C=c,
        H1=h1,
        W1=w1,
        H2=h2,
        W2=w2,
        GROUP_R=GROUP_R2,
        BLOCK_R=REDUCE_BLOCK_R2,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _final_second_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial2_sum,
        partial2_dot,
        arg16_1,
        sum2,
        dot2,
        dot2_inv,
        NUM_GROUPS=groups2,
        C=c,
        BLOCK_GROUPS=_ceil_pow2(groups2),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _dense_output_kernel[(triton.cdiv(r2, OUT_BLOCK_R), triton.cdiv(c, BLOCK_C))](
        selected2,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        sum2,
        dot2,
        out,
        R=r2,
        C=c,
        H2=h2,
        W2=w2,
        BLOCK_R=OUT_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_output,
        num_stages=3,
    )
    return sum1, dot1_inv, sum2, dot2_inv, out
