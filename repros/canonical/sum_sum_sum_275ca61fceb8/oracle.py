"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 dual batch-norm-backward return tuple, including the bf16-rounded add/copy producer, raw full-channel and sliced-channel reduction side outputs, and both dependent bf16 gradient tensors with their required strides, whereas Inductor schedules the channels-last copy/contiguous clone path, sibling reductions, slice, and dependent epilogues as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output lowering that shares a memory-format-changing producer across overlapping channel reductions and multiple layout-distinct side outputs; the fix is COOPERATIVE_SPLIT_K: add a split-row multi-output lowering that preserves the exact bf16 cast boundaries, finalizes the small channel summaries once, and sinks the dependent stores into the producer schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C24 = 24
C12 = 12
H = 56
W = 56
HW = H * W
ROWS = N * HW
INV_ROWS = 6.228077168367346e-07

GROUP_ROWS = 32768
GROUP_R_BLOCK = 2048
GROUP_BLOCK_C = 1
NUM_GROUPS = 49
FINAL_BLOCK_GROUPS = 64
STORE_BLOCK_ROWS = 64
STORE_BLOCK_C = 32


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
def _rounded_add(arg0_ptr, arg1_ptr, contig_offsets, cl_offsets, mask):
    lhs = tl.load(arg0_ptr + contig_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + cl_offsets, mask=mask, other=0.0).to(tl.float32)
    return _f32_add(lhs, rhs).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _partial_grouped_reductions_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean24_ptr,
    arg6_ptr,
    mean12_ptr,
    partial_x24_ptr,
    partial_dot24_ptr,
    partial_dot12_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    GROUP_ROWS_: tl.constexpr,
    R_BLOCK_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    group = tl.program_id(0)
    c24 = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    c12 = tl.where(c24 >= C12_, c24 - C12_, 0)
    c24_mask = c24 < C24_
    tail_mask = (c24 >= C12_) & c24_mask
    group_start = group * GROUP_ROWS_

    acc_x24 = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_dot24 = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_dot12 = tl.zeros((BLOCK_C_,), dtype=tl.float32)

    for start in tl.range(0, GROUP_ROWS_, R_BLOCK_):
        rows = group_start + start + tl.arange(0, R_BLOCK_)
        row_mask = rows < ROWS_
        n = rows // HW_
        hw = rows - n * HW_

        contig_offsets = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
        cl_offsets = n[None, :] * (C24_ * HW_) + hw[None, :] * C24_ + c24[:, None]
        mask24 = c24_mask[:, None] & row_mask[None, :]
        x24 = _rounded_add(arg0_ptr, arg1_ptr, contig_offsets, cl_offsets, mask24)
        centered24 = _f32_sub(
            tl.load(arg2_ptr + cl_offsets, mask=mask24, other=0.0).to(tl.float32),
            tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None],
        )
        acc_x24 += tl.sum(tl.where(mask24, x24, 0.0), axis=1)
        acc_dot24 += tl.sum(tl.where(mask24, _f32_mul(x24, centered24), 0.0), axis=1)

        arg6_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c12[:, None]
        mask12 = tail_mask[:, None] & row_mask[None, :]
        centered12 = _f32_sub(
            tl.load(arg6_ptr + arg6_offsets, mask=mask12, other=0.0).to(tl.float32),
            tl.load(mean12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)[:, None],
        )
        acc_dot12 += tl.sum(tl.where(mask12, _f32_mul(x24, centered12), 0.0), axis=1)

    tl.store(partial_x24_ptr + group * C24_ + c24, acc_x24, mask=c24_mask)
    tl.store(partial_dot24_ptr + group * C24_ + c24, acc_dot24, mask=c24_mask)
    tl.store(partial_dot12_ptr + group * C12_ + c12, acc_dot12, mask=tail_mask)


@triton.jit
def _finalize_reductions_kernel(
    partial_x24_ptr,
    partial_dot24_ptr,
    partial_dot12_ptr,
    scale24_ptr,
    grad24_ptr,
    scale12_ptr,
    grad12_ptr,
    sum24_out_ptr,
    dot24_out_ptr,
    sum12_out_ptr,
    dot12_out_ptr,
    mean24_ptr,
    coeff24_ptr,
    coeff12_ptr,
    gain24_ptr,
    gain12_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    NUM_GROUPS_: tl.constexpr,
    BLOCK_GROUPS_: tl.constexpr,
    INV_ROWS_: tl.constexpr,
):
    c24 = tl.arange(0, 32)
    c12 = tl.arange(0, 16)
    groups = tl.arange(0, BLOCK_GROUPS_)
    c24_mask = c24 < C24_
    c12_mask = c12 < C12_

    mask24 = (groups[:, None] < NUM_GROUPS_) & c24_mask[None, :]
    offsets24 = groups[:, None] * C24_ + c24[None, :]
    sum_x24 = tl.sum(
        tl.load(partial_x24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32),
        axis=0,
    )
    dot24 = tl.sum(
        tl.load(partial_dot24_ptr + offsets24, mask=mask24, other=0.0).to(tl.float32),
        axis=0,
    )

    mask12 = (groups[:, None] < NUM_GROUPS_) & c12_mask[None, :]
    offsets12 = groups[:, None] * C12_ + c12[None, :]
    dot12 = tl.sum(
        tl.load(partial_dot12_ptr + offsets12, mask=mask12, other=0.0).to(tl.float32),
        axis=0,
    )
    high_offsets12 = groups[:, None] * C24_ + (c12[None, :] + C12_)
    sum_x12 = tl.sum(
        tl.load(partial_x24_ptr + high_offsets12, mask=mask12, other=0.0).to(tl.float32),
        axis=0,
    )

    scale24 = tl.load(scale24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    grad24 = tl.load(grad24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    scale12 = tl.load(scale12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)
    grad12 = tl.load(grad12_ptr + c12, mask=c12_mask, other=0.0).to(tl.float32)

    tl.store(sum24_out_ptr + c24, sum_x24, mask=c24_mask)
    tl.store(dot24_out_ptr + c24, _f32_mul(dot24, scale24), mask=c24_mask)
    tl.store(sum12_out_ptr + c12, sum_x12, mask=c12_mask)
    tl.store(dot12_out_ptr + c12, _f32_mul(dot12, scale12), mask=c12_mask)

    tl.store(mean24_ptr + c24, _f32_mul(sum_x24, INV_ROWS_), mask=c24_mask)
    tl.store(
        coeff24_ptr + c24,
        _f32_mul(_f32_mul(dot24, INV_ROWS_), _f32_mul(scale24, scale24)),
        mask=c24_mask,
    )
    tl.store(
        coeff12_ptr + c12,
        _f32_mul(_f32_mul(dot12, INV_ROWS_), _f32_mul(scale12, scale12)),
        mask=c12_mask,
    )
    tl.store(gain24_ptr + c24, _f32_mul(scale24, grad24), mask=c24_mask)
    tl.store(gain12_ptr + c12, _f32_mul(scale12, grad12), mask=c12_mask)


@triton.jit
def _store_outputs_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    input_mean24_ptr,
    arg6_ptr,
    input_mean12_ptr,
    mean24_ptr,
    coeff24_ptr,
    coeff12_ptr,
    gain24_ptr,
    gain12_ptr,
    out_copy_ptr,
    out24_ptr,
    out12_ptr,
    C24_: tl.constexpr,
    C12_: tl.constexpr,
    ROWS_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_ROWS_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    row_tile = tl.program_id(0)
    channel_tile = tl.program_id(1)
    rows = row_tile * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
    c24 = channel_tile * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    row_mask = rows < ROWS_
    c24_mask = c24 < C24_

    n = rows // HW_
    hw = rows - n * HW_
    contig_offsets = n[None, :] * (C24_ * HW_) + c24[:, None] * HW_ + hw[None, :]
    cl_offsets = n[None, :] * (C24_ * HW_) + hw[None, :] * C24_ + c24[:, None]
    mask24 = c24_mask[:, None] & row_mask[None, :]

    x24 = _rounded_add(arg0_ptr, arg1_ptr, contig_offsets, cl_offsets, mask24)
    tl.store(
        out_copy_ptr + cl_offsets,
        x24.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask24,
    )

    centered24 = _f32_sub(
        tl.load(arg2_ptr + cl_offsets, mask=mask24, other=0.0).to(tl.float32),
        tl.load(input_mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)[:, None],
    )
    mean24 = tl.load(mean24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    coeff24 = tl.load(coeff24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    gain24 = tl.load(gain24_ptr + c24, mask=c24_mask, other=0.0).to(tl.float32)
    out24 = _f32_mul(
        _f32_sub(_f32_sub(x24, _f32_mul(centered24, coeff24[:, None])), mean24[:, None]),
        gain24[:, None],
    )
    tl.store(
        out24_ptr + contig_offsets,
        out24.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask24,
    )

    c12 = tl.where(c24 >= C12_, c24 - C12_, 0)
    tail_mask = (c24 >= C12_) & c24_mask
    arg6_offsets = n[None, :] * (C12_ * HW_) + hw[None, :] * C12_ + c12[:, None]
    out12_offsets = arg6_offsets
    mask12 = tail_mask[:, None] & row_mask[None, :]
    centered12 = _f32_sub(
        tl.load(arg6_ptr + arg6_offsets, mask=mask12, other=0.0).to(tl.float32),
        tl.load(input_mean12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)[:, None],
    )
    coeff12 = tl.load(coeff12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)
    gain12 = tl.load(gain12_ptr + c12, mask=tail_mask, other=0.0).to(tl.float32)
    out12 = _f32_mul(
        _f32_sub(_f32_sub(x24, _f32_mul(centered12, coeff12[:, None])), mean24[:, None]),
        gain12[:, None],
    )
    tl.store(
        out12_ptr + out12_offsets,
        out12.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask12,
    )


@oracle_impl(hardware="B200", point="ccd5faf2")
def oracle_forward(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, _shape0, _shape1 = inputs
    device = arg0.device

    partial_x24 = torch.empty((NUM_GROUPS, C24), device=device, dtype=torch.float32)
    partial_dot24 = torch.empty((NUM_GROUPS, C24), device=device, dtype=torch.float32)
    partial_dot12 = torch.empty((NUM_GROUPS, C12), device=device, dtype=torch.float32)

    _partial_grouped_reductions_kernel[(NUM_GROUPS, triton.cdiv(C24, GROUP_BLOCK_C))](
        arg0,
        arg1,
        arg2,
        arg3,
        arg6,
        arg7,
        partial_x24,
        partial_dot24,
        partial_dot12,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        GROUP_ROWS_=GROUP_ROWS,
        R_BLOCK_=GROUP_R_BLOCK,
        BLOCK_C_=GROUP_BLOCK_C,
        num_warps=8,
    )

    out1 = torch.empty((C24,), device=device, dtype=torch.float32)
    out2 = torch.empty((C24,), device=device, dtype=torch.float32)
    out4 = torch.empty((C12,), device=device, dtype=torch.float32)
    out5 = torch.empty((C12,), device=device, dtype=torch.float32)
    mean24 = torch.empty((C24,), device=device, dtype=torch.float32)
    coeff24 = torch.empty((C24,), device=device, dtype=torch.float32)
    coeff12 = torch.empty((C12,), device=device, dtype=torch.float32)
    gain24 = torch.empty((C24,), device=device, dtype=torch.float32)
    gain12 = torch.empty((C12,), device=device, dtype=torch.float32)

    _finalize_reductions_kernel[(1,)](
        partial_x24,
        partial_dot24,
        partial_dot12,
        arg4,
        arg5,
        arg8,
        arg9,
        out1,
        out2,
        out4,
        out5,
        mean24,
        coeff24,
        coeff12,
        gain24,
        gain12,
        C24_=C24,
        C12_=C12,
        NUM_GROUPS_=NUM_GROUPS,
        BLOCK_GROUPS_=FINAL_BLOCK_GROUPS,
        INV_ROWS_=INV_ROWS,
        num_warps=8,
    )

    out0 = torch.empty_strided(
        (N, C24, H, W),
        (C24 * HW, 1, W * C24, C24),
        device=device,
        dtype=arg0.dtype,
    )
    out3 = torch.empty_strided(
        (N, C24, H, W),
        (C24 * HW, HW, W, 1),
        device=device,
        dtype=arg0.dtype,
    )
    out6 = torch.empty_strided(
        (N, C12, H, W),
        (C12 * HW, 1, W * C12, C12),
        device=device,
        dtype=arg6.dtype,
    )

    _store_outputs_kernel[
        (triton.cdiv(ROWS, STORE_BLOCK_ROWS), triton.cdiv(C24, STORE_BLOCK_C))
    ](
        arg0,
        arg1,
        arg2,
        arg3,
        arg6,
        arg7,
        mean24,
        coeff24,
        coeff12,
        gain24,
        gain12,
        out0,
        out3,
        out6,
        C24_=C24,
        C12_=C12,
        ROWS_=ROWS,
        HW_=HW,
        BLOCK_ROWS_=STORE_BLOCK_ROWS,
        BLOCK_C_=STORE_BLOCK_C,
        num_warps=4,
    )

    return out0, out1, out2, out3, out4, out5, out6
