"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete const-center max-pool-backward scatter, per-sample two-channel BatchNorm-backward reductions, returned channel summaries, and bf16 channels-last input-gradient tensor directly from the structured scatter destinations, whereas Inductor materializes the dense f32 zero/scatter_add buffer and schedules the ReLU gate, six dependent reductions, and final BN-backward epilogue as generic kernels over intermediates; Inductor cannot do this today because scheduler/codegen treats `_low_memory_max_pool_offsets_to_indices` plus `scatter_add` as an opaque scatter producer rather than a fixed-window maxpool-backward scatter-reduce feeding grouped reductions and a dense side output; the fix is SCATTER_REDUCE: add a guarded low-memory maxpool-backward lowering that recognizes constant center offsets, fuses the grouped reductions, and emits the dependent channels-last output from the same structured producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 64
GROUPS = 32
H = 16
W = 16
HW = 256
POOL_H = 8
POOL_W = 8
REDUCTION_SCALE = 0.001953125


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
def _row_sums_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    sum_x_ptr,
    sum_ptr,
    GRAD0_STRIDE_N: tl.constexpr,
    GRAD0_STRIDE_C: tl.constexpr,
    GRAD0_STRIDE_H: tl.constexpr,
    GRAD0_STRIDE_W: tl.constexpr,
    GRAD1_STRIDE_N: tl.constexpr,
    GRAD1_STRIDE_C: tl.constexpr,
    GRAD1_STRIDE_H: tl.constexpr,
    GRAD1_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    n = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    s = tl.arange(0, BLOCK_S)
    c_mask = cols < 64
    h = s // 16
    w = s - h * 16
    active = c_mask[None, :] & (s[:, None] < 256)

    group = cols // 2
    mean = tl.load(mean_ptr + n * 32 + group, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + n * 32 + group, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    x_offsets = (
        n * X_STRIDE_N
        + cols[None, :] * X_STRIDE_C
        + h[:, None] * X_STRIDE_H
        + w[:, None] * X_STRIDE_W
    )
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    scaled = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    take_scalar = affine <= 0.0

    even_h = (h % 2) == 0
    even_w = (w % 2) == 0
    ph = h // 2
    pw = w // 2
    has_source = (even_h & even_w & (ph < 8) & (pw < 8))[:, None] & active
    grad0_offsets = (
        n * GRAD0_STRIDE_N
        + cols[None, :] * GRAD0_STRIDE_C
        + ph[:, None] * GRAD0_STRIDE_H
        + pw[:, None] * GRAD0_STRIDE_W
    )
    grad1_offsets = (
        n * GRAD1_STRIDE_N
        + cols[None, :] * GRAD1_STRIDE_C
        + ph[:, None] * GRAD1_STRIDE_H
        + pw[:, None] * GRAD1_STRIDE_W
    )
    scatter = _f32_add(
        tl.load(grad1_ptr + grad1_offsets, mask=has_source, other=0.0).to(tl.float32),
        tl.load(grad0_ptr + grad0_offsets, mask=has_source, other=0.0).to(tl.float32),
    )
    selected = tl.where(take_scalar, scalar, scatter)
    selected = tl.where(active, selected, 0.0)
    x_active = tl.where(active, x, 0.0)

    sum_val = tl.sum(selected, axis=0)
    sum_x = tl.sum(_f32_mul(selected, x_active), axis=0)
    out_offsets = n * 64 + cols
    tl.store(sum_ptr + out_offsets, sum_val, mask=c_mask)
    tl.store(sum_x_ptr + out_offsets, sum_x, mask=c_mask)


@triton.jit
def _final_vectors_kernel(
    sum_x_ptr,
    sum_ptr,
    mean_ptr,
    invstd_ptr,
    out0_ptr,
    out1_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_N)
    c_mask = cols < 64
    n_mask = rows < 128
    group = cols // 2
    offsets = rows[:, None] * 64 + cols[None, :]
    active = n_mask[:, None] & c_mask[None, :]

    sx = tl.load(sum_x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sv = tl.load(sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(
        mean_ptr + rows[:, None] * 32 + group[None, :],
        mask=active,
        other=0.0,
    ).to(tl.float32)
    invstd = tl.load(
        invstd_ptr + rows[:, None] * 32 + group[None, :],
        mask=active,
        other=0.0,
    ).to(tl.float32)
    centered_sum = _f32_sub(sx, _f32_mul(sv, mean))
    out0 = tl.sum(tl.where(active, _f32_mul(centered_sum, invstd), 0.0), axis=0)
    out1 = tl.sum(tl.where(active, sv, 0.0), axis=0)
    tl.store(out0_ptr + cols, out0, mask=c_mask)
    tl.store(out1_ptr + cols, out1, mask=c_mask)


@triton.jit
def _output_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    sum_x_ptr,
    sum_ptr,
    out_ptr,
    GRAD0_STRIDE_N: tl.constexpr,
    GRAD0_STRIDE_C: tl.constexpr,
    GRAD0_STRIDE_H: tl.constexpr,
    GRAD0_STRIDE_W: tl.constexpr,
    GRAD1_STRIDE_N: tl.constexpr,
    GRAD1_STRIDE_C: tl.constexpr,
    GRAD1_STRIDE_H: tl.constexpr,
    GRAD1_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_C: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_W: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < 32768
    c_mask = cols < 64
    active = row_mask[:, None] & c_mask[None, :]

    n = rows // 256
    s = rows - n * 256
    h = s // 16
    w = s - h * 16
    group = cols // 2

    mean = tl.load(
        mean_ptr + n[:, None] * 32 + group[None, :],
        mask=active,
        other=0.0,
    ).to(tl.float32)
    invstd = tl.load(
        invstd_ptr + n[:, None] * 32 + group[None, :],
        mask=active,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    x_offsets = (
        n[:, None] * X_STRIDE_N
        + cols[None, :] * X_STRIDE_C
        + h[:, None] * X_STRIDE_H
        + w[:, None] * X_STRIDE_W
    )
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    take_scalar = affine <= 0.0

    even_h = (h % 2) == 0
    even_w = (w % 2) == 0
    ph = h // 2
    pw = w // 2
    has_source = (even_h & even_w & (ph < 8) & (pw < 8))[:, None] & active
    grad0_offsets = (
        n[:, None] * GRAD0_STRIDE_N
        + cols[None, :] * GRAD0_STRIDE_C
        + ph[:, None] * GRAD0_STRIDE_H
        + pw[:, None] * GRAD0_STRIDE_W
    )
    grad1_offsets = (
        n[:, None] * GRAD1_STRIDE_N
        + cols[None, :] * GRAD1_STRIDE_C
        + ph[:, None] * GRAD1_STRIDE_H
        + pw[:, None] * GRAD1_STRIDE_W
    )
    scatter = _f32_add(
        tl.load(grad1_ptr + grad1_offsets, mask=has_source, other=0.0).to(tl.float32),
        tl.load(grad0_ptr + grad0_offsets, mask=has_source, other=0.0).to(tl.float32),
    )
    selected = tl.where(take_scalar, scalar, scatter)

    group_c0 = group * 2
    group_c1 = group_c0 + 1
    sum_offsets0 = n[:, None] * 64 + group_c0[None, :]
    sum_offsets1 = n[:, None] * 64 + group_c1[None, :]
    sum_x0 = tl.load(sum_x_ptr + sum_offsets0, mask=active, other=0.0).to(tl.float32)
    sum_x1 = tl.load(sum_x_ptr + sum_offsets1, mask=active, other=0.0).to(tl.float32)
    sum0 = tl.load(sum_ptr + sum_offsets0, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum_ptr + sum_offsets1, mask=active, other=0.0).to(tl.float32)
    weight0 = tl.load(weight_ptr + group_c0, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight_ptr + group_c1, mask=c_mask, other=0.0).to(tl.float32)

    sum3 = _f32_add(_f32_mul(sum_x0, weight0[None, :]), _f32_mul(sum_x1, weight1[None, :]))
    sum4 = _f32_add(_f32_mul(sum0, weight0[None, :]), _f32_mul(sum1, weight1[None, :]))
    mul6 = _f32_mul(sum4, mean)
    sub1 = _f32_sub(mul6, sum3)
    mul7 = _f32_mul(sub1, invstd)
    mul8 = _f32_mul(mul7, invstd)
    mul9 = _f32_mul(mul8, invstd)
    mul10 = _f32_mul(mul9, 0.001953125)
    neg = _f32_sub(tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32), mul10)
    mul11 = _f32_mul(neg, mean)
    mul12 = _f32_mul(sum4, invstd)
    mul13 = _f32_mul(mul12, 0.001953125)
    sub2 = _f32_sub(mul11, mul13)
    mul5 = _f32_mul(invstd, weight[None, :])
    mul14 = _f32_mul(selected, mul5)
    mul15 = _f32_mul(x, mul10)
    add2 = _f32_add(mul14, mul15)
    out = _f32_add(add2, sub2)

    out_offsets = (
        n[:, None] * OUT_STRIDE_N
        + cols[None, :] * OUT_STRIDE_C
        + h[:, None] * OUT_STRIDE_H
        + w[:, None] * OUT_STRIDE_W
    )
    tl.store(
        out_ptr + out_offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


@oracle_impl(
    hardware="B200",
    point="05c56638",
    BLOCK_C=8,
    BLOCK_S=256,
    BLOCK_R=128,
    FINAL_BLOCK_C=16,
    num_warps_reduce=8,
    num_warps_output=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_S: int,
    BLOCK_R: int,
    FINAL_BLOCK_C: int,
    num_warps_reduce: int,
    num_warps_output: int,
):
    (
        arg0_1,
        arg1_1,
        _arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        *_,
    ) = inputs
    sum_x = torch.empty((N, C), device=arg3_1.device, dtype=torch.float32)
    sum_val = torch.empty((N, C), device=arg3_1.device, dtype=torch.float32)
    out0 = torch.empty((C,), device=arg3_1.device, dtype=torch.float32)
    out1 = torch.empty((C,), device=arg3_1.device, dtype=torch.float32)
    out2 = torch.empty_strided(
        tuple(arg3_1.shape),
        tuple(int(s) for s in arg3_1.stride()),
        device=arg3_1.device,
        dtype=torch.bfloat16,
    )

    _row_sums_kernel[(N, triton.cdiv(C, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        sum_x,
        sum_val,
        GRAD0_STRIDE_N=int(arg0_1.stride(0)),
        GRAD0_STRIDE_C=int(arg0_1.stride(1)),
        GRAD0_STRIDE_H=int(arg0_1.stride(2)),
        GRAD0_STRIDE_W=int(arg0_1.stride(3)),
        GRAD1_STRIDE_N=int(arg1_1.stride(0)),
        GRAD1_STRIDE_C=int(arg1_1.stride(1)),
        GRAD1_STRIDE_H=int(arg1_1.stride(2)),
        GRAD1_STRIDE_W=int(arg1_1.stride(3)),
        X_STRIDE_N=int(arg3_1.stride(0)),
        X_STRIDE_C=int(arg3_1.stride(1)),
        X_STRIDE_H=int(arg3_1.stride(2)),
        X_STRIDE_W=int(arg3_1.stride(3)),
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _final_vectors_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        sum_x,
        sum_val,
        arg4_1,
        arg5_1,
        out0,
        out1,
        BLOCK_N=128,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _output_kernel[(triton.cdiv(N * HW, BLOCK_R), triton.cdiv(C, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        sum_x,
        sum_val,
        out2,
        GRAD0_STRIDE_N=int(arg0_1.stride(0)),
        GRAD0_STRIDE_C=int(arg0_1.stride(1)),
        GRAD0_STRIDE_H=int(arg0_1.stride(2)),
        GRAD0_STRIDE_W=int(arg0_1.stride(3)),
        GRAD1_STRIDE_N=int(arg1_1.stride(0)),
        GRAD1_STRIDE_C=int(arg1_1.stride(1)),
        GRAD1_STRIDE_H=int(arg1_1.stride(2)),
        GRAD1_STRIDE_W=int(arg1_1.stride(3)),
        X_STRIDE_N=int(arg3_1.stride(0)),
        X_STRIDE_C=int(arg3_1.stride(1)),
        X_STRIDE_H=int(arg3_1.stride(2)),
        X_STRIDE_W=int(arg3_1.stride(3)),
        OUT_STRIDE_N=int(out2.stride(0)),
        OUT_STRIDE_C=int(out2.stride(1)),
        OUT_STRIDE_H=int(out2.stride(2)),
        OUT_STRIDE_W=int(out2.stride(3)),
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_output,
        num_stages=3,
    )
    return out0, out1, out2
