"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GoogleFnet fp32 layer-norm-backward scope, including the returned input-gradient tensor, both `[768]` column reductions, and the `select_scatter` output that preserves lane 1 from the input tensor, whereas Inductor currently schedules the row-local reductions, sibling column reductions, input-gradient materialization, and select_scatter side output as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen cannot fuse row-local layer-norm-backward reductions with sibling cooperative column reductions and a full select_scatter side-output store in one producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled layer-norm-backward producer that emits required side outputs while cooperatively accumulating compatible column reductions."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
ROWS_PER_SPLIT = 4
BLOCK_C = 1024


@triton.jit
def _ln_backward_select_scatter_atomic_kernel(
    mm_ptr,
    residual_ptr,
    gamma_ptr,
    xhat_ptr,
    scale_ptr,
    full_ptr,
    grad_ptr,
    dgamma_ptr,
    dbeta_ptr,
    scatter_ptr,
    mm_stride_m: tl.constexpr,
    mm_stride_c: tl.constexpr,
    residual_stride_b: tl.constexpr,
    residual_stride_s: tl.constexpr,
    residual_stride_c: tl.constexpr,
    gamma_stride_c: tl.constexpr,
    xhat_stride_b: tl.constexpr,
    xhat_stride_s: tl.constexpr,
    xhat_stride_c: tl.constexpr,
    scale_stride_b: tl.constexpr,
    scale_stride_s: tl.constexpr,
    scale_stride_c: tl.constexpr,
    full_stride_b: tl.constexpr,
    full_stride_s: tl.constexpr,
    full_stride_c: tl.constexpr,
    full_stride_l: tl.constexpr,
    grad_stride_b: tl.constexpr,
    grad_stride_s: tl.constexpr,
    grad_stride_c: tl.constexpr,
    scatter_stride_b: tl.constexpr,
    scatter_stride_s: tl.constexpr,
    scatter_stride_c: tl.constexpr,
    scatter_stride_l: tl.constexpr,
    rows_per_split: tl.constexpr,
    seq: tl.constexpr,
    rows: tl.constexpr,
    hidden: tl.constexpr,
    block_c: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, block_c)
    col_mask = cols < hidden
    gamma = tl.load(gamma_ptr + cols * gamma_stride_c, mask=col_mask, other=0.0).to(tl.float32)

    acc_dgamma = tl.zeros((block_c,), dtype=tl.float32)
    acc_dbeta = tl.zeros((block_c,), dtype=tl.float32)

    for row_offset in tl.static_range(0, rows_per_split):
        row = pid * rows_per_split + row_offset
        row_active = row < rows
        batch = row // seq
        token = row - batch * seq
        mask = col_mask & row_active

        mm_offsets = row * mm_stride_m + cols * mm_stride_c
        residual_offsets = (
            batch * residual_stride_b
            + token * residual_stride_s
            + cols * residual_stride_c
        )
        xhat_offsets = (
            batch * xhat_stride_b
            + token * xhat_stride_s
            + cols * xhat_stride_c
        )
        scale_offset = batch * scale_stride_b + token * scale_stride_s
        grad_offsets = (
            batch * grad_stride_b
            + token * grad_stride_s
            + cols * grad_stride_c
        )
        scatter_lane0_offsets = (
            batch * scatter_stride_b
            + token * scatter_stride_s
            + cols * scatter_stride_c
        )
        full_lane1_offsets = (
            batch * full_stride_b
            + token * full_stride_s
            + cols * full_stride_c
            + full_stride_l
        )

        mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + xhat_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_offset + 0 * scale_stride_c, mask=row_active, other=0.0).to(tl.float32)

        add = mm + residual
        weighted = add * gamma
        row_sum = tl.sum(weighted, axis=0)
        row_dot = tl.sum(weighted * xhat, axis=0)
        grad = scale * (weighted * hidden - row_sum - xhat * row_dot)
        lane1 = tl.load(full_ptr + full_lane1_offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(grad_ptr + grad_offsets, grad, mask=mask)
        tl.store(scatter_ptr + scatter_lane0_offsets, grad, mask=mask)
        tl.store(scatter_ptr + scatter_lane0_offsets + scatter_stride_l, lane1, mask=mask)

        acc_dgamma += tl.where(mask, add * xhat, 0.0)
        acc_dbeta += tl.where(mask, add, 0.0)

    tl.atomic_add(dgamma_ptr + cols, acc_dgamma, sem="relaxed", mask=col_mask)
    tl.atomic_add(dbeta_ptr + cols, acc_dbeta, sem="relaxed", mask=col_mask)


# 356087b1: (T([16384,768], f32), T([32,512,768], f32), T([768], f32), T([32,512,768], f32), T([32,512,1], f32), T([32,512,768,2], f32), S([32,512,768]))
@oracle_impl(hardware="B200", point="356087b1", num_warps=8)
def oracle_forward(inputs, *, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    del _shape_param_0

    grad = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    dgamma = torch.empty_strided((HIDDEN,), (1,), device=arg0_1.device, dtype=torch.float32)
    dbeta = torch.empty_strided((HIDDEN,), (1,), device=arg0_1.device, dtype=torch.float32)
    scatter = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=arg5_1.device,
        dtype=arg5_1.dtype,
    )
    dgamma.zero_()
    dbeta.zero_()

    _ln_backward_select_scatter_atomic_kernel[(triton.cdiv(ROWS, ROWS_PER_SPLIT),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        grad,
        dgamma,
        dbeta,
        scatter,
        mm_stride_m=arg0_1.stride(0),
        mm_stride_c=arg0_1.stride(1),
        residual_stride_b=arg1_1.stride(0),
        residual_stride_s=arg1_1.stride(1),
        residual_stride_c=arg1_1.stride(2),
        gamma_stride_c=arg2_1.stride(0),
        xhat_stride_b=arg3_1.stride(0),
        xhat_stride_s=arg3_1.stride(1),
        xhat_stride_c=arg3_1.stride(2),
        scale_stride_b=arg4_1.stride(0),
        scale_stride_s=arg4_1.stride(1),
        scale_stride_c=arg4_1.stride(2),
        full_stride_b=arg5_1.stride(0),
        full_stride_s=arg5_1.stride(1),
        full_stride_c=arg5_1.stride(2),
        full_stride_l=arg5_1.stride(3),
        grad_stride_b=grad.stride(0),
        grad_stride_s=grad.stride(1),
        grad_stride_c=grad.stride(2),
        scatter_stride_b=scatter.stride(0),
        scatter_stride_s=scatter.stride(1),
        scatter_stride_c=scatter.stride(2),
        scatter_stride_l=scatter.stride(3),
        rows_per_split=ROWS_PER_SPLIT,
        seq=SEQ,
        rows=ROWS,
        hidden=HIDDEN,
        block_c=BLOCK_C,
        num_warps=num_warps,
    )
    return grad, dgamma, dbeta, scatter
