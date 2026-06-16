"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Visformer attention-backward recompute scope in one Triton row kernel, including the `[768,196,196]` gradient view, cropped `[768,200,200]` bf16 logits view, broadcast f32/bool row inputs, exact natural exp/div probability reconstruction, fp32 product sum, prims.fma-equivalent epilogue, explicit bf16 cast, final bf16 scale, and returned contiguous `[768,196,196]` view, whereas Inductor lowers the captured slice/view/cast/where/exp/div/mul/sum/fma/cast/mul graph through generic pointwise and reduction scheduling; Inductor cannot do this today because the scheduler does not keep this fixed-width row producer, reduction, and dependent full-row bf16 epilogue resident across the sliced mixed-stride inputs; the fix is SCHEDULER_FUSION: add a guarded row-reduction epilogue schedule that sinks the probability recompute, exact f32 reduction, fused multiply-add, bf16 rounding boundaries, and view-only output layout into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _visformer_recompute_softmax_bwd_kernel(
    grad_ptr,
    logits_ptr,
    row_arg2_ptr,
    row_arg3_ptr,
    row_mask_ptr,
    row_denom_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    LOGITS_STRIDE_BH: tl.constexpr,
    LOGITS_STRIDE_Q: tl.constexpr,
    ROW_ARG2_STRIDE_B: tl.constexpr,
    ROW_ARG2_STRIDE_H: tl.constexpr,
    ROW_ARG2_STRIDE_Q: tl.constexpr,
    ROW_ARG3_STRIDE_B: tl.constexpr,
    ROW_ARG3_STRIDE_H: tl.constexpr,
    ROW_ARG3_STRIDE_Q: tl.constexpr,
    ROW_MASK_STRIDE_B: tl.constexpr,
    ROW_MASK_STRIDE_H: tl.constexpr,
    ROW_MASK_STRIDE_Q: tl.constexpr,
    ROW_DENOM_STRIDE_B: tl.constexpr,
    ROW_DENOM_STRIDE_H: tl.constexpr,
    ROW_DENOM_STRIDE_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // N_COLS
    q_idx = rows - flat_bh * N_COLS
    batch = flat_bh // 6
    head = flat_bh - batch * 6

    compact_offsets = rows[:, None] * N_COLS + cols[None, :]
    logits_offsets = (
        flat_bh[:, None] * LOGITS_STRIDE_BH
        + q_idx[:, None] * LOGITS_STRIDE_Q
        + cols[None, :]
    )
    arg2_offsets = (
        batch * ROW_ARG2_STRIDE_B
        + head * ROW_ARG2_STRIDE_H
        + q_idx * ROW_ARG2_STRIDE_Q
    )
    arg3_offsets = (
        batch * ROW_ARG3_STRIDE_B
        + head * ROW_ARG3_STRIDE_H
        + q_idx * ROW_ARG3_STRIDE_Q
    )
    mask_offsets = (
        batch * ROW_MASK_STRIDE_B
        + head * ROW_MASK_STRIDE_H
        + q_idx * ROW_MASK_STRIDE_Q
    )
    denom_offsets = (
        batch * ROW_DENOM_STRIDE_B
        + head * ROW_DENOM_STRIDE_H
        + q_idx * ROW_DENOM_STRIDE_Q
    )

    grad = tl.load(grad_ptr + compact_offsets, mask=mask, other=0.0).to(tl.float32)
    logits = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    arg2 = tl.load(row_arg2_ptr + arg2_offsets, mask=row_mask, other=0.0).to(tl.float32)
    arg3 = tl.load(row_arg3_ptr + arg3_offsets, mask=row_mask, other=0.0).to(tl.float32)
    choose_scaled_sub = tl.load(row_mask_ptr + mask_offsets, mask=row_mask, other=0)
    denom = tl.load(row_denom_ptr + denom_offsets, mask=row_mask, other=1.0).to(tl.float32)

    logits_scaled_bf16 = _round_bf16_to_fp32(logits * 0.125)
    branch_scaled_sub = (logits - arg2[:, None]) * 0.125
    branch_sub_scaled = logits_scaled_bf16 - arg3[:, None]
    exp_arg = tl.where(choose_scaled_sub[:, None], branch_scaled_sub, branch_sub_scaled)
    probs = libdevice.exp(exp_arg) / denom[:, None]

    product = grad * probs
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
    fma = tl.fma(-probs, row_sum, product)
    rounded = fma.to(tl.bfloat16).to(tl.float32)
    out = (rounded * 0.125).to(tl.bfloat16)

    tl.store(out_ptr + compact_offsets, out, mask=mask)


# 66d9f76b: (T([768,196,196], bf16), T([768,200,200], bf16), T([128,6,196,1], f32, stride=(1184,196,1,1)), ...)
@oracle_impl(hardware="B200", point="66d9f76b", BLOCK_M=4, BLOCK_N=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    n_cols = out_shape[2]
    n_rows = out_shape[0] * out_shape[1]
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _visformer_recompute_softmax_bwd_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        LOGITS_STRIDE_BH=arg1_1.stride(0),
        LOGITS_STRIDE_Q=arg1_1.stride(1),
        ROW_ARG2_STRIDE_B=arg2_1.stride(0),
        ROW_ARG2_STRIDE_H=arg2_1.stride(1),
        ROW_ARG2_STRIDE_Q=arg2_1.stride(2),
        ROW_ARG3_STRIDE_B=arg3_1.stride(0),
        ROW_ARG3_STRIDE_H=arg3_1.stride(1),
        ROW_ARG3_STRIDE_Q=arg3_1.stride(2),
        ROW_MASK_STRIDE_B=arg4_1.stride(0),
        ROW_MASK_STRIDE_H=arg4_1.stride(1),
        ROW_MASK_STRIDE_Q=arg4_1.stride(2),
        ROW_DENOM_STRIDE_B=arg5_1.stride(0),
        ROW_DENOM_STRIDE_H=arg5_1.stride(1),
        ROW_DENOM_STRIDE_Q=arg5_1.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
