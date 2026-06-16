"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Visformer attention-backward recompute scope in one Triton row kernel, including the `[768,49,49]` gradient view, cropped `[768,56,56]` bf16 logits view, broadcast f32/bool row inputs, exact natural exp/div probability reconstruction, fp32 product sum, prims.fma-equivalent epilogue, explicit bf16 cast, final bf16 scale, and returned contiguous `[768,49,49]` view, whereas Inductor lowers the captured slice/view/cast/where/exp/div/mul/sum/fma/cast/mul graph through generic pointwise and reduction scheduling; Inductor cannot do this today because the scheduler does not keep this fixed-width row producer, reduction, and dependent full-row bf16 epilogue resident across the sliced mixed-stride inputs; the fix is SCHEDULER_FUSION: add a guarded row-reduction epilogue schedule that sinks the probability recompute, exact f32 reduction, fused multiply-add, bf16 rounding boundaries, and view-only output layout into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SCALE = 0.08838834764831845


@triton.jit
def _round_to_bf16_f32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _visformer_softmax_backward_49_kernel(
    grad_ptr,
    logits_ptr,
    row_shift_true_ptr,
    row_shift_false_ptr,
    branch_ptr,
    row_denom_ptr,
    out_ptr,
    SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < 37632
    col_mask = cols < 49
    mask = row_mask[:, None] & col_mask[None, :]

    flat_bh = rows // 49
    q_idx = rows - flat_bh * 49
    compact_offsets = rows[:, None] * 49 + cols[None, :]
    logits_offsets = flat_bh[:, None] * 3136 + q_idx[:, None] * 56 + cols[None, :]

    grad = tl.load(grad_ptr + compact_offsets, mask=mask, other=0.0).to(tl.float32)
    logits = tl.load(logits_ptr + logits_offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift_true = tl.load(row_shift_true_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_shift_false = tl.load(row_shift_false_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    branch = tl.load(branch_ptr + rows, mask=row_mask, other=0) != 0
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    logits_scaled_bf16 = _round_to_bf16_f32(logits * SCALE_)
    branch_scaled_sub = (logits - row_shift_true[:, None]) * SCALE_
    branch_sub_scaled = logits_scaled_bf16 - row_shift_false[:, None]
    exp_arg = tl.where(branch[:, None], branch_scaled_sub, branch_sub_scaled)
    probs = libdevice.exp(exp_arg) / row_denom[:, None]

    product = grad * probs
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    fma = tl.fma(-probs, row_sum, product)
    rounded = _round_to_bf16_f32(fma)
    out = (rounded * SCALE_).to(tl.bfloat16)
    tl.store(out_ptr + compact_offsets, out, mask=mask)


# cdec8791: Visformer small train, bf16 softmax-backward recompute on 49-wide rows.
@oracle_impl(hardware="B200", point="cdec8791", BLOCK_M=16, BLOCK_N=64, num_warps=4)
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
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _visformer_softmax_backward_49_kernel[(triton.cdiv(37632, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out,
        SCALE_=SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
