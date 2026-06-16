"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LayoutLM bf16 attention softmax-backward row update in one Triton kernel, including the shape-param views, bool-to-bf16 dropout scaling, bf16-rounded dropped-gradient product, both captured score-scale branches with their distinct bf16/fp32 rounding boundaries, natural-exp probability reconstruction, fp32 row product reduction, exact fma.rn.f32 epilogue, final bf16 cast, bf16 multiply-by-0.125 rounding, and returned contiguous view, whereas Inductor lowers the decomposed view/convert/mul/sub/where/exp/div/sum/neg/fma/cast/mul/view graph through generic reduction scheduling around materialized producers; Inductor cannot do this today because its scheduler/codegen does not preserve these mixed bf16/fp32 producer boundaries while sinking probability reconstruction and the dependent fma epilogue into one shape-specialized row program; the fix is SCHEDULER_FUSION: add a guarded attention-backward row-reduction template that preserves bf16 producer casts, libdevice exp, fma.rn semantics, the post-fma bf16 scaling, and direct output-layout emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112
POST_SCALE = 0.125


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _layoutlm_softmax_backward_scaled_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    row_shift_true_ptr,
    row_shift_false_ptr,
    branch_ptr,
    row_denom_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    POST_SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * N_COLS + cols[None, :]

    grad = tl.load(
        grad_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    keep = tl.load(
        keep_ptr + offsets,
        mask=mask,
        other=0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    logits = tl.load(
        logits_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    row_shift_true = tl.load(
        row_shift_true_ptr + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    row_shift_false = tl.load(
        row_shift_false_ptr + rows,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    branch = tl.load(branch_ptr + rows, mask=row_mask, other=0).to(tl.int1)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    keep_scale = _round_to_bf16_f32(keep * DROPOUT_SCALE_)
    dropped_grad = _round_to_bf16_f32(grad * keep_scale)

    logits_scaled_bf16 = _round_to_bf16_f32(logits * POST_SCALE_)
    true_scores = (logits - row_shift_true[:, None]) * POST_SCALE_
    false_scores = logits_scaled_bf16 - row_shift_false[:, None]
    scores = tl.where(branch[:, None], true_scores, false_scores)

    probs = libdevice.exp(scores) / row_denom[:, None]
    product = dropped_grad * probs
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    fma = _fma_rn_f32(-probs, row_sum, product)
    rounded = _round_to_bf16_f32(fma)
    out = _round_to_bf16_f32(rounded * POST_SCALE_)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# 03fa1c14: (T([384,512,512], bf16), T([32,12,512,512], b8), T([384,512,512], bf16), ...)
@oracle_impl(hardware="B200", point="03fa1c14", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
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
    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols

    _layoutlm_softmax_backward_scaled_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        POST_SCALE_=POST_SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
