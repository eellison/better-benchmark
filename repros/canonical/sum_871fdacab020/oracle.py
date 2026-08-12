"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT bf16 masked scaled-attention softmax-backward row update in one Triton kernel, including the `[192,128,128] -> [16,12,128,128]` views, bool dropout-mask scaling, bf16 divide-by-8 logits rounding, broadcast bf16 mask fill before natural libdevice exp, fp32 row product reduction, fma.rn.f32 epilogue, explicit bf16 rounding, final broadcast mask fill, separate bf16 divide-by-8 rounding, and returned contiguous view, whereas Inductor lowers the decomposed view/convert/mul/div/where/sub/exp/div/sum/fma/cast/where/div/view graph through generic reduction scheduling around materialized producers; Inductor cannot do this today because its scheduler does not preserve the pre-exp bf16 mask/scale boundary, probability reconstruction, row reduction, fma epilogue, dtype boundaries, broadcast where operations, and post-cast scale as one B200-tuned full-scope row template; the fix is SCHEDULER_FUSION: add a guarded attention-backward row-reduction template that sinks the producer and epilogue while preserving bf16 scale/mask casts, libdevice exp, fma, broadcast masks, post-scale, and output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176
MASK_FILL_F32 = -998244352.0
POST_SCALE_F32 = 0.125


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
def _bert_masked_scaled_softmax_backward_bf16_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    final_mask_ptr,
    row_shift_ptr,
    row_denom_ptr,
    fill_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
    MASK_FILL: tl.constexpr,
    POST_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * N_COLS + cols[None, :]

    batch = rows // (HEADS * QUERY)
    query = rows % QUERY
    final_mask_offsets = (batch[:, None] * QUERY + query[:, None]) * N_COLS + cols[None, :]

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0, eviction_policy="evict_first").to(tl.float32)
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    final_mask = tl.load(final_mask_ptr + final_mask_offsets, mask=mask, other=0).to(tl.int1)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    scaled_logits = _round_to_bf16_f32(logits * POST_SCALE)
    masked_logits = tl.where(final_mask, MASK_FILL, scaled_logits)
    probs = libdevice.exp(masked_logits - row_shift[:, None]) / row_denom[:, None]

    scaled_grad = grad * (keep * DROPOUT_SCALE)
    product = scaled_grad * probs
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    fma = _fma_rn_f32(-probs, row_sum, product)
    rounded = _round_to_bf16_f32(fma)
    selected = tl.where(final_mask, fill, rounded)
    out = _round_to_bf16_f32(selected * POST_SCALE)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


# 202893ec: (T([192,128,128], bf16), T([16,12,128,128], b8), T([192,128,128], bf16), ...)
@oracle_impl(hardware="B200", point="202893ec", BLOCK_M=8, BLOCK_N=128, num_warps=4)
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
    n_cols = int(out_shape[-1])
    n_rows = arg0_1.numel() // n_cols

    _bert_masked_scaled_softmax_backward_bf16_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
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
        HEADS=int(arg1_1.shape[1]),
        QUERY=int(arg1_1.shape[2]),
        DROPOUT_SCALE=DROPOUT_SCALE_F32,
        MASK_FILL=MASK_FILL_F32,
        POST_SCALE=POST_SCALE_F32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
