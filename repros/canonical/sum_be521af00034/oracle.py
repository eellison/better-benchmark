"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DeBERTa attention softmax-backward row update in one Triton kernel, including the [192,512,512] to [8,24,512,512] view, bool dropout-mask scaling, bf16-to-fp32 logits, natural-exp probability reconstruction from row shift and denominator, fp32 product reduction, exact fma.rn.f32 epilogue, explicit bf16 rounding, broadcast final mask fill, and returned contiguous [192,512,512] view, whereas Inductor lowers the decomposed cast/view/mul/sub/exp/div/sum/fma/cast/where/view graph through generic reduction scheduling around materialized producers; Inductor cannot do this today because its scheduler does not preserve this fixed-width probability reconstruction, row reduction, fma epilogue, dtype boundary, and broadcast where as one B200-tuned full-scope row template; the fix is SCHEDULER_FUSION: add a guarded attention-backward row-reduction template that sinks the producer and epilogue while preserving libdevice exp, fma, bf16 cast, mask broadcast, and output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176


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
def _deberta_softmax_backward_bf16_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    final_mask_ptr,
    fill_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    HEADS: tl.constexpr,
    QUERY: tl.constexpr,
    SCALE: tl.constexpr,
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

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    final_mask = tl.load(final_mask_ptr + final_mask_offsets, mask=mask, other=0).to(tl.int1)
    fill = tl.load(fill_ptr)

    div = libdevice.exp(logits - row_shift[:, None]) / row_denom[:, None]
    scaled_grad = grad * (keep * SCALE)
    product = scaled_grad * div
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
    fma = _fma_rn_f32(-div, row_sum, product)
    rounded = fma.to(tl.bfloat16)
    out = tl.where(final_mask, fill, rounded)

    tl.store(out_ptr + offsets, out, mask=mask)


# (T([192,512,512], bf16), T([8,24,512,512], b8), T([8,24,512,512], bf16), T([8,24,512,1], f32), T([8,24,512,1], f32), T([8,1,512,512], b8), T([], bf16), S([8,24,512,512]), S([192,512,512]))
@oracle_impl(hardware="B200", point="1e831143", BLOCK_M=8, BLOCK_N=512, num_warps=8)
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
    ) = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols

    _deberta_softmax_backward_bf16_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
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
        HEADS=int(arg2_1.shape[1]),
        QUERY=int(arg2_1.shape[2]),
        SCALE=DROPOUT_SCALE_F32,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
