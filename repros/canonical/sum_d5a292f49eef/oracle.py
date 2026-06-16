"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete T5/MT5 bf16 attention softmax-backward row update in one Triton kernel, including the metadata-only views, bool-to-bf16 dropout scale, bf16-rounded dropped-gradient product, bf16-to-fp32 logits, natural-exp probability reconstruction from row shift and denominator, fp32 row product reduction, exact fma.rn.f32 epilogue, final bf16 cast, and returned contiguous view, whereas Inductor lowers the decomposed view/cast/mul/cast/view/cast/sub/exp/div/sum/fma/cast/view graph through generic reduction scheduling around materialized producers; Inductor cannot do this today because its scheduler/codegen does not preserve the explicit bf16 rounding boundaries around dropout while sinking probability reconstruction and the dependent fma epilogue into one shape-specialized row program; the fix is SCHEDULER_FUSION: add a guarded attention-backward row-reduction template that preserves bf16 producer casts, libdevice exp, fma.rn semantics, and output layout."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


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
def _t5_softmax_backward_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
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

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.int1)
    logits = tl.load(logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    keep_scale = (keep.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
    dropped_grad = (grad * keep_scale).to(tl.bfloat16).to(tl.float32)
    probs = libdevice.exp(logits - row_shift[:, None]) / row_denom[:, None]
    product = dropped_grad * probs
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None].to(tl.float32)
    out = _fma_rn_f32(-probs, row_sum, product)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _t5_softmax_backward_large_kernel(
    grad_ptr,
    keep_ptr,
    logits_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N_ROWS
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
    chunk_cols = tl.arange(0, CHUNK)

    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    for start in tl.static_range(0, 1024, CHUNK):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < N_COLS)
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
        ).to(tl.int1)
        logits = tl.load(
            logits_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        keep_scale = (keep.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
        dropped_grad = (grad * keep_scale).to(tl.bfloat16).to(tl.float32)
        probs = libdevice.exp(logits - row_shift[:, None]) / row_denom[:, None]
        product = tl.where(mask, dropped_grad * probs, 0.0)
        row_sum += tl.sum(product, axis=1)

    for start in tl.static_range(0, 1024, CHUNK):
        cols = start + chunk_cols
        mask = row_mask[:, None] & (cols[None, :] < N_COLS)
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
        ).to(tl.int1)
        logits = tl.load(
            logits_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        keep_scale = (keep.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)
        dropped_grad = (grad * keep_scale).to(tl.bfloat16).to(tl.float32)
        probs = libdevice.exp(logits - row_shift[:, None]) / row_denom[:, None]
        product = dropped_grad * probs
        out = _fma_rn_f32(-probs, row_sum[:, None], product)

        tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=mask)


def _launch_t5_softmax_backward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_2),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols

    if n_cols == 1024:
        _t5_softmax_backward_large_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            out,
            N_ROWS=n_rows,
            N_COLS=n_cols,
            SCALE=DROPOUT_SCALE,
            BLOCK_M=BLOCK_M,
            CHUNK=512,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _t5_softmax_backward_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            out,
            N_ROWS=n_rows,
            N_COLS=n_cols,
            SCALE=DROPOUT_SCALE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=3,
        )
    return out


# 293147ee: (T([192,128,128], bf16), T([32,6,128,128], b8), T([192,128,128], bf16), T([32,6,128,1], f32), T([32,6,128,1], f32), S([32,6,128,128]), S([32,6,128,128]), S([192,128,128]))
@oracle_impl(hardware="B200", point="293147ee", BLOCK_M=8, BLOCK_N=128, num_warps=4)
# 64b0de65: (T([64,1024,1024], bf16), T([8,8,1024,1024], b8), T([64,1024,1024], bf16), T([8,8,1024,1], f32), T([8,8,1024,1], f32), S([8,8,1024,1024]), S([8,8,1024,1024]), S([64,1024,1024]))
@oracle_impl(hardware="B200", point="64b0de65", BLOCK_M=1, BLOCK_N=1024, num_warps=1)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    return _launch_t5_softmax_backward(
        inputs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )
