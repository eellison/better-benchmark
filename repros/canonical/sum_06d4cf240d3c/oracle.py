"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTa bf16 attention softmax-backward tail in one last-dimension Triton row kernel, including the `[192,512,512] -> [8,24,512,512]` view, bool dropout-mask conversion and scale, bf16 score promotion, natural-exp probability reconstruction from the saved row max and denominator, row product reduction, exact fma.rn epilogue, bf16 output cast, observable scalar-zero output, all-false broadcast mask output, and final contiguous `[192,512,512]` view, whereas Inductor lowers the decomposed view/convert/mul/sub/exp/div/sum/fma/cast/where graph as generic producer and reduction schedules with separate side-output materialization; Inductor cannot do this today because scheduler fusion does not sink this softmax-backward row reduction through the probability-reconstruction producer and required bf16/output epilogues while preserving the full returned-output scope; the fix is SCHEDULER_FUSION: add a guarded softmax-backward row template that keeps the exp/div producer, row dot product, fma epilogue, bf16 cast, and constant side outputs in one schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512
N_ROWS = BATCH * HEADS * Q_LEN
SCALE = 1.1111111640930176


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
def _deberta_softmax_backward_kernel(
    grad_ptr,
    keep_ptr,
    score_ptr,
    row_shift_ptr,
    row_denom_ptr,
    scalar_zero_ptr,
    false_mask_ptr,
    out_ptr,
    N_ROWS_: tl.constexpr,
    K_LEN_: tl.constexpr,
    HEADS_: tl.constexpr,
    Q_LEN_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    offsets = rows[:, None] * K_LEN_ + cols[None, :]
    row_mask = rows < N_ROWS_

    grad = tl.load(grad_ptr + offsets).to(tl.float32)
    keep = tl.load(keep_ptr + offsets).to(tl.float32)
    score = tl.load(score_ptr + offsets).to(tl.float32)
    row_shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    probs = libdevice.exp(score - row_shift[:, None]) / row_denom[:, None]
    scaled_grad = grad * (keep * SCALE_)
    product = scaled_grad * probs
    row_sum = tl.sum(product, axis=1)
    out = _fma_rn_f32(-probs, row_sum[:, None], product).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(out_ptr + offsets, out, mask=row_mask[:, None])
    tl.store(scalar_zero_ptr, 0.0, mask=tl.program_id(0) == 0)

    q = rows % Q_LEN_
    head = (rows // Q_LEN_) % HEADS_
    batch = rows // (HEADS_ * Q_LEN_)
    false_offsets = (batch[:, None] * Q_LEN_ + q[:, None]) * K_LEN_ + cols[None, :]
    tl.store(false_mask_ptr + false_offsets, cols[None, :] < 0, mask=(head[:, None] == 0) & row_mask[:, None])


# c2297120: hf DeBERTaV2 train softmax-backward bf16 output plus constant side outputs.
@oracle_impl(hardware="B200", point="c2297120", BLOCK_M=2, BLOCK_K=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    scalar_zero = torch.empty((), device=arg0_1.device, dtype=torch.bfloat16)
    false_mask = torch.empty_strided(
        (BATCH, 1, Q_LEN, K_LEN),
        (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        (BATCH * HEADS, Q_LEN, K_LEN),
        (Q_LEN * K_LEN, K_LEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _deberta_softmax_backward_kernel[(triton.cdiv(N_ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        scalar_zero,
        false_mask,
        out,
        N_ROWS_=N_ROWS,
        K_LEN_=K_LEN,
        HEADS_=HEADS,
        Q_LEN_=Q_LEN,
        SCALE_=SCALE,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=4,
    )
    return scalar_zero, false_mask, out
