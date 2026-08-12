"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Blenderbot masked attention-softmax scope, including the scalar bool-mask select, bf16 score+bias add, all-`-inf` row guard, natural-exp stable softmax, zero fill for fully masked rows, and the returned `[512, 128, 128]` view from the same bf16 storage; Inductor already recognizes much of the online-softmax region but still carries the explicit `any(eq(-inf))` predicate and generic row schedule; Inductor cannot fully eliminate this today because the all-masked predicate is not folded into the row-max predicate while preserving bf16 output casting and aliasing; the fix is ALGEBRAIC_ELIMINATION: derive the zero-row guard inside the row-softmax lowering and emit the view-preserving bf16 store directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import math as tl_math

from oracle_harness import oracle_impl


N_ROWS = 65536
K = 128


@triton.jit
def _masked_softmax_kernel(
    scores_ptr,
    mask_ptr,
    true_scalar_ptr,
    false_scalar_ptr,
    out_ptr,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    cols = tl.arange(0, RBLOCK)[None, :]
    q = rows % 128

    score = tl.load(scores_ptr + cols + 128 * rows).to(tl.float32)
    mask = tl.load(mask_ptr + q, eviction_policy="evict_last").to(tl.int1)
    true_scalar = tl.load(true_scalar_ptr).to(tl.float32)
    false_scalar = tl.load(false_scalar_ptr).to(tl.float32)
    bias = tl.where(mask, true_scalar, false_scalar)
    added = score + bias

    row_max = triton_helpers.max2(added, 1)[:, None]
    has_any = row_max != -float("inf")
    expv = tl_math.exp(added - row_max)
    denom = tl.sum(expv, 1)[:, None]
    softmax = expv / denom
    out = tl.where(has_any == 0, 0.0, softmax.to(tl.float32))
    tl.store(out_ptr + cols + 128 * rows, out)


@oracle_impl(hardware="B200", point="56ba83ca", XBLOCK=16, num_warps=4)
def oracle_forward(inputs, *, XBLOCK: int, num_warps: int):
    mask, true_scalar, false_scalar, scores = inputs[:4]
    out = torch.empty_strided(
        (16, 32, 128, 128),
        (524288, 16384, 128, 1),
        device=scores.device,
        dtype=torch.bfloat16,
    )
    _masked_softmax_kernel[(triton.cdiv(N_ROWS, XBLOCK),)](
        scores,
        mask,
        true_scalar,
        false_scalar,
        out,
        XBLOCK=XBLOCK,
        RBLOCK=K,
        num_warps=num_warps,
    )
    return out, torch.as_strided(out, (512, 128, 128), (16384, 128, 1))
