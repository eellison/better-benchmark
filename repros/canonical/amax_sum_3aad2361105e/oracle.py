"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete static large-vocabulary bf16 cross-entropy forward scope by reading each logits row once with online fp32 max and denominator accumulators, safely gathering the target logit, applying the observable bf16 log-softmax rounding before the negation, and returning the bf16 per-row ignored-label-masked loss vector, whereas Inductor lowers the decomposed cast/amax/sub/exp/sum/log/log-softmax-cast/gather/neg/mask graph as generic row reductions and pointwise/gather work that materializes and rereads full row-sized intermediates; Inductor cannot do this today because its pattern library does not canonicalize ignore-index cross entropy into an online row template while preserving the bf16 log-softmax cast boundary and output-only gathered loss scope; the fix is NEW_PATTERN: add a guarded cross-entropy lowering that emits online row accumulators plus a target-gather epilogue directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _cross_entropy_forward_kernel(
    labels_ptr,
    logits_ptr,
    out_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N_COLS

    label = tl.load(labels_ptr + row)
    valid = label != -100
    safe_label = tl.where(valid, label, 0)
    target = tl.load(logits_ptr + row_start + safe_label, mask=valid, other=0.0).to(tl.float32)

    row_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for block_start in tl.range(0, N_COLS, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS
        logits = tl.load(
            logits_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(row_max, block_max)
        denom = _f32_add(
            denom * libdevice.exp(_f32_sub(row_max, next_max)),
            tl.sum(libdevice.exp(_f32_sub(logits, next_max)), axis=0),
        )
        row_max = next_max

    log_denom = libdevice.log(denom)
    logp = _f32_sub(_f32_sub(target, row_max), log_denom)
    loss = _f32_sub(0.0, logp.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    zero = tl.full((), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(out_ptr + row, tl.where(valid, loss, zero))


# b899b223: (T([8192], i64, gen=Index(262144)), T([8192,262144], bf16))
@oracle_impl(hardware="B200", point="b899b223", BLOCK_N=8192, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_N: int, num_warps: int, num_stages: int):
    labels, logits = inputs
    rows = int(logits.shape[0])
    cols = int(logits.shape[1])
    out = torch.empty_strided((rows,), (1,), device=logits.device, dtype=torch.bfloat16)

    _cross_entropy_forward_kernel[(rows,)](
        labels,
        logits,
        out,
        N_COLS=cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
