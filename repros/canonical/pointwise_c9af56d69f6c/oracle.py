"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete static Gemma bf16 causal-mask scope by materializing the thirteen required `[1,1,1000,1000]` 0/-inf backing buffers directly and returning their exact `[1,8,1000,1000]` zero-stride expanded views, whereas Inductor lowers the iota/unsqueeze/le, repeated scalar fills, repeated where materializations, and expand returns through generic pointwise/view scheduling; Inductor cannot do this today because its algebraic simplifier does not fold repeated static causal-mask construction into direct backing-buffer materialization while preserving the separate output storage and expanded-view strides; the fix is ALGEBRAIC_ELIMINATION: recognize this static causal-mask fragment and replace it with one direct materialization feeding the full repeated expanded-output scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1000
HEADS = 8
N_ELEMENTS = SEQ * SEQ
BASE_SHAPE = (1, 1, SEQ, SEQ)
BASE_STRIDE = (N_ELEMENTS, N_ELEMENTS, SEQ, 1)
OUT_SHAPE = (1, HEADS, SEQ, SEQ)


@triton.jit
def _causal_mask_13_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    out12,
    S: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    col = offsets % S
    row = offsets // S
    value = tl.where(col <= row, 0.0, -float("inf"))
    tl.store(out0 + offsets, value, mask=mask)
    tl.store(out1 + offsets, value, mask=mask)
    tl.store(out2 + offsets, value, mask=mask)
    tl.store(out3 + offsets, value, mask=mask)
    tl.store(out4 + offsets, value, mask=mask)
    tl.store(out5 + offsets, value, mask=mask)
    tl.store(out6 + offsets, value, mask=mask)
    tl.store(out7 + offsets, value, mask=mask)
    tl.store(out8 + offsets, value, mask=mask)
    tl.store(out9 + offsets, value, mask=mask)
    tl.store(out10 + offsets, value, mask=mask)
    tl.store(out11 + offsets, value, mask=mask)
    tl.store(out12 + offsets, value, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    del inputs
    device = torch.device("cuda")
    bases = [
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(13)
    ]
    grid = (triton.cdiv(N_ELEMENTS, BLOCK),)
    _causal_mask_13_kernel[grid](
        *bases,
        S=SEQ,
        TOTAL=N_ELEMENTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
