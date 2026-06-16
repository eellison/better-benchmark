"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete static Gemma bf16 causal-mask scope by materializing the twenty-nine required `[1,1,1000,1000]` 0/-inf backing buffers directly and returning their exact `[1,8,1000,1000]` zero-stride expanded views, whereas Inductor lowers the iota/unsqueeze/window-predicate construction, repeated scalar fills, repeated where materializations, and expand returns through generic pointwise/view scheduling; Inductor cannot do this today because its algebraic simplifier does not prove the captured 1024-wide sliding-window lower bound is tautologically true for `seq=1000` and does not fold the repeated static causal mask into direct backing-buffer materialization while preserving separate output storage and expanded-view strides; the fix is ALGEBRAIC_ELIMINATION: recognize this static mask fragment and replace it with one direct materialization feeding the full repeated expanded-output scope."""

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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=4),
    ],
    key=["TOTAL"],
)
@triton.jit
def _causal_mask_29_kernel(
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
    out13,
    out14,
    out15,
    out16,
    out17,
    out18,
    out19,
    out20,
    out21,
    out22,
    out23,
    out24,
    out25,
    out26,
    out27,
    out28,
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
    tl.store(out13 + offsets, value, mask=mask)
    tl.store(out14 + offsets, value, mask=mask)
    tl.store(out15 + offsets, value, mask=mask)
    tl.store(out16 + offsets, value, mask=mask)
    tl.store(out17 + offsets, value, mask=mask)
    tl.store(out18 + offsets, value, mask=mask)
    tl.store(out19 + offsets, value, mask=mask)
    tl.store(out20 + offsets, value, mask=mask)
    tl.store(out21 + offsets, value, mask=mask)
    tl.store(out22 + offsets, value, mask=mask)
    tl.store(out23 + offsets, value, mask=mask)
    tl.store(out24 + offsets, value, mask=mask)
    tl.store(out25 + offsets, value, mask=mask)
    tl.store(out26 + offsets, value, mask=mask)
    tl.store(out27 + offsets, value, mask=mask)
    tl.store(out28 + offsets, value, mask=mask)


# d7517139: Gemma static causal mask, 29 expanded bf16 outputs.
@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    device = torch.device("cuda")
    bases = [
        torch.empty_strided(
            BASE_SHAPE,
            BASE_STRIDE,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(29)
    ]

    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK"]),)
    _causal_mask_29_kernel[grid](
        *bases,
        S=SEQ,
        TOTAL=N_ELEMENTS,
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
