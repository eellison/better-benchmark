"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Mistral bf16 causal-mask fanout scope by materializing all thirty-two fresh `[1,1,1000,1000]` graph outputs in one Triton multi-output kernel that shares the generated iota/le predicate and bf16 `where` constants, whereas Inductor lowers the identical graph-output siblings as repeated pointwise materializations from the same structured predicate. Inductor cannot do this today because the pointwise scheduler does not group same-domain output siblings with shared shape-derived producers into one store schedule; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for identical structured pointwise roots so one schedule can emit all required fresh output stores while reusing the causal predicate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUTPUT_COUNT = 32


@triton.jit
def _causal_mask_32_kernel(
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
    out29,
    out30,
    out31,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < S
    offsets = row * S + cols
    values = tl.where(cols <= row, 0.0, -float("inf"))

    tl.store(out0 + offsets, values, mask=mask)
    tl.store(out1 + offsets, values, mask=mask)
    tl.store(out2 + offsets, values, mask=mask)
    tl.store(out3 + offsets, values, mask=mask)
    tl.store(out4 + offsets, values, mask=mask)
    tl.store(out5 + offsets, values, mask=mask)
    tl.store(out6 + offsets, values, mask=mask)
    tl.store(out7 + offsets, values, mask=mask)
    tl.store(out8 + offsets, values, mask=mask)
    tl.store(out9 + offsets, values, mask=mask)
    tl.store(out10 + offsets, values, mask=mask)
    tl.store(out11 + offsets, values, mask=mask)
    tl.store(out12 + offsets, values, mask=mask)
    tl.store(out13 + offsets, values, mask=mask)
    tl.store(out14 + offsets, values, mask=mask)
    tl.store(out15 + offsets, values, mask=mask)
    tl.store(out16 + offsets, values, mask=mask)
    tl.store(out17 + offsets, values, mask=mask)
    tl.store(out18 + offsets, values, mask=mask)
    tl.store(out19 + offsets, values, mask=mask)
    tl.store(out20 + offsets, values, mask=mask)
    tl.store(out21 + offsets, values, mask=mask)
    tl.store(out22 + offsets, values, mask=mask)
    tl.store(out23 + offsets, values, mask=mask)
    tl.store(out24 + offsets, values, mask=mask)
    tl.store(out25 + offsets, values, mask=mask)
    tl.store(out26 + offsets, values, mask=mask)
    tl.store(out27 + offsets, values, mask=mask)
    tl.store(out28 + offsets, values, mask=mask)
    tl.store(out29 + offsets, values, mask=mask)
    tl.store(out30 + offsets, values, mask=mask)
    tl.store(out31 + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    stride = (heads * seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(OUTPUT_COUNT)
    )

    _causal_mask_32_kernel[(batch * heads * seq,)](
        *outputs,
        S=seq,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return outputs
