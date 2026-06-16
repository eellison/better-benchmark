"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Blenderbot bf16 causal-mask fanout scope by materializing all twenty-four fresh `[32,1,128,128]` graph outputs in one Triton multi-output kernel that shares the generated iota/le predicate and bf16 `where` constants, whereas Inductor lowers the identical graph-output siblings as repeated pointwise materializations from the same structured predicate; Inductor cannot do this today because the pointwise scheduler does not group same-domain output siblings with shared shape-derived producers into one store schedule; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for identical structured pointwise roots so one schedule can emit all required fresh output stores while reusing the causal predicate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _causal_mask_24_kernel(
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
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cols = offsets % S
    rows = (offsets // S) % S
    values = tl.where(cols <= rows, 0.0, -float("inf"))

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


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch = expand_shape[0]
    seq = expand_shape[2]
    numel = batch * seq * seq
    stride = (seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(24)
    )

    _causal_mask_24_kernel[(triton.cdiv(numel, BLOCK),)](
        *outputs,
        N=numel,
        S=seq,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return outputs
