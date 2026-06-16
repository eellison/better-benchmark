"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Bart/MBart bf16 causal-mask fanout scope by materializing all twelve fresh `[8,1,1024,1024]` graph outputs in one Triton multi-output kernel that shares the generated iota/le predicate and bf16 `where` constants, whereas Inductor lowers the identical graph-output siblings as repeated pointwise materializations from the same structured predicate; Inductor cannot do this today because the pointwise scheduler does not group same-domain output siblings with shared shape-derived producers into one store schedule; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for identical structured pointwise roots so one schedule can emit all required fresh output stores while reusing the causal predicate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _causal_mask_12_kernel(
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
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    row = pid % S
    offsets = pid * S + cols
    mask = cols < S
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


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    stride = (heads * seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(12)
    )

    _causal_mask_12_kernel[(batch * heads * seq,)](
        *outputs,
        S=seq,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return outputs
