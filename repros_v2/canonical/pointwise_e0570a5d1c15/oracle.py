"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Qwen bf16 causal-mask fanout scope by materializing twenty-eight fresh `[1,1,1000,1000]` `where(causal, 0, -inf)` outputs in one Triton kernel that shares the generated iota/le predicate and nonfinite bf16 constants, whereas Inductor lowers the identical graph-output siblings as repeated pointwise materializations from the same structured predicate. Inductor cannot do this today because the pointwise scheduler does not group same-domain output siblings with shared shape-derived producers into one store schedule while preserving distinct output storages; the fix is SCHEDULER_FUSION: add graph-output sibling fusion for repeated causal-mask roots so one schedule can emit all required fresh outputs while reusing the generated predicate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 1000
OUTPUT_COUNT = 28


@triton.jit
def _causal_mask_28_kernel(
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


def _as_shape(shape_param):
    return tuple(1 if int(dim) == -1 else int(dim) for dim in shape_param)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (shape_param,) = inputs
    shape = _as_shape(shape_param)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(
            shape,
            (SEQ * SEQ, SEQ * SEQ, SEQ, 1),
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(OUTPUT_COUNT)
    )

    _causal_mask_28_kernel[(SEQ,)](
        *outputs,
        S=SEQ,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return outputs
