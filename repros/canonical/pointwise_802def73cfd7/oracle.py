"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete OPT inference causal-mask bias scope in one Triton materialization kernel, including the generated 2048x2048 `key <= query` predicate, the indexed all-true `[4,2048]` source mask, twelve independent bf16 `where(mask, 0.0, -inf)` outputs, and the exact contiguous `[4,1,2048,2048]` return strides with no aliasing between outputs, whereas Inductor lowers the decomposed iota/unsqueeze/le/index/bitwise_and/expand graph and then repeats the identical scalar-fill `where` twelve times through generic pointwise scheduling; Inductor cannot do this today because its simplifier does not fold the all-true indexed source mask and common-subexpression the repeated attention-bias materializations into one full-scope multi-output store; the fix is ALGEBRAIC_ELIMINATION: recognize generated all-true mask sources and emit one multi-output causal-bias materialization that preserves exact bf16 `0.0`/`-inf` constants and output storage independence."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _twelve_causal_bias_kernel(
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
    N_ELEMENTS: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS
    q = (offsets // SEQ) % SEQ
    k = offsets % SEQ
    values = tl.where(k <= q, 0.0, -float("inf"))
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


def _materialized_shape(expand_shape):
    return tuple(1 if int(dim) == -1 else int(dim) for dim in expand_shape)


@oracle_impl(
    hardware="B200",
    point="d7517139",
    BLOCK=1024,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    _source_shape, expand_shape_arg = inputs
    shape = _materialized_shape(expand_shape_arg)
    stride = (shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)
    outs = tuple(
        torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
        for _ in range(12)
    )
    n_elements = shape[0] * shape[1] * shape[2] * shape[3]
    _twelve_causal_bias_kernel[(triton.cdiv(n_elements, BLOCK),)](
        *outs,
        N_ELEMENTS=n_elements,
        SEQ=shape[3],
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return outs
