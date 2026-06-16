"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Pegasus causal source-mask scope as one Triton materialization kernel, including the f32-to-bool conversion of `arg0`, the generated `key <= query` causal predicate, the indexed source-mask broadcast, and the fresh contiguous `bool[128, 1, 128, 128]` return. Inductor lowers the same iota/unsqueeze/index/bitwise-and/expand graph into generic pointwise work; it cannot materially do less local work for this captured scope because the required output is the full materialized bool mask and the cheap structured predicates are already pointwise-affine. The fix is BANDWIDTH_BOUND: record the direct materialization floor unless a broader graph can consume the structured mask without storing it."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
SEQ = 128
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@triton.jit
def _causal_source_mask_kernel(
    in_ptr,
    out_ptr,
    IN_S0: tl.constexpr,
    IN_S1: tl.constexpr,
    SEQ_: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    q_block = tl.program_id(1)
    k_block = tl.program_id(2)

    q = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    q_mask = q < SEQ_
    k_mask = k < SEQ_

    source = tl.load(in_ptr + batch * IN_S0 + k * IN_S1, mask=k_mask, other=0.0)
    source_mask = source != 0.0
    values = (k[None, :] <= q[:, None]) & source_mask[None, :]

    out_offsets = batch * (SEQ_ * SEQ_) + q[:, None] * SEQ_ + k[None, :]
    tl.store(out_ptr + out_offsets, values, mask=q_mask[:, None] & k_mask[None, :])


@oracle_impl(
    hardware="B200",
    point="60ceae96",
    BLOCK_Q=32,
    BLOCK_K=128,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int, num_warps: int):
    arg0_1, _shape_param_0 = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    _causal_source_mask_kernel[
        (BATCH, triton.cdiv(SEQ, BLOCK_Q), triton.cdiv(SEQ, BLOCK_K))
    ](
        arg0_1,
        out,
        IN_S0=arg0_1.stride(0),
        IN_S1=arg0_1.stride(1),
        SEQ_=SEQ,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
