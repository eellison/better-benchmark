"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Blenderbot causal-mask scope as one Triton materialization kernel, converting the captured f32 `[16,128]` source mask to bool, applying the fixed `k <= q` causal predicate, writing the fresh contiguous bool `[16,1,128,128]` base, and returning the final expand view, whereas Inductor lowers the decomposed full/iota/add/unsqueeze/le/index/bitwise_and/expand graph through generic pointwise and view scheduling; Inductor cannot do this today because its pattern library has no guarded causal-mask materialization template that replaces the indexing graph while preserving the visible expanded output envelope; the fix is NEW_PATTERN: add a static causal-mask template for source-mask-and-tril predicates with direct contiguous output materialization and metadata expand return."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _causal_mask_kernel(
    mask_ptr,
    out_ptr,
    MASK_S0: tl.constexpr,
    MASK_S1: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    q_offsets = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)

    src = tl.load(
        mask_ptr + batch * MASK_S0 + k_offsets * MASK_S1,
        mask=k_offsets < SEQ,
        other=0.0,
    )
    src_mask = src != 0.0
    causal = k_offsets[None, :] <= q_offsets[:, None]
    out = causal & src_mask[None, :]

    out_offsets = batch * (SEQ * SEQ) + q_offsets[:, None] * SEQ + k_offsets[None, :]
    store_mask = (q_offsets[:, None] < SEQ) & (k_offsets[None, :] < SEQ)
    tl.store(out_ptr + out_offsets, out, mask=store_mask)


# 003d4361: (T([16,128], f32), S([16,-1,128,128]))
@oracle_impl(hardware="B200", point="003d4361", BLOCK_Q=32, BLOCK_K=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int, num_warps: int):
    source_mask, expand_shape = inputs
    batch = int(source_mask.shape[0])
    seq = int(source_mask.shape[1])
    base = torch.empty_strided(
        (batch, 1, seq, seq),
        (seq * seq, seq * seq, seq, 1),
        device=source_mask.device,
        dtype=torch.bool,
    )

    grid = (
        batch,
        triton.cdiv(seq, BLOCK_Q),
        triton.cdiv(seq, BLOCK_K),
    )
    _causal_mask_kernel[grid](
        source_mask,
        base,
        MASK_S0=int(source_mask.stride(0)),
        MASK_S1=int(source_mask.stride(1)),
        SEQ=seq,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return base.expand(tuple(int(dim) for dim in expand_shape))
