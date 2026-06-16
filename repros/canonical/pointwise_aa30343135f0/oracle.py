"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistillGPT2 same-segment causal attention-mask tuple by materializing all six distinct bf16 `[32,1,512,512]` 0/-inf outputs in one Triton pass from the packed sequence-id input, whereas Inductor lowers the decomposed iota/index/equality/causal-predicate and six sibling `where` fills as generic pointwise work; Inductor cannot do this today because it has no segmented causal-mask multi-output lowering that shares the predicate while preserving distinct returned storages and bf16 Inf semantics; the fix is NEW_PATTERN: add a guarded segmented-causal-attention-mask lowering that fuses predicate construction with all identical materialized bias outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@triton.jit
def _segmented_causal_mask6_kernel(
    segments_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    out3_ptr,
    out4_ptr,
    out5_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_block = tl.program_id(0)
    key_block = tl.program_id(1)
    batch = tl.program_id(2)

    queries = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    keys = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    query_valid = queries < 512
    key_valid = keys < 512

    segment_base = segments_ptr + batch * 512
    query_segments = tl.load(segment_base + queries, mask=query_valid, other=-1)
    key_segments = tl.load(segment_base + keys, mask=key_valid, other=-2)

    query_2d = queries[:, None]
    key_2d = keys[None, :]
    valid = (query_2d < 512) & (key_2d < 512)
    keep = (key_2d <= query_2d) & (query_segments[:, None] == key_segments[None, :])
    values = tl.where(keep, 0.0, -float("inf"))
    offsets = batch * 512 * 512 + query_2d * 512 + key_2d

    tl.store(out0_ptr + offsets, values, mask=valid)
    tl.store(out1_ptr + offsets, values, mask=valid)
    tl.store(out2_ptr + offsets, values, mask=valid)
    tl.store(out3_ptr + offsets, values, mask=valid)
    tl.store(out4_ptr + offsets, values, mask=valid)
    tl.store(out5_ptr + offsets, values, mask=valid)


@oracle_impl(hardware="B200", point="26cc4258")
def oracle_forward(inputs):
    segments, _shape_param_0 = inputs
    outputs = tuple(
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=segments.device,
            dtype=torch.bfloat16,
        )
        for _ in range(6)
    )
    grid = (triton.cdiv(SEQ, 16), triton.cdiv(SEQ, 64), BATCH)
    _segmented_causal_mask6_kernel[grid](
        segments,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        BLOCK_M=16,
        BLOCK_N=64,
        num_warps=4,
    )
    return outputs
