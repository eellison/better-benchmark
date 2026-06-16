"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-2 segment-aware causal attention-mask scope by materializing the `i64[8]` iota side output, writing the compact bf16 `[8,1,1024,1024]` zero-or-`-inf` backing buffer, and returning the exact `[8,12,1024,1024]` head-expanded view, whereas Inductor lowers the paired advanced-index segment lookups, causal comparison, equality, where, side output, and expand chain as generic pointwise work; Inductor cannot do this today because it has no guarded segment-causal-mask lowering that recognizes the shared position vector, paired cumsum gathers, and metadata-only head expansion while preserving the visible side outputs; the fix is NEW_PATTERN: add a shape-specialized segment-causal attention-mask lowering that emits the compact base-buffer kernel and exact expand-layout returns."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _segment_mask_kernel(
    position_ptr,
    segment_ptr,
    iota_ptr,
    out_ptr,
    SEQ: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch = tl.program_id(0)
    q_block = tl.program_id(1)
    k_block = tl.program_id(2)

    if (q_block == 0) & (k_block == 0):
        tl.store(iota_ptr + batch, batch)

    q_offsets = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    q_mask = q_offsets < SEQ
    k_mask = k_offsets < SEQ

    q_pos = tl.load(position_ptr + q_offsets, mask=q_mask, other=0)
    k_pos = tl.load(position_ptr + k_offsets, mask=k_mask, other=0)
    segment_base = segment_ptr + batch * SEQ
    q_segment = tl.load(segment_base + q_pos, mask=q_mask, other=-1)
    k_segment = tl.load(segment_base + k_pos, mask=k_mask, other=-2)

    keep = (k_pos[None, :] <= q_pos[:, None]) & (k_segment[None, :] == q_segment[:, None])
    values = tl.where(keep, 0.0, -float("inf"))
    out_offsets = batch * (SEQ * SEQ) + q_offsets[:, None] * SEQ + k_offsets[None, :]
    tl.store(out_ptr + out_offsets, values, mask=q_mask[:, None] & k_mask[None, :])


# 1536bfa0: (T([1,1024], i64, gen=Index(1024)), T([8,1024], i64, gen=Index(8)), S([8,-1,1024,1024]), S([8,12,1024,1024]))
@oracle_impl(hardware="B200", point="1536bfa0", BLOCK_Q=16, BLOCK_K=128, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_Q: int, BLOCK_K: int, num_warps: int, num_stages: int):
    positions, segments, _shape_param_0, shape_param_1 = inputs
    del _shape_param_0

    batch = int(segments.shape[0])
    seq = int(positions.shape[1])
    iota = torch.empty_strided((batch,), (1,), device=positions.device, dtype=torch.int64)
    base = torch.empty_strided(
        (batch, 1, seq, seq),
        (seq * seq, seq * seq, seq, 1),
        device=positions.device,
        dtype=torch.bfloat16,
    )
    _segment_mask_kernel[
        (batch, triton.cdiv(seq, BLOCK_Q), triton.cdiv(seq, BLOCK_K))
    ](
        positions,
        segments,
        iota,
        base,
        SEQ=seq,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return iota, base, base.expand(tuple(int(dim) for dim in shape_param_1))
