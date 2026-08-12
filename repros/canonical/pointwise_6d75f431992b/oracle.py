"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistillGPT2 segment-aware causal attention-mask bias by materializing the bf16 `[32,1,512,512]` 0/-inf backing buffer and returning both that tensor and the exact `[32,12,512,512]` zero-stride head expand view, whereas Inductor lowers the paired advanced-index segment lookups, causal comparison, equality, where, and expand chain as generic pointwise work; Inductor cannot do this today because it has no shape-specialized segment-causal-mask pattern that recognizes the shared position vector, paired cumsum gathers, bf16 0/-inf bias, and metadata-only head expansion as one attention-mask idiom; the fix is NEW_PATTERN: add a guarded lowering for segment-aware causal mask construction that emits the compact base-buffer kernel and preserves the expanded output layout contract."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 12
SEQ = 512
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)


@triton.jit
def _segment_causal_mask_kernel(
    positions_ptr,
    cumsum_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < (32 * 512 * 512)
    k = offsets % 512
    q = (offsets // 512) % 512
    batch = offsets // (512 * 512)

    q_pos = tl.load(positions_ptr + q, mask=mask, other=0)
    k_pos = tl.load(positions_ptr + k, mask=mask, other=0)
    q_segment = tl.load(cumsum_ptr + batch * 512 + q_pos, mask=mask, other=-1)
    k_segment = tl.load(cumsum_ptr + batch * 512 + k_pos, mask=mask, other=-2)

    keep = (k_pos <= q_pos) & (k_segment == q_segment)
    values = tl.where(keep, 0.0, -float("inf"))
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="9210619b", BLOCK=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    positions, cumsum, _shape0, _shape1 = inputs
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=positions.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(BATCH * SEQ * SEQ, BLOCK),)
    _segment_causal_mask_kernel[grid](
        positions,
        cumsum,
        base,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return base, base.expand(EXPAND_SHAPE)
