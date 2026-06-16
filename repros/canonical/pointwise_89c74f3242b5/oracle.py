"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete PLBart bool attention-mask-to-bf16-bias scope by materializing the compact `[16,1,1024,1024]` 0/-inf base buffer once and returning both that tensor and the exact `[16,12,1024,1024]` zero-stride expanded view, whereas Inductor lowers the scalar fill, where, and expand chain through generic pointwise/view scheduling; Inductor cannot do this today because it has no guarded attention-mask-bias materialization pattern that treats the head expansion as metadata while preserving the sibling base output, bf16 0/-inf values, and shared storage contract; the fix is NEW_PATTERN: add a mask-to-bias lowering that emits the compact base-buffer kernel and returns all layout-only expanded aliases from that storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 16
HEADS = 12
SEQ = 1024
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
N_ELEMENTS = BATCH * SEQ * SEQ


@triton.jit
def _mask_to_bias_kernel(mask_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < (16 * 1024 * 1024)
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    values = tl.where(keep, 0.0, -float("inf"))
    tl.store(out_ptr + offsets, values, mask=valid)


@oracle_impl(hardware="B200", point="4d6c86f7", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    mask, _shape0 = inputs
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(N_ELEMENTS, BLOCK),)
    _mask_to_bias_kernel[grid](
        mask,
        base,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return base, base.expand(EXPAND_SHAPE)
