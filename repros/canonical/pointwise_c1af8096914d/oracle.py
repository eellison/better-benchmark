"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 bool attention-mask-to-bf16-bias scope by materializing the two returned scalar bf16 constants, the compact `[64,1,128,128]` 0/-inf base buffer, and the exact `[64,16,128,128]` zero-stride expanded view from that base, whereas Inductor lowers the scalar fills, where, and expand chain through generic pointwise/view scheduling; Inductor cannot do this today because it has no guarded attention-mask-bias materialization pattern that preserves sibling scalar outputs, bf16 0/-inf values, and the shared base/expanded-view storage contract; the fix is NEW_PATTERN: add a mask-to-bias lowering that emits the compact base-buffer kernel and returns all scalar and layout-only expanded aliases from the same repro scope."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 64
HEADS = 16
SEQ = 128
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
N_ELEMENTS = BATCH * SEQ * SEQ


@triton.jit
def _mask_to_bias_with_scalars_kernel(
    mask_ptr,
    neg_inf_ptr,
    zero_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    values = tl.where(keep, 0.0, -float("inf"))
    tl.store(out_ptr + offsets, values, mask=valid)
    scalar_ptr_offsets = offsets * 0
    tl.store(neg_inf_ptr + scalar_ptr_offsets, -float("inf"), mask=offsets == 0)
    tl.store(zero_ptr + scalar_ptr_offsets, 0.0, mask=offsets == 0)


@oracle_impl(hardware="B200", point="6c0a9ee3", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    mask, _shape0 = inputs
    neg_inf = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    zero = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(N_ELEMENTS, BLOCK),)
    _mask_to_bias_with_scalars_kernel[grid](
        mask,
        neg_inf,
        zero,
        base,
        TOTAL=N_ELEMENTS,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return neg_inf, zero, base, base.expand(EXPAND_SHAPE)
