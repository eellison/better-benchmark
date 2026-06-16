"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Blenderbot bool attention-mask bias scope by returning the two fresh bf16 scalar constants and materializing the bf16 `[16,1,128,128]` `where(mask, 0, -inf)` output in one Triton pass, whereas Inductor lowers the same scalar fills and mask materialization as generic pointwise work; Inductor cannot materially do less local work for this captured region because the observable output scope includes the scalar constants plus the full bias tensor with `-inf` values, so the remaining cost is allocation, launch, mask reads, and stores; the fix is BANDWIDTH_BOUND: record this as an at-floor materialization unless a larger graph eliminates the returned mask bias or the scalar outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


OUT_SHAPE = (16, 1, 128, 128)
OUT_STRIDE = (16384, 16384, 128, 1)
N_ELEMENTS = 16 * 128 * 128


@triton.jit
def _attention_mask_bias_kernel(
    mask_ptr,
    neg_inf_ptr,
    zero_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < 262144
    mask_values = tl.load(mask_ptr + offsets, mask=valid, other=0)
    values = tl.where(mask_values != 0, 0.0, -float("inf"))
    tl.store(out_ptr + offsets, values, mask=valid)
    tl.store(neg_inf_ptr, -float("inf"), mask=pid == 0)
    tl.store(zero_ptr, 0.0, mask=pid == 0)


@oracle_impl(hardware="B200", point="61a8ad35")
def oracle_forward(inputs):
    (mask,) = inputs
    neg_inf = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    zero = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    grid = (triton.cdiv(N_ELEMENTS, 1024),)
    _attention_mask_bias_kernel[grid](
        mask,
        neg_inf,
        zero,
        output,
        BLOCK=1024,
        num_warps=4,
    )
    return neg_inf, zero, output
