"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 contiguous sigmoid scope as a storage-linear Triton kernel, including bf16 input promotion to fp32, natural `exp(-x)`, fp32 reciprocal expression, and final bf16 output rounding for every captured point, whereas Inductor lowers the same one-op graph through its generic unary pointwise scheduler; Inductor cannot do this today because its pointwise pattern library does not have a B200-tuned tiny-contiguous sigmoid lowering that strips generic schedule overhead while preserving the ATen sigmoid dtype boundary; the fix is NEW_PATTERN: add a guarded contiguous bf16 sigmoid pointwise template or prove the generic unary pointwise lowering is already at the hardware floor."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _sigmoid_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(input_ptr + offsets).to(tl.float32)
    y = 1.0 / (1.0 + libdevice.exp(-x))
    if USE_MASK:
        tl.store(output_ptr + offsets, y, mask=mask)
    else:
        tl.store(output_ptr + offsets, y)


# c06edc2f: T([96,65], bf16)
@oracle_impl(hardware="B200", point="c06edc2f", BLOCK_SIZE=256, USE_MASK=True, num_warps=4)
# c60f6058: T([256,1,1,1], bf16)
@oracle_impl(hardware="B200", point="c60f6058", BLOCK_SIZE=256, USE_MASK=False, num_warps=4)
# dda9fef3: T([32,1,1,1], bf16)
@oracle_impl(hardware="B200", point="dda9fef3", BLOCK_SIZE=32, USE_MASK=False, num_warps=1)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    _sigmoid_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x,
        out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return out
