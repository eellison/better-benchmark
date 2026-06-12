"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 MobileNetV3 hard-sigmoid broadcast multiply as one storage-linear Triton pointwise kernel for the captured contiguous NCHW layouts, including explicit bf16-to-fp32 gate promotion, add/clamp/div gate math, the required bf16 gate rounding boundary, and final bf16 multiply/store, whereas Inductor lowers the same cast/add/clamp/div/cast/mul scope through generic pointwise scheduling over a singleton-spatial broadcast; Inductor cannot do this today because pointwise codegen has no B200-tuned hard-sigmoid singleton-gate broadcast template that specializes NCHW gate indexing while preserving the intermediate bf16 rounding; the fix is NEW_PATTERN: add a guarded MobileNetV3 hard-sigmoid broadcast-mul pointwise template or equivalent autotuned schedule for singleton-spatial gates."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _nchw_hardsigmoid_mul_kernel(
    gate_ptr,
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    hw: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if USE_MASK:
        mask = offsets < n_elements
        gate = tl.load(gate_ptr + offsets // hw, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    else:
        gate = tl.load(gate_ptr + offsets // hw).to(tl.float32)
        x = tl.load(x_ptr + offsets)

    shifted = gate + 3.0
    clamped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    scale = (clamped / 6.0).to(tl.bfloat16)
    out = x * scale

    if USE_MASK:
        tl.store(out_ptr + offsets, out, mask=mask)
    else:
        tl.store(out_ptr + offsets, out)


@oracle_impl(hardware="B200", point="d7edb67c", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="f9969da7", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="f76fad38", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="0c85aae0", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="be7a1f98", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
@oracle_impl(hardware="B200", point="38d11fef", BLOCK_SIZE=1024, USE_MASK=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, USE_MASK, num_warps):
    gate, x = inputs
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    hw = int(x.shape[2]) * int(x.shape[3])
    _nchw_hardsigmoid_mul_kernel[(triton.cdiv(x.numel(), BLOCK_SIZE),)](
        gate,
        x,
        output,
        n_elements=x.numel(),
        hw=hw,
        BLOCK_SIZE=BLOCK_SIZE,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
    )
    return output
