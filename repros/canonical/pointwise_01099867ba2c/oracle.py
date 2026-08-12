"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 NFNet broadcast-gate residual scope in one flat channels-last Triton pointwise schedule while preserving the sigmoid bf16 rounding, the four sequential bf16 multiply roundings, the scalar dtype read, and the final bf16 residual add, whereas Inductor lowers the decomposed sigmoid/mul/mul/mul/mul/add graph with its generic pointwise schedule for these broadcasted channels-last tensors; Inductor cannot do this today because its scheduler does not select a layout-specialized NFNet broadcast-gate residual template that keeps the explicit bf16 rounding boundaries in a compact single-output kernel; the fix is SCHEDULER_FUSION: add a guarded channels-last pointwise schedule for broadcast gate residual blocks with explicit dtype-boundary preservation."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _broadcast_gate_residual_flat_kernel(
    gate_ptr,
    payload_ptr,
    scalar_ptr,
    residual_ptr,
    out_ptr,
    N_ELEMENTS: tl.constexpr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N_ELEMENTS
    c_offsets = offsets % CHANNELS
    n_offsets = offsets // (CHANNELS * HW)
    gate_offsets = n_offsets * CHANNELS + c_offsets

    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16)

    payload = tl.load(payload_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x = (payload * sigmoid.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 2.0).to(tl.bfloat16)
    x = (x.to(tl.float32) * scalar).to(tl.bfloat16)
    x = (x.to(tl.float32) * 0.2).to(tl.bfloat16)
    out = (x.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=mask)


# fac11cdd: (T([128,1536,1,1], bf16), T([128,1536,8,8], bf16, channels_last), T([], bf16), ...)
@oracle_impl(hardware="B200", point="fac11cdd", BLOCK=1024, num_warps=4, num_stages=3)
# fd33c7c3: (T([128,1536,1,1], bf16), T([128,1536,6,6], bf16, channels_last), T([], f32), ...)
@oracle_impl(hardware="B200", point="fd33c7c3", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    gate, payload, scalar, residual = inputs
    batch, channels, height, width = payload.shape
    hw = int(height * width)
    out = torch.empty_strided(
        tuple(payload.shape),
        tuple(payload.stride()),
        device=payload.device,
        dtype=torch.bfloat16,
    )

    n_elements = payload.numel()
    _broadcast_gate_residual_flat_kernel[(triton.cdiv(n_elements, BLOCK),)](
        gate,
        payload,
        scalar,
        residual,
        out,
        N_ELEMENTS=n_elements,
        CHANNELS=int(channels),
        HW=hw,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
