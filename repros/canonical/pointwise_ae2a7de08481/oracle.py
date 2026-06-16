"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 gated exact-GELU scope in one channels-last Triton tile, including reuse of each `[N,C,1,1]` sigmoid gate across a spatial block, every bf16 rounding boundary before the fp32 GELU, the returned bf16 residual-add side output, and the final two bf16 scalar multiplies, whereas Inductor lowers the decomposed sigmoid/mul/add/cast/erf/cast/mul chain through generic pointwise scheduling over every output element; Inductor cannot do this today because its pointwise scheduler does not tile expensive broadcast-invariant producers by channel while also sinking sibling output stores with explicit dtype boundaries; the fix is SCHEDULER_FUSION: add a guarded channels-last NFNet gated-GELU pointwise template that hoists per-channel gates across spatial tiles and emits all returned outputs from the same schedule."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _nfnet_gate_gelu_kernel(
    gate_ptr,
    payload_ptr,
    scalar_ptr,
    residual_ptr,
    add_out_ptr,
    final_out_ptr,
    CHANNELS: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    c_offsets = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (hw_offsets[:, None] < HW) & (c_offsets[None, :] < CHANNELS)

    offsets = n * CHANNELS * HW + hw_offsets[:, None] * CHANNELS + c_offsets[None, :]
    gate_offsets = n * CHANNELS + c_offsets

    gate = tl.load(gate_ptr + gate_offsets, mask=c_offsets < CHANNELS, other=0.0).to(tl.float32)
    gate = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16)
    payload = tl.load(payload_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.bfloat16)

    x = (payload * gate[None, :].to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 2.0).to(tl.bfloat16)
    x = (x.to(tl.float32) * scalar.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 0.2).to(tl.bfloat16)
    add = (x.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add, mask=mask)

    add_f32 = add.to(tl.float32)
    half = add_f32 * 0.5
    erf_arg = add_f32 * 0.7071067811865476
    gelu = half * (libdevice.erf(erf_arg) + 1.0)
    gelu = gelu.to(tl.bfloat16)
    scaled = (gelu * 1.7015043497085571).to(tl.bfloat16)
    final = (scaled * 0.9622504486493761).to(tl.bfloat16)
    tl.store(final_out_ptr + offsets, final, mask=mask)


@triton.jit
def _nfnet_gate_gelu_flat_kernel(
    gate_ptr,
    payload_ptr,
    scalar_ptr,
    residual_ptr,
    add_out_ptr,
    final_out_ptr,
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
    gate = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16)
    payload = tl.load(payload_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.bfloat16)

    x = (payload * gate.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 2.0).to(tl.bfloat16)
    x = (x.to(tl.float32) * scalar.to(tl.float32)).to(tl.bfloat16)
    x = (x.to(tl.float32) * 0.2).to(tl.bfloat16)
    add = (x.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add, mask=mask)

    add_f32 = add.to(tl.float32)
    gelu = (add_f32 * 0.5) * (libdevice.erf(add_f32 * 0.7071067811865476) + 1.0)
    gelu = gelu.to(tl.bfloat16)
    scaled = (gelu * 1.7015043497085571).to(tl.bfloat16)
    final = (scaled * 0.9622504486493761).to(tl.bfloat16)
    tl.store(final_out_ptr + offsets, final, mask=mask)


@oracle_impl(hardware="B200", point="fac11cdd", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
@oracle_impl(hardware="B200", point="27a04b02", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
@oracle_impl(hardware="B200", point="a4fc89e1", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
@oracle_impl(hardware="B200", point="fd33c7c3", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
@oracle_impl(hardware="B200", point="6498d204", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
@oracle_impl(hardware="B200", point="b99b11a3", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32, num_warps=4)
def oracle_forward(
    inputs,
    *,
    USE_FLAT: bool,
    BLOCK: int,
    BLOCK_C: int,
    BLOCK_HW: int,
    num_warps: int,
):
    gate, payload, scalar, residual = inputs
    batch, channels, height, width = payload.shape
    hw = height * width
    add_out = torch.empty_strided(
        tuple(payload.shape),
        tuple(payload.stride()),
        device=payload.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        tuple(payload.shape),
        tuple(payload.stride()),
        device=payload.device,
        dtype=torch.bfloat16,
    )
    if USE_FLAT:
        n_elements = payload.numel()
        _nfnet_gate_gelu_flat_kernel[(triton.cdiv(n_elements, BLOCK),)](
            gate,
            payload,
            scalar,
            residual,
            add_out,
            final_out,
            N_ELEMENTS=n_elements,
            CHANNELS=channels,
            HW=hw,
            BLOCK=BLOCK,
            num_warps=num_warps,
            num_stages=4,
        )
    else:
        grid = (
            batch,
            triton.cdiv(hw, BLOCK_HW),
            triton.cdiv(channels, BLOCK_C),
        )
        _nfnet_gate_gelu_kernel[grid](
            gate,
            payload,
            scalar,
            residual,
            add_out,
            final_out,
            CHANNELS=channels,
            HW=hw,
            BLOCK_C=BLOCK_C,
            BLOCK_HW=BLOCK_HW,
            num_warps=num_warps,
            num_stages=4,
        )
    return add_out, final_out
