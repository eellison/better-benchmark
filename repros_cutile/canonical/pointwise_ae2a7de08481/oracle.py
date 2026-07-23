"""cuTile port of pointwise_ae2a7de08481: NFNet gated GELU + returned add + scaling.

Single flat cuTile kernel that computes sigmoid gate, mul chain, residual add,
exact-erf GELU (via Abramowitz-Stegun 7.1.26 polynomial in-kernel), and the
two bf16 scale multiplies — matching the Triton kernel's structure.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SQRT_HALF = 0.7071067811865476


def _erf_approx(x):
    """Abramowitz-Stegun 7.1.26 polynomial approximation for erf."""
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108

    x_abs = ct.abs(x)
    x2 = x_abs * x_abs
    x3 = x2 * x_abs
    x4 = x3 * x_abs
    denom = 1.0 + a1 * x_abs + a2 * x2 + a3 * x3 + a4 * x4
    denom2 = denom * denom
    denom4 = denom2 * denom2
    erf_abs = 1.0 - 1.0 / denom4
    sign = ct.where(x >= 0.0, 1.0, -1.0)
    return sign * erf_abs


@ct.kernel
def _nfnet_gate_gelu_flat_kernel(
    gate_ptr,        # bf16 [N, C]
    payload_ptr,     # bf16 [N*C*HW] (flat channels-last)
    scalar_ptr,      # bf16 [] scalar
    residual_ptr,    # bf16 [N*C*HW]
    add_out_ptr,     # bf16 [N*C*HW]
    final_out_ptr,   # bf16 [N*C*HW]
    N_ELEMENTS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < N_ELEMENTS
    c_offsets = offsets % CHANNELS
    n_offsets = offsets // (CHANNELS * HW)
    gate_offsets = n_offsets * CHANNELS + c_offsets

    # Gather gate per element (gate_ptr is flat [N*C]).
    gate = ct.astype(ct.gather(gate_ptr, (gate_offsets,)), ct.float32)
    gate_sig = ct.astype(1.0 / (1.0 + ct.exp(-gate)), ct.bfloat16)

    payload = ct.astype(
        ct.load(payload_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    residual = ct.astype(
        ct.load(residual_ptr, index=(pid,), shape=(BLOCK,),
                padding_mode=ct.PaddingMode.ZERO), ct.float32)
    scalar = ct.astype(ct.load(scalar_ptr, index=(0,), shape=(1,)), ct.bfloat16)
    scalar_bcast = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    # broadcast scalar: since it's a scalar, we need to explicitly broadcast
    # via scattered gather. Simpler: load and reshape to (1,) then it broadcasts.
    scalar_val = ct.astype(scalar, ct.float32)

    x = ct.astype(payload * ct.astype(gate_sig, ct.float32), ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * 2.0, ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * ct.astype(scalar_val, ct.float32), ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * 0.2, ct.bfloat16)
    add = ct.astype(ct.astype(x, ct.float32) + residual, ct.bfloat16)
    # Store add
    add_masked = ct.where(mask, add,
                          ct.full((BLOCK,), 0.0, dtype=ct.bfloat16))
    # Use scatter with mask
    ct.scatter(add_out_ptr, (offsets,), add_masked, mask=mask)

    add_f32 = ct.astype(add, ct.float32)
    erf_arg = add_f32 * SQRT_HALF
    erf_val = _erf_approx(erf_arg)
    gelu = ct.astype((add_f32 * 0.5) * (erf_val + 1.0), ct.bfloat16)
    scaled = ct.astype(ct.astype(gelu, ct.float32) * 1.7015043497085571, ct.bfloat16)
    final = ct.astype(ct.astype(scaled, ct.float32) * 0.9622504486493761, ct.bfloat16)
    final_masked = ct.where(mask, final,
                            ct.full((BLOCK,), 0.0, dtype=ct.bfloat16))
    ct.scatter(final_out_ptr, (offsets,), final_masked, mask=mask)


@oracle_impl(hardware="B200", point="fac11cdd", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="27a04b02", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="a4fc89e1", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="fd33c7c3", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="6498d204", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
@oracle_impl(hardware="B200", point="b99b11a3", USE_FLAT=True, BLOCK=512, BLOCK_C=32, BLOCK_HW=32)
def oracle_forward(inputs, *, USE_FLAT, BLOCK, BLOCK_C, BLOCK_HW):
    del USE_FLAT, BLOCK_C, BLOCK_HW
    gate, payload, scalar, residual = inputs
    device = payload.device
    payload_shape = tuple(payload.shape)
    payload_stride = tuple(payload.stride())
    batch, channels, height, width = payload.shape
    hw = height * width
    n_elements = payload.numel()

    add_out = torch.empty_strided(
        payload_shape, payload_stride, device=device, dtype=torch.bfloat16
    )
    final_out = torch.empty_strided(
        payload_shape, payload_stride, device=device, dtype=torch.bfloat16
    )

    # Use flat views (channels-last physical storage)
    payload_flat = torch.as_strided(payload, (n_elements,), (1,))
    residual_flat = torch.as_strided(residual, (n_elements,), (1,))
    add_flat = torch.as_strided(add_out, (n_elements,), (1,))
    final_flat = torch.as_strided(final_out, (n_elements,), (1,))
    # gate is [N, C, 1, 1] typically; we need a flat [N*C] view
    gate_2d = gate.reshape(batch * channels)

    # scalar may be rank-0; expose as (1,) via view for cuTile.
    scalar_1d = scalar.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _nfnet_gate_gelu_flat_kernel,
        (gate_2d, payload_flat, scalar_1d, residual_flat, add_flat, final_flat,
         n_elements, channels, hw, BLOCK),
    )
    return add_out, final_out
