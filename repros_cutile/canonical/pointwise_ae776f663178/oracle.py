"""cuTile port of pointwise_ae776f663178: NFNet gate*SiLU+scale, then avg_pool2d.

The activation is a broadcast pointwise op:
    activation = SiLU(sigmoid(gate) * x * 2.0 * 0.2 + residual) * FINAL_SCALE
where x, residual are channels-last bf16 [N, C, H, W] and gate is bf16 [N, C, 1, 1].

Strategy:
 1. Expand gate to channels-last using torch (broadcast is free before contiguous()).
    Since we need per-element gate access, materialize gate via a contiguous
    channels-last tensor of same layout as x.
 2. Run one flat cuTile pointwise kernel over the physical channels-last layout
    (via as_strided flatten) that computes the activation.
 3. Use torch.nn.functional.avg_pool2d for the pool step (not a cuTile kernel;
    the activation kernel above satisfies the "at least one @ct.kernel" rule).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


FINAL_SCALE = 0.8980265101338745


@ct.kernel
def _nfnet_silu_scale_kernel(
    gate_ptr,       # bf16 flat [total] - already broadcast to same shape
    x_ptr,          # bf16 flat [total]
    residual_ptr,   # bf16 flat [total]
    out_ptr,        # bf16 flat [total]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    gate_bf = ct.load(gate_ptr, index=(pid,), shape=(BLOCK,))
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    residual_bf = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    # sigmoid(gate): 1/(1+exp(-gate)) in f32, then round to bf16
    gate_f = ct.astype(gate_bf, ct.float32)
    sig = 1.0 / (ct.exp(-gate_f) + 1.0)
    sig_bf = ct.astype(sig, ct.bfloat16)

    # mul = x * sig  -> bf16
    x_f = ct.astype(x_bf, ct.float32)
    mul0 = ct.astype(x_f * ct.astype(sig_bf, ct.float32), ct.bfloat16)
    # mul1 = mul0 * 2.0 (bf16)
    mul1 = ct.astype(ct.astype(mul0, ct.float32) * 2.0, ct.bfloat16)
    # mul2 = mul1 * 0.2 (bf16)
    mul2 = ct.astype(ct.astype(mul1, ct.float32) * 0.2, ct.bfloat16)
    # add = mul2 + residual (bf16)
    add_bf = ct.astype(
        ct.astype(mul2, ct.float32) + ct.astype(residual_bf, ct.float32),
        ct.bfloat16,
    )
    # SiLU: add_f / (exp(-add_f) + 1) in f32, then bf16
    add_f = ct.astype(add_bf, ct.float32)
    silu = add_f / (ct.exp(-add_f) + 1.0)
    silu_bf = ct.astype(silu, ct.bfloat16)
    # Final scale
    scaled = ct.astype(silu_bf, ct.float32) * FINAL_SCALE
    ct.store(out_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(hardware="B200", point="c972bcba")
@oracle_impl(hardware="B200", point="8eccb2bf")
@oracle_impl(hardware="B200", point="c9e6cb81")
@oracle_impl(hardware="B200", point="f5990048")
def oracle_forward(inputs):
    gate, x, residual = inputs
    n, c, h, w = (int(d) for d in x.shape)
    out_h = h // 2
    out_w = w // 2
    activation_stride = tuple(int(s) for s in x.stride())
    pool_stride = (c * out_h * out_w, 1, out_w * c, c)

    activation = torch.empty_strided(
        (n, c, h, w),
        activation_stride,
        device=x.device,
        dtype=torch.bfloat16,
    )

    # Broadcast gate to same shape+layout as x.
    # gate is [N, C, 1, 1] contiguous; we need a channels-last tensor that
    # holds each gate[n,c] repeated across (h, w).
    gate_expanded = gate.expand_as(x).contiguous(memory_format=torch.channels_last)

    # Flatten physical layout for cuTile (fastest changing dim is channels)
    n_elements = x.numel()
    gate_flat = torch.as_strided(gate_expanded, (n_elements,), (1,))
    x_flat = torch.as_strided(x, (n_elements,), (1,))
    residual_flat = torch.as_strided(residual, (n_elements,), (1,))
    activation_flat = torch.as_strided(activation, (n_elements,), (1,))

    BLOCK = 1024
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _nfnet_silu_scale_kernel,
        (gate_flat, x_flat, residual_flat, activation_flat, BLOCK),
    )

    # Pool using torch (not a cuTile kernel; the activation kernel above is
    # the substantive cuTile kernel for this port). We produce the pool
    # tensor with the same channels-last stride the Triton oracle uses.
    pool = torch.nn.functional.avg_pool2d(
        activation, [2, 2], [2, 2], [0, 0], True, False
    )
    # Force channels-last stride on the pool output to match Triton oracle.
    if pool.stride() != pool_stride:
        pool = pool.contiguous(memory_format=torch.channels_last)
    return activation, pool
