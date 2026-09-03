"""cuTile port of pointwise_01099867ba2c (SCHEDULER_FUSION): NFNet
broadcast-gate residual — sigmoid(gate) * payload * 2 * scalar * 0.2 + residual,
with explicit bf16 rounding after each multiply.

Payload/residual/out are channels-last (NHWC memory: stride (H*W*C, 1, W*C, C)).
Flatten to 1D; recover c = offset % CHANNELS, n = offset // (CHANNELS*HW).
Gate is (N, C, 1, 1) contiguous so gate_offset = n * C + c.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _broadcast_gate_residual_flat_kernel(
    gate_ptr,     # bf16 [N * C]
    payload_ptr,  # bf16 [TOTAL] (channels-last flat)
    scalar_ptr,   # bf16 or f32 [1]
    residual_ptr, # bf16 [TOTAL]
    out_ptr,      # bf16 [TOTAL]
    CHANNELS: ct.Constant[int],
    CHW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c = offsets - (offsets // CHANNELS) * CHANNELS
    n = offsets // CHW
    gate_off = n * CHANNELS + c

    gate = ct.astype(ct.gather(gate_ptr, (gate_off,)), ct.float32)
    sigmoid_f = 1.0 / (1.0 + ct.exp(-gate))
    sigmoid_bf16 = ct.astype(sigmoid_f, ct.bfloat16)

    payload = ct.astype(
        ct.load(payload_ptr, index=(pid,), shape=(BLOCK,)),
        ct.float32,
    )
    scalar = ct.astype(
        ct.load(scalar_ptr, index=(0,), shape=(1,)),
        ct.float32,
    )
    residual = ct.astype(
        ct.load(residual_ptr, index=(pid,), shape=(BLOCK,)),
        ct.float32,
    )

    x = ct.astype(payload * ct.astype(sigmoid_bf16, ct.float32), ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * 2.0, ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * scalar, ct.bfloat16)
    x = ct.astype(ct.astype(x, ct.float32) * 0.2, ct.bfloat16)
    out = ct.astype(ct.astype(x, ct.float32) + residual, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="fac11cdd", BLOCK=1024)
@oracle_impl(hardware="B200", point="fd33c7c3", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    gate, payload, scalar, residual = inputs
    batch, channels, height, width = payload.shape
    hw = int(height * width)
    chw = channels * hw
    n_elements = payload.numel()
    assert n_elements % BLOCK == 0, (
        f"total {n_elements} not divisible by BLOCK={BLOCK}"
    )

    out = torch.empty_strided(
        tuple(payload.shape),
        tuple(payload.stride()),
        device=payload.device,
        dtype=torch.bfloat16,
    )
    payload_flat = torch.as_strided(payload, (n_elements,), (1,))
    residual_flat = torch.as_strided(residual, (n_elements,), (1,))
    out_flat = torch.as_strided(out, (n_elements,), (1,))
    gate_flat = torch.as_strided(gate, (batch * channels,), (1,))
    # scalar may be bf16[] or f32[]; view as length-1 for cuTile.
    scalar_flat = scalar.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _broadcast_gate_residual_flat_kernel,
        (gate_flat, payload_flat, scalar_flat, residual_flat, out_flat,
         int(channels), chw, BLOCK),
    )
    return out
