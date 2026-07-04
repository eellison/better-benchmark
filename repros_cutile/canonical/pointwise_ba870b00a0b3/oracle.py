"""cuTile port of pointwise_ba870b00a0b3: MobileNetV3 hard-sigmoid broadcast multiply.

For a bf16 [N, C, 1, 1] gate and bf16 [N, C, H, W] x, compute
  gate_scale = ((gate + 3.0).clip(0, 6) / 6.0).to(bf16)
  out = x * gate_scale
with broadcasting over the spatial (H, W) axes.

We tile over (nc, hw) — one program handles a fixed (BLOCK_NC, BLOCK_HW) tile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _hardsigmoid_broadcast_mul_kernel(
    gate_ptr,     # bf16 [N*C]
    x_ptr,        # bf16 [N*C, HW]
    out_ptr,      # bf16 [N*C, HW]
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)
    col_tile = ct.bid(1)

    # Load one gate scalar for this row.
    gate = ct.load(gate_ptr, index=(row,), shape=(1,))
    gate_f = ct.astype(gate, ct.float32)
    shifted = gate_f + 3.0
    # clamp to [0, 6]
    clamped_lo = ct.where(shifted > 0.0, shifted, ct.zeros(shape=(1,), dtype=ct.float32))
    clamped = ct.where(clamped_lo < 6.0, clamped_lo, ct.full(shape=(1,), fill_value=6.0, dtype=ct.float32))
    scale_bf16 = ct.astype(clamped * (1.0 / 6.0), ct.bfloat16)
    scale_f = ct.astype(scale_bf16, ct.float32)

    # Load a spatial tile of x.
    x = ct.load(x_ptr, index=(row, col_tile), shape=(1, BLOCK_HW))
    x_f = ct.astype(x, ct.float32)
    # Broadcast scale (1,) -> (1, BLOCK_HW) via reshape.
    scale_2d = ct.reshape(scale_f, (1, 1))
    out = ct.astype(x_f * scale_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, col_tile), tile=out)


def _launch(gate, x):
    n = int(x.shape[0])
    c = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    hw = h * w
    nc = n * c
    # Pick BLOCK_HW as smallest power of two >= hw.
    block_hw = 1
    while block_hw < hw:
        block_hw *= 2
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    gate_flat = gate.reshape(nc)
    x_flat = x.reshape(nc, hw)
    out_flat = output.reshape(nc, hw)
    stream = torch.cuda.current_stream()
    grid = (nc, ct.cdiv(hw, block_hw), 1)
    ct.launch(stream, grid, _hardsigmoid_broadcast_mul_kernel,
              (gate_flat, x_flat, out_flat, block_hw))
    return output


@oracle_impl(hardware="B200", point="d7edb67c")
@oracle_impl(hardware="B200", point="f9969da7")
@oracle_impl(hardware="B200", point="f76fad38")
@oracle_impl(hardware="B200", point="0c85aae0")
@oracle_impl(hardware="B200", point="be7a1f98")
@oracle_impl(hardware="B200", point="38d11fef")
def oracle_forward(inputs):
    gate, x = inputs
    return _launch(gate, x)
