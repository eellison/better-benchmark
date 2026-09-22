"""cuTile port of sum_sum_cc37fdc0b4ba: GhostNet product/gate + channel sum.

Two kernels mirroring the Triton oracle:
  1. _product_gate_flat_kernel: flat NC-block program with a static-range
     over HW=49, producing gate_f32 side output, gated bf16, and (only pid=0)
     the zero scalar.
  2. _sum_gated_kernel: 2D reduce over N, producing (C,) channel sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 960
H = 7
W = 7
HW = H * W
NUM_NC = N * C


@ct.kernel
def _product_gate_flat_kernel(
    x_ptr,             # bf16 flat [N * HW * C]  (channels-last view)
    y_ptr,             # bf16 flat [N * HW * C]
    gate_ptr,          # bf16 flat [N * C]
    gate_f32_ptr,      # f32  flat [N * C]
    gated_bf16_ptr,    # bf16 flat [N * C]
    zero_ptr,          # f32 scalar
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offs < NUM_NC
    n = offs // C
    c = offs - n * C

    acc = ct.zeros((BLOCK,), dtype=ct.float32)
    for hw in range(HW):
        gather_offsets = n * (C * HW) + hw * C + c
        x = ct.gather(x_ptr, gather_offsets, mask=active, padding_value=0.0)
        y = ct.gather(y_ptr, gather_offsets, mask=active, padding_value=0.0)
        prod = ct.astype(x, ct.float32) * ct.astype(y, ct.float32)
        acc = acc + ct.astype(ct.astype(prod, ct.bfloat16), ct.float32)

    rounded_sum = ct.astype(ct.astype(acc, ct.bfloat16), ct.float32)
    gate = ct.gather(gate_ptr, offs, mask=active, padding_value=0.0)
    gate_f32 = ct.astype(gate, ct.float32)
    gate_active = (gate_f32 > -3.0) & (gate_f32 < 3.0)
    gated = ct.where(gate_active, rounded_sum * 0.16666666666666666, 0.0)

    ct.scatter(gate_f32_ptr, offs, gate_f32, mask=active)
    ct.scatter(gated_bf16_ptr, offs,
               ct.astype(gated, ct.bfloat16), mask=active)
    if pid == 0:
        ct.store(zero_ptr, index=(0,),
                 tile=ct.zeros((1,), dtype=ct.float32))


@ct.kernel
def _sum_gated_kernel(
    gated_bf16_ptr,  # bf16 [N, C]
    out_ptr,         # f32 [C]
    BLOCK_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    c_tile = ct.bid(0)
    values = ct.load(gated_bf16_ptr, index=(0, c_tile),
                     shape=(BLOCK_N, BLOCK_C))
    vf = ct.astype(values, ct.float32)
    sums = ct.sum(vf, axis=0)  # -> (BLOCK_C,)
    rounded = ct.astype(ct.astype(sums, ct.bfloat16), ct.float32)
    ct.store(out_ptr, index=(c_tile,), tile=rounded)


@oracle_impl(hardware="B200", point="e93c5538", BLOCK=256, BLOCK_C=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK: int, BLOCK_C: int, BLOCK_N: int):
    x, y, gate = inputs
    gate_f32 = torch.empty_strided(
        (N, C, 1, 1), (C, 1, 1, 1), device=gate.device, dtype=torch.float32)
    zero = torch.empty((), device=gate.device, dtype=torch.float32)
    gated_bf16 = torch.empty_strided(
        (N, C, 1, 1), (C, 1, 1, 1), device=gate.device, dtype=torch.bfloat16)
    channel_sum = torch.empty((C,), device=gate.device, dtype=torch.float32)

    # x, y are channels-last [N, C, H, W]. Physical layout is [N, HW, C].
    # Flat 1D metadata-only views.
    x_flat = torch.as_strided(x, (x.numel(),), (1,))
    y_flat = torch.as_strided(y, (y.numel(),), (1,))
    gate_flat = gate.view(N * C)
    gate_f32_flat = gate_f32.view(N * C)
    gated_bf16_flat = gated_bf16.view(N * C)
    zero_1d = zero.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(NUM_NC, BLOCK), 1, 1),
        _product_gate_flat_kernel,
        (x_flat, y_flat, gate_flat, gate_f32_flat, gated_bf16_flat,
         zero_1d, BLOCK),
    )
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _sum_gated_kernel,
        (gated_bf16.view(N, C), channel_sum, BLOCK_C, BLOCK_N),
    )
    return gate_f32, zero, gated_bf16, channel_sum
