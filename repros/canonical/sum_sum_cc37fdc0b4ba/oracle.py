"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 product spatial-sum, explicit bf16 round-trip, hard-sigmoid-gradient gate, returned f32 side tensor, returned zero scalar, returned bf16 gate tensor, and final bf16 batch/channel sum in two layout-aware Triton kernels, whereas Inductor lowers the multiply, spatial reduction, cast/gate epilogue, and sibling channel reduction as generic scheduled fragments; Inductor cannot do this today because its reduction scheduler does not keep the channels-last 7x7 product reduction and dependent bf16 gate resident while emitting both the materialized side output and the second reduction; the fix is SCHEDULER_FUSION: add a fused spatial-reduction epilogue lowering that preserves bf16 cast boundaries and feeds the follow-on channel sum without a generic materialization schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C = 960
H = 7
W = 7
HW = H * W
NUM_NC = N * C


@triton.jit
def _product_gate_flat_kernel(
    x_ptr,
    y_ptr,
    gate_ptr,
    gate_f32_ptr,
    zero_ptr,
    gated_bf16_ptr,
    num_nc: tl.constexpr,
    n_total: tl.constexpr,
    c_total: tl.constexpr,
    hw_total: tl.constexpr,
    block: tl.constexpr,
):
    offsets_nc = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets_nc < num_nc
    n = offsets_nc // c_total
    c = offsets_nc - n * c_total

    acc = tl.zeros((block,), tl.float32)
    for hw in tl.static_range(0, 49):
        offsets = n * (c_total * hw_total) + hw * c_total + c
        x = tl.load(x_ptr + offsets, mask=active, other=0.0)
        y = tl.load(y_ptr + offsets, mask=active, other=0.0)
        acc += (x * y).to(tl.bfloat16).to(tl.float32)

    rounded_sum = acc.to(tl.bfloat16).to(tl.float32)
    gate_f32 = tl.load(gate_ptr + offsets_nc, mask=active, other=0.0).to(tl.float32)
    gate_active = (gate_f32 > -3.0) & (gate_f32 < 3.0)
    gated = tl.where(gate_active, rounded_sum * 0.16666666666666666, 0.0)

    tl.store(gate_f32_ptr + offsets_nc, gate_f32, mask=active)
    tl.store(gated_bf16_ptr + offsets_nc, gated.to(tl.bfloat16), mask=active)
    if tl.program_id(0) == 0:
        tl.store(zero_ptr, 0.0)


@triton.jit
def _sum_gated_kernel(
    gated_bf16_ptr,
    out_ptr,
    n_total: tl.constexpr,
    c_total: tl.constexpr,
    block_c: tl.constexpr,
    block_n: tl.constexpr,
):
    c = tl.program_id(0) * block_c + tl.arange(0, block_c)
    n = tl.arange(0, block_n)
    c_mask = c < c_total
    n_mask = n < n_total
    offsets = n[:, None] * c_total + c[None, :]
    values = tl.load(gated_bf16_ptr + offsets, mask=n_mask[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(out_ptr + c, sums.to(tl.bfloat16).to(tl.float32), mask=c_mask)


@oracle_impl(hardware="B200", point="e93c5538", block=256, block_c=4, block_n=512, num_warps_prod=4, num_warps_sum=4)
def oracle_forward(inputs, *, block: int, block_c: int, block_n: int, num_warps_prod: int, num_warps_sum: int):
    x, y, gate = inputs
    gate_f32 = torch.empty_strided((N, C, 1, 1), (C, 1, 1, 1), device=gate.device, dtype=torch.float32)
    zero = torch.empty((), device=gate.device, dtype=torch.float32)
    gated_bf16 = torch.empty_strided((N, C, 1, 1), (C, 1, 1, 1), device=gate.device, dtype=torch.bfloat16)
    channel_sum = torch.empty((C,), device=gate.device, dtype=torch.float32)

    _product_gate_flat_kernel[(triton.cdiv(NUM_NC, block),)](
        x,
        y,
        gate,
        gate_f32,
        zero,
        gated_bf16,
        num_nc=NUM_NC,
        n_total=N,
        c_total=C,
        hw_total=HW,
        block=block,
        num_warps=num_warps_prod,
    )
    _sum_gated_kernel[(triton.cdiv(C, block_c),)](
        gated_bf16,
        channel_sum,
        n_total=N,
        c_total=C,
        block_c=block_c,
        block_n=block_n,
        num_warps=num_warps_sum,
    )
    return gate_f32, zero, gated_bf16, channel_sum
