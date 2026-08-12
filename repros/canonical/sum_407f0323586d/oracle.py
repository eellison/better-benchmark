"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete NFNet bf16 residual-SiLU-gradient materialization plus channel sum by
streaming the shared channels-last producer once, storing the visible bf16
activation, and accumulating per-channel partial sums from those bf16-rounded
stored values. Inductor lowers the bf16 residual add, sigmoid derivative
pointwise chain, returned activation, and dependent dim-(0,2,3) reduction as
generic scheduled work that rereads the same large producer; it cannot fuse a
required materialized side output with the sibling channel reduction while
preserving the bf16 add/output/reduction rounding boundaries. The fix is
SCHEDULER_FUSION: emit a multi-output pointwise-plus-reduction schedule for
channels-last NFNet activation gradients."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 128
H = 56
W = 56
K_TOTAL = N * H * W


@triton.jit
def _materialize_partial_sum_kernel(
    x0_ptr,
    x1_ptr,
    gate_ptr,
    out_ptr,
    partial_ptr,
    C_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    offsets = k_offsets[:, None] * C_ + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL_) & (c_offsets[None, :] < C_)

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    residual = (x0 + x1).to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    value = residual * sigmoid * (gate * (1.0 - sigmoid) + 1.0)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + offsets, value_bf16, mask=mask)
    partial = tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + tl.program_id(0) * C_ + c_offsets,
        partial,
        mask=c_offsets < C_,
    )


@triton.jit
def _final_channel_sum_kernel(
    partial_ptr,
    sum_ptr,
    C_: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = (partial_offsets[:, None] < NUM_PARTIALS) & (c_offsets[None, :] < C_)
    partials = tl.load(
        partial_ptr + partial_offsets[:, None] * C_ + c_offsets[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(partials, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + c_offsets, rounded, mask=c_offsets < C_)


# 97f7c01b: three T([128,128,56,56], bf16, stride=(401408,1,7168,128))
@oracle_impl(
    hardware="B200",
    point="97f7c01b",
    BLOCK_K=256,
    BLOCK_C=32,
    FINAL_BLOCK_C=32,
    producer_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    producer_warps: int,
    final_warps: int,
):
    x0, x1, gate = inputs
    num_partials = triton.cdiv(K_TOTAL, BLOCK_K)

    out = torch.empty_strided(
        (N, C, H, W),
        tuple(int(s) for s in x0.stride()),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty_strided(
        (num_partials, C),
        (C, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C,), (1,), device=x0.device, dtype=torch.float32)

    _materialize_partial_sum_kernel[
        (num_partials, triton.cdiv(C, BLOCK_C))
    ](
        x0,
        x1,
        gate,
        out,
        partials,
        C_=C,
        K_TOTAL_=K_TOTAL,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=producer_warps,
        num_stages=3,
    )
    _final_channel_sum_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partials,
        sum_out,
        C_=C,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_partials),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    return out, sum_out
