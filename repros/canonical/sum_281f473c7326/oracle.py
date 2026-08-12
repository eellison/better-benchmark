"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle materializes the returned channels-last bf16 scaled GELU-derivative product and emits the sibling f32 channel sum from cooperative N/H/W partial reductions in the same tiled pass, whereas Inductor materializes the pointwise tensor and rereads it through a separate generic reduction over N/H/W; Inductor cannot do this today because its scheduler cannot coordinate a live returned side output with split spatial-batch reduction partials while preserving the required bf16/f32 rounding boundaries for both returns; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to emit a fused materialize-plus-partial-reduction producer for returned tensors with sibling channel reductions."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _materialize_partial_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    sum_ptr,
    M: tl.constexpr,
    C: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
    STORE_DIRECT_SUM: tl.constexpr,
):
    row_group = tl.program_id(0)
    channel_group = tl.program_id(1)

    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (rows < M) & (channels < C)
    offsets = rows * C + channels

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    gamma = tl.full([1, 1], 1.7015043497085571, tl.float32)
    rsqrt2 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    one = tl.full([1, 1], 1.0, tl.float32)
    half = tl.full([1, 1], 0.5, tl.float32)
    neg_half = tl.full([1, 1], -0.5, tl.float32)
    rsqrt2pi = tl.full([1, 1], 0.3989422804014327, tl.float32)

    scaled_lhs = lhs * gamma
    scaled_lhs_for_sum = (lhs * gamma).to(tl.bfloat16).to(tl.float32)
    cdf = (libdevice.erf(rhs * rsqrt2) + one) * half
    pdf_term = rhs * (libdevice.exp((rhs * rhs) * neg_half) * rsqrt2pi)
    value = scaled_lhs * (cdf + pdf_term)
    value_for_sum = scaled_lhs_for_sum * (cdf + pdf_term)
    value_bf16 = value.to(tl.bfloat16)
    value_for_sum_bf16 = value_for_sum.to(tl.bfloat16)

    tl.store(out_ptr + offsets, value_bf16, mask=mask)

    partial = tl.sum(tl.where(mask, value_for_sum_bf16.to(tl.float32), 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = out_channels < C
    if STORE_DIRECT_SUM:
        tl.store(sum_ptr + out_channels, partial, mask=channel_mask)
    else:
        tl.store(partial_ptr + row_group * C + out_channels, partial, mask=channel_mask)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    C: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (groups < NUM_GROUPS) & (channels < C)
    values = tl.load(
        partial_ptr + groups * C + channels,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(sum_ptr + out_channels, total, mask=out_channels < C)


# (T([128,16,96,96], bf16, stride=(147456,1,1536,16)), T([128,16,96,96], bf16, stride=(147456,1,1536,16)))
@oracle_impl(hardware="B200", point="75a244e4", ROW_BLOCK=256, BLOCK_C=16, FINAL_BLOCK_C=16, num_warps=8)
# (T([128,32,96,96], bf16, stride=(294912,1,3072,32)), T([128,32,96,96], bf16, stride=(294912,1,3072,32)))
@oracle_impl(hardware="B200", point="9fd4ebb8", ROW_BLOCK=128, BLOCK_C=32, FINAL_BLOCK_C=32, num_warps=8)
# (T([128,128,48,48], bf16, stride=(294912,1,6144,128)), T([128,128,48,48], bf16, stride=(294912,1,6144,128)))
@oracle_impl(hardware="B200", point="8bfea79e", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# (T([128,256,24,24], bf16, stride=(147456,1,6144,256)), T([128,256,24,24], bf16, stride=(147456,1,6144,256)))
@oracle_impl(hardware="B200", point="d0c6a4b4", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# (T([128,768,12,12], bf16, stride=(110592,1,9216,768)), T([128,768,12,12], bf16, stride=(110592,1,9216,768)))
@oracle_impl(hardware="B200", point="1b161b98", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# (T([128,768,6,6], bf16, stride=(27648,1,4608,768)), T([128,768,6,6], bf16, stride=(27648,1,4608,768)))
@oracle_impl(hardware="B200", point="bd252f89", ROW_BLOCK=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    lhs, rhs = inputs
    n = int(lhs.shape[0])
    c = int(lhs.shape[1])
    h = int(lhs.shape[2])
    w = int(lhs.shape[3])
    m = n * h * w

    out = torch.empty_strided(
        tuple(lhs.shape),
        tuple(lhs.stride()),
        device=lhs.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((c,), (1,), device=lhs.device, dtype=torch.float32)

    num_groups = triton.cdiv(m, ROW_BLOCK)
    direct_sum = num_groups == 1
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty_strided(
            (num_groups, c),
            (c, 1),
            device=lhs.device,
            dtype=torch.float32,
        )

    _materialize_partial_kernel[(num_groups, triton.cdiv(c, BLOCK_C))](
        lhs,
        rhs,
        out,
        partial,
        sum_out,
        M=m,
        C=c,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        STORE_DIRECT_SUM=direct_sum,
        num_warps=num_warps,
    )
    if not direct_sum:
        group_block = 1 << (num_groups - 1).bit_length()
        _final_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
            partial,
            sum_out,
            C=c,
            NUM_GROUPS=num_groups,
            GROUP_BLOCK=group_block,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=8,
        )

    return (out, sum_out)
