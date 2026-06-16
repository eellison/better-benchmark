"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete NFNet bf16 broadcast-scale GELU-gradient scope, including the as_strided/expand-derived bf16 scale path, the visible contiguous bf16 `[128,3072,6,6]` producer, and the returned f32 copy of its bf16 `sum([0, 2, 3])`, whereas Inductor schedules the scatter/broadcast pointwise chain and the channel reduction as generic materialize-then-reduce regions. Inductor cannot do this today because scheduler/codegen lacks a cooperative split-K template that sinks an exact-erf/exp producer with explicit bf16 rounding boundaries into a returned dense side output and a sibling channel reduction. The fix is COOPERATIVE_SPLIT_K: materialize the required bf16 output and accumulate channel partials from those rounded values in one producer pass, then finalize the bf16-rounded channel sums directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 3072
HEIGHT = 6
WIDTH = 6
HW = HEIGHT * WIDTH
ROWS = BATCH * HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _materialize_partial_kernel(
    scale_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    HW_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    NORMAL_PDF_SCALE_: tl.constexpr,
    GAMMA_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    channel_group = tl.program_id(1)
    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (rows < ROWS_) & (channels < CHANNELS_)

    n = rows // HW_
    hw = rows - n * HW_
    h = hw // 6
    w = hw - h * 6

    scale_offsets = n * CHANNELS_ + channels
    rhs_offsets = n * (CHANNELS_ * HW_) + h * (6 * CHANNELS_) + w * CHANNELS_ + channels
    out_offsets = n * (CHANNELS_ * HW_) + channels * HW_ + hw

    scale_source = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
    div_bf16 = (scale_source / 36.0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    scale = (div_bf16 * GAMMA_).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    rhs = tl.load(rhs_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)
    erf_term = libdevice.erf(rhs * RSQRT2_)
    cdf = (erf_term + 1.0) * 0.5
    exp_term = libdevice.exp((rhs * rhs) * -0.5)
    pdf_term = rhs * (exp_term * NORMAL_PDF_SCALE_)
    derivative = cdf + pdf_term
    value_bf16 = (scale * derivative).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_ptr + out_offsets, value_bf16, mask=mask)

    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = out_channels < CHANNELS_
    partial = tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=0)
    tl.store(
        partial_ptr + row_group * CHANNELS_ + out_channels,
        partial,
        mask=channel_mask,
    )


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    CHANNELS_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (groups < NUM_GROUPS) & (channels < CHANNELS_)

    values = tl.load(
        partial_ptr + groups * CHANNELS_ + channels,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + out_channels, rounded, mask=out_channels < CHANNELS_)


@oracle_impl(
    hardware="B200",
    point="74c4e87d",
    ROW_BLOCK=64,
    BLOCK_C=64,
    FINAL_BLOCK_C=32,
    num_warps_producer=8,
    num_warps_final=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps_producer: int,
    num_warps_final: int,
):
    arg0_1, arg1_1, *_ = inputs
    num_groups = triton.cdiv(ROWS, ROW_BLOCK)

    out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (num_groups, CHANNELS),
        (CHANNELS, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((CHANNELS,), (1,), device=arg0_1.device, dtype=torch.float32)

    _materialize_partial_kernel[(num_groups, triton.cdiv(CHANNELS, BLOCK_C))](
        arg0_1,
        arg1_1,
        out,
        partial,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        HW_=HW,
        RSQRT2_=RSQRT2,
        NORMAL_PDF_SCALE_=NORMAL_PDF_SCALE,
        GAMMA_=GAMMA,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_producer,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial,
        sum_out,
        CHANNELS_=CHANNELS,
        NUM_GROUPS=num_groups,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )

    return out, sum_out
