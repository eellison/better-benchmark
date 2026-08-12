"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the
complete NFNet channels-last bf16 add plus exact-GELU-gradient scope, including
the returned `[128,128,48,48]` bf16 materialization and returned f32 copy of the
bf16 `sum([0,2,3])`, from one producer pass plus a small finalizer. Inductor
lowers the bf16 add/gamma rounding, exact erf/exp derivative, visible bf16
product, and sibling channel reduction as generic materialize-then-reduce
regions that reread the producer. It cannot do this today because scheduler
codegen does not keep a returned channels-last exact-GELU-gradient producer and
its compatible channel reduction in one cooperative split-K plan while
preserving every bf16 rounding boundary; the fix is COOPERATIVE_SPLIT_K: emit
the side-output stores and channel partials from the same tile, then finalize the
bf16-rounded channel sums."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


N = 128
C = 128
H = 48
W = 48
HW = H * W
ROWS = N * HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _materialize_partial_kernel(
    lhs0_ptr,
    lhs1_ptr,
    rhs_ptr,
    out_ptr,
    partial_ptr,
    GAMMA_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    NORMAL_PDF_SCALE_: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    channel_group = tl.program_id(1)
    rows = row_group * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (rows < 294912) & (channels < 128)

    n = rows // 2304
    hw = rows - n * 2304
    h = hw // 48
    w = hw - h * 48
    offsets = n * 294912 + h * 6144 + w * 128 + channels

    lhs0 = tl.load(lhs0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    lhs1 = tl.load(lhs1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    add_f32 = lhs0 + lhs1
    add_bf16 = add_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
    mul_one_bf16 = add_bf16.to(tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    scaled_for_sum = (mul_one_bf16.to(tl.float32) * GAMMA_).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    cdf = (libdevice.erf(rhs * RSQRT2_) + 1.0) * 0.5
    pdf_term = rhs * (libdevice.exp((rhs * rhs) * -0.5) * NORMAL_PDF_SCALE_)
    derivative = cdf + pdf_term
    value_bf16 = ((add_f32 * GAMMA_) * derivative).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    value_for_sum_bf16 = (scaled_for_sum.to(tl.float32) * derivative).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(out_ptr + offsets, value_bf16, mask=mask)

    partial = tl.sum(tl.where(mask, value_for_sum_bf16.to(tl.float32), 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    tl.store(partial_ptr + row_group * 128 + out_channels, partial, mask=out_channels < 128)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channel_group = tl.program_id(0)
    groups = tl.arange(0, GROUP_BLOCK)[:, None]
    channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    mask = (groups < NUM_GROUPS) & (channels < 128)

    values = tl.load(
        partial_ptr + groups * 128 + channels,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    out_channels = channel_group * BLOCK_C + tl.arange(0, BLOCK_C)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + out_channels, rounded, mask=out_channels < 128)


@oracle_impl(
    hardware="B200",
    point="8e74fca9",
    ROW_BLOCK=128,
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
    lhs0, lhs1, rhs = inputs
    num_groups = triton.cdiv(ROWS, ROW_BLOCK)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=lhs0.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (num_groups, C),
        (C, 1),
        device=lhs0.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C,), (1,), device=lhs0.device, dtype=torch.float32)

    _materialize_partial_kernel[(num_groups, triton.cdiv(C, BLOCK_C))](
        lhs0,
        lhs1,
        rhs,
        out,
        partial,
        GAMMA_=GAMMA,
        RSQRT2_=RSQRT2,
        NORMAL_PDF_SCALE_=NORMAL_PDF_SCALE,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_producer,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial,
        sum_out,
        NUM_GROUPS=num_groups,
        GROUP_BLOCK=_ceil_pow2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=num_warps_final,
        num_stages=3,
    )
    return out, sum_out
