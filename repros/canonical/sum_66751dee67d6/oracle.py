"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete ConvNeXtV2 bf16/fp32 residual add and channel-sum scope, including the
visible f32 add output, the visible bf16-rounded add output with the captured
non-contiguous stride, and the final fp32 channel sum over those bf16-rounded
values. Inductor lowers the add, cast, materialized side outputs, and
dependent dim-(0,2,3) reduction as generic scheduled regions with repeated
large-tensor traffic; it cannot fuse a required live materialized producer with
its sibling reduction while preserving both output layouts and bf16 reduction
rounding boundaries. The fix is SCHEDULER_FUSION: emit the strided add/cast
stores and per-channel partial reductions from the same producer tile, then
perform only a compact partial-sum epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 80
H = 56
W = 56
ROWS = N * H * W


@triton.jit
def _materialize_add_partial_sum_kernel(
    arg0_ptr,
    arg1_ptr,
    add_ptr,
    bf16_ptr,
    partial_ptr,
    ARG0_S0: tl.constexpr,
    ARG0_S1: tl.constexpr,
    ARG0_S2: tl.constexpr,
    ARG0_S3: tl.constexpr,
    ARG1_S0: tl.constexpr,
    ARG1_S1: tl.constexpr,
    ARG1_S2: tl.constexpr,
    ARG1_S3: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    OUT_S3: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    ROWS_: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_group = tl.program_id(0)
    c_block = tl.program_id(1)
    channels = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = channels < C_
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for offset in tl.range(0, GROUP_ROWS, BLOCK_ROWS):
        linear_rows = row_group * GROUP_ROWS + offset + tl.arange(0, BLOCK_ROWS)
        row_mask = linear_rows < ROWS_

        nhw = H_ * W_
        n = linear_rows // nhw
        wh = linear_rows - n * nhw
        w = wh // H_
        h = wh - w * H_

        mask = row_mask[:, None] & c_mask[None, :]
        arg0_offsets = (
            n[:, None] * ARG0_S0
            + channels[None, :] * ARG0_S1
            + h[:, None] * ARG0_S2
            + w[:, None] * ARG0_S3
        )
        arg1_offsets = (
            n[:, None] * ARG1_S0
            + channels[None, :] * ARG1_S1
            + h[:, None] * ARG1_S2
            + w[:, None] * ARG1_S3
        )
        out_offsets = (
            n[:, None] * OUT_S0
            + channels[None, :] * OUT_S1
            + h[:, None] * OUT_S2
            + w[:, None] * OUT_S3
        )

        value = tl.load(arg1_ptr + arg1_offsets, mask=mask, other=0.0).to(tl.float32)
        value += tl.load(arg0_ptr + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
        rounded = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

        tl.store(add_ptr + out_offsets, value, mask=mask)
        tl.store(bf16_ptr + out_offsets, rounded, mask=mask)
        acc += tl.sum(tl.where(mask, rounded.to(tl.float32), 0.0), axis=0)

    tl.store(
        partial_ptr + row_group * C_ + channels,
        acc,
        mask=c_mask,
    )


@triton.jit
def _final_channel_sum_kernel(
    partial_ptr,
    sum_ptr,
    C_: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BLOCK_GROUPS)
    mask = (groups[:, None] < NUM_GROUPS) & (channels[None, :] < C_)
    values = tl.load(
        partial_ptr + groups[:, None] * C_ + channels[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + channels, rounded, mask=channels < C_)


# e47b47c7: (T([128,80,56,56], bf16, stride=(250880,1,4480,80)),
#            T([128,80,56,56], f32, stride=(250880,1,80,4480)))
@oracle_impl(
    hardware="B200",
    point="e47b47c7",
    GROUP_ROWS=2048,
    BLOCK_ROWS=64,
    BLOCK_C=32,
    FINAL_BLOCK_GROUPS=256,
    FINAL_BLOCK_C=32,
    producer_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    GROUP_ROWS: int,
    BLOCK_ROWS: int,
    BLOCK_C: int,
    FINAL_BLOCK_GROUPS: int,
    FINAL_BLOCK_C: int,
    producer_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1 = inputs

    add = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, C, C * H),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    rounded = torch.empty_strided(
        (N, C, H, W),
        (C * H * W, 1, C, C * H),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    num_groups = triton.cdiv(ROWS, GROUP_ROWS)
    partial = torch.empty_strided(
        (num_groups, C),
        (C, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)

    _materialize_add_partial_sum_kernel[
        (num_groups, triton.cdiv(C, BLOCK_C))
    ](
        arg0_1,
        arg1_1,
        add,
        rounded,
        partial,
        ARG0_S0=arg0_1.stride(0),
        ARG0_S1=arg0_1.stride(1),
        ARG0_S2=arg0_1.stride(2),
        ARG0_S3=arg0_1.stride(3),
        ARG1_S0=arg1_1.stride(0),
        ARG1_S1=arg1_1.stride(1),
        ARG1_S2=arg1_1.stride(2),
        ARG1_S3=arg1_1.stride(3),
        OUT_S0=add.stride(0),
        OUT_S1=add.stride(1),
        OUT_S2=add.stride(2),
        OUT_S3=add.stride(3),
        C_=C,
        H_=H,
        W_=W,
        ROWS_=ROWS,
        GROUP_ROWS=GROUP_ROWS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_C=BLOCK_C,
        num_warps=producer_warps,
        num_stages=3,
    )
    _final_channel_sum_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial,
        sum_out,
        C_=C,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=FINAL_BLOCK_GROUPS,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    return add, rounded, sum_out
