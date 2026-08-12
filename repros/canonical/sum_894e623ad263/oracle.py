"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete NFNet bf16 scatter/expand/div plus sigmoid-gradient materialization and channel-sum scope by replacing the zero-fill/as_strided_scatter/as_strided/expand chain with direct loads from the `[128, 2304]` source, writing the required contiguous `[128, 2304, 7, 7]` bf16 output, and accumulating per-channel sums from those rounded bf16 output values. Inductor materializes the structured singleton scatter/expand producer and then schedules the sibling reduction through generic pointwise/reduction code, because scheduler/codegen does not represent this average-pool-style scatter expansion as a source-space producer that can feed both a full returned tensor and channel reduction while preserving the bf16 div, output cast, and sum cast boundaries. The fix is SCATTER_REDUCE: add a structured scatter/expand plus channel-reduction lowering for singleton spatial sources."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 2304
H = 7
W = 7
HW = H * W


@triton.jit
def _materialize_spatial_sum_kernel(
    scale_ptr,
    act_ptr,
    out_ptr,
    partial_ptr,
    ACT_S0: tl.constexpr,
    ACT_S1: tl.constexpr,
    ACT_S2: tl.constexpr,
    ACT_S3: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    OUT_S3: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    h = hw // W_
    w = hw - h * W_
    mask = (channels[:, None] < C_) & (hw[None, :] < HW_)

    scale = tl.load(
        scale_ptr + n * C_ + channels,
        mask=channels < C_,
        other=0.0,
    ).to(tl.float32)
    scale = (scale / 49.0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    act_offsets = (
        n * ACT_S0
        + channels[:, None] * ACT_S1
        + h[None, :] * ACT_S2
        + w[None, :] * ACT_S3
    )
    act = tl.load(act_ptr + act_offsets, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(act)
    value = scale[:, None] * sigmoid * (act * (1.0 - sigmoid) + 1.0)
    value_bf16 = value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    out_offsets = (
        n * OUT_S0
        + channels[:, None] * OUT_S1
        + h[None, :] * OUT_S2
        + w[None, :] * OUT_S3
    )
    tl.store(out_ptr + out_offsets, value_bf16, mask=mask)

    spatial = tl.sum(tl.where(mask, value_bf16.to(tl.float32), 0.0), axis=1)
    tl.store(partial_ptr + n * C_ + channels, spatial, mask=channels < C_)


@triton.jit
def _final_channel_sum_kernel(
    partial_ptr,
    sum_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N_) & (channels[None, :] < C_)
    values = tl.load(
        partial_ptr + rows[:, None] * C_ + channels[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + channels, rounded, mask=channels < C_)


# timm_nfnet_l0 train, pooled bf16 [128,2304] expanded over [7,7] with channels-last activation.
@oracle_impl(
    hardware="B200",
    point="943c9ed8",
    BLOCK_HW=64,
    BLOCK_C=64,
    FINAL_BLOCK_C=64,
    materialize_warps=8,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    materialize_warps: int,
    final_warps: int,
):
    scale_src, activation, _sp0, _sp1, _sp2, _sp3, _sp4, _sp5, _sp6 = inputs
    del _sp0, _sp1, _sp2, _sp3, _sp4, _sp5, _sp6

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=scale_src.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided((N, C), (C, 1), device=scale_src.device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=scale_src.device, dtype=torch.float32)

    _materialize_spatial_sum_kernel[(N, triton.cdiv(C, BLOCK_C))](
        scale_src,
        activation,
        out,
        partial,
        ACT_S0=activation.stride(0),
        ACT_S1=activation.stride(1),
        ACT_S2=activation.stride(2),
        ACT_S3=activation.stride(3),
        OUT_S0=out.stride(0),
        OUT_S1=out.stride(1),
        OUT_S2=out.stride(2),
        OUT_S3=out.stride(3),
        C_=C,
        H_=H,
        W_=W,
        HW_=HW,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=materialize_warps,
        num_stages=3,
    )
    _final_channel_sum_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial,
        sum_out,
        N_=N,
        C_=C,
        BLOCK_N=N,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    return out, sum_out
