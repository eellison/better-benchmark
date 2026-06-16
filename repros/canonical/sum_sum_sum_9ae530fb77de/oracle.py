"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 gate-gradient fragment in one spatial Triton loop plus one compact finalize loop, including the returned bf16 sigmoid gate, returned bf16 `arg0 * 0.2 * arg3 * 2.0` tensor, global f32 sum over the bf16-rounded gated product, bf16-rounded per-`(N,C)` spatial reduction epilogue, returned bf16 gate-gradient tensor, and final bf16-rounded f32 channel vector, whereas Inductor schedules the sigmoid side output, dense scaled producer, scalar reduction, spatial reduction, derivative multiply, and channel sum as separate generic regions around materialized intermediates; Inductor cannot fuse this full multi-output scope today because scheduler/codegen does not form one dtype-boundary-preserving reduction plan that shares the channels-last spatial loop across the scalar, per-row, and channel reductions; the fix is SCHEDULER_FUSION: emit the returned side outputs and compact scalar/channel partials from one spatial loop nest, then finalize the scalar and channel vector directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _spatial_gate_kernel(
    arg0_ptr,
    gate_ptr,
    arg2_ptr,
    scalar_ptr,
    sigmoid_out_ptr,
    scaled_out_ptr,
    row_out_ptr,
    scalar_partials_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NUM_C_BLOCKS: tl.constexpr,
    A0_S0: tl.constexpr,
    A0_S1: tl.constexpr,
    A0_S2: tl.constexpr,
    A0_S3: tl.constexpr,
    A2_S0: tl.constexpr,
    A2_S1: tl.constexpr,
    A2_S2: tl.constexpr,
    A2_S3: tl.constexpr,
    OUT_S0: tl.constexpr,
    OUT_S1: tl.constexpr,
    OUT_S2: tl.constexpr,
    OUT_S3: tl.constexpr,
    GATE_S0: tl.constexpr,
    GATE_S1: tl.constexpr,
    SIG_S0: tl.constexpr,
    SIG_S1: tl.constexpr,
    ROW_S0: tl.constexpr,
    ROW_S1: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)[:, None]
    c_b = c[None, :]

    hw_size = H * W
    h = hw // W
    w = hw - h * W
    mask = (hw < hw_size) & (c_b < C) & (n < N)

    arg0_offsets = n * A0_S0 + c_b * A0_S1 + h * A0_S2 + w * A0_S3
    arg2_offsets = n * A2_S0 + c_b * A2_S1 + h * A2_S2 + w * A2_S3
    out_offsets = n * OUT_S0 + c_b * OUT_S1 + h * OUT_S2 + w * OUT_S3
    gate_offsets = n * GATE_S0 + c * GATE_S1
    sigmoid_offsets = n * SIG_S0 + c * SIG_S1
    row_offsets = n * ROW_S0 + c * ROW_S1
    c_mask = c < C

    arg0 = tl.load(
        arg0_ptr + arg0_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    arg2 = tl.load(
        arg2_ptr + arg2_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offsets, mask=c_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )

    mul0 = (arg0 * 0.2).to(tl.bfloat16, fp_downcast_rounding="rtne")
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16, fp_downcast_rounding="rtne")
    sigmoid_f32 = sigmoid.to(tl.float32)

    gate_prod = (arg2 * sigmoid_f32[None, :]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    gate_prod2 = (gate_prod.to(tl.float32) * 2.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    scalar_terms = (mul0.to(tl.float32) * gate_prod2.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    scalar_by_c = tl.sum(tl.where(mask, scalar_terms.to(tl.float32), 0.0), axis=0)
    scalar_partial = tl.sum(tl.where(c_mask, scalar_by_c, 0.0), axis=0)
    tl.store(scalar_partials_ptr + n * NUM_C_BLOCKS + c_block, scalar_partial)

    mul4 = (mul0.to(tl.float32) * scalar).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    scaled = (mul4.to(tl.float32) * 2.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(scaled_out_ptr + out_offsets, scaled, mask=mask)

    spatial_terms = (scaled.to(tl.float32) * arg2).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    spatial = tl.sum(tl.where(mask, spatial_terms.to(tl.float32), 0.0), axis=0)
    spatial = spatial.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    deriv = sigmoid_f32 * (1.0 - sigmoid_f32)
    row_value = (spatial * deriv).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(sigmoid_out_ptr + sigmoid_offsets, sigmoid, mask=c_mask)
    tl.store(row_out_ptr + row_offsets, row_value, mask=c_mask)


@triton.jit
def _channel_finalize_kernel(
    row_out_ptr,
    final_out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    ROW_S0: tl.constexpr,
    ROW_S1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)[:, None]
    mask = (n < N) & (c[None, :] < C)
    offsets = n * ROW_S0 + c[None, :] * ROW_S1
    values = tl.load(row_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(final_out_ptr + c, rounded, mask=c < C)


@triton.jit
def _scalar_finalize_kernel(
    scalar_partials_ptr,
    scalar_out_ptr,
    COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < COUNT
    values = tl.load(scalar_partials_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    total = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(scalar_out_ptr, total)


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


# fd33c7c3: timm dm_nfnet_f0 train, channels-last [128,1536,6,6].
@oracle_impl(
    hardware="B200",
    point="fd33c7c3",
    BLOCK_HW=64,
    BLOCK_C=128,
    FINAL_BLOCK_C=32,
    spatial_warps=1,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    spatial_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    num_c_blocks = triton.cdiv(channels, BLOCK_C)
    num_channel_programs = triton.cdiv(channels, FINAL_BLOCK_C)

    sigmoid_out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    scaled_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    scalar_out = torch.empty((), device=arg1_1.device, dtype=torch.float32)
    scalar_partials = torch.empty(
        (n, num_c_blocks),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _spatial_gate_kernel[(n, num_c_blocks)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        sigmoid_out,
        scaled_out,
        row_out,
        scalar_partials,
        N=n,
        C=channels,
        H=height,
        W=width,
        NUM_C_BLOCKS=num_c_blocks,
        A0_S0=arg0_1.stride(0),
        A0_S1=arg0_1.stride(1),
        A0_S2=arg0_1.stride(2),
        A0_S3=arg0_1.stride(3),
        A2_S0=arg2_1.stride(0),
        A2_S1=arg2_1.stride(1),
        A2_S2=arg2_1.stride(2),
        A2_S3=arg2_1.stride(3),
        OUT_S0=scaled_out.stride(0),
        OUT_S1=scaled_out.stride(1),
        OUT_S2=scaled_out.stride(2),
        OUT_S3=scaled_out.stride(3),
        GATE_S0=arg1_1.stride(0),
        GATE_S1=arg1_1.stride(1),
        SIG_S0=sigmoid_out.stride(0),
        SIG_S1=sigmoid_out.stride(1),
        ROW_S0=row_out.stride(0),
        ROW_S1=row_out.stride(1),
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=spatial_warps,
        num_stages=4,
    )
    _channel_finalize_kernel[(num_channel_programs,)](
        row_out,
        final_out,
        N=n,
        C=channels,
        ROW_S0=row_out.stride(0),
        ROW_S1=row_out.stride(1),
        BLOCK_N=_ceil_pow2(n),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )
    _scalar_finalize_kernel[(1,)](
        scalar_partials,
        scalar_out,
        COUNT=n * num_c_blocks,
        BLOCK=_ceil_pow2(n * num_c_blocks),
        num_warps=final_warps,
        num_stages=3,
    )

    return sigmoid_out, scalar_out, scaled_out, row_out, final_out
