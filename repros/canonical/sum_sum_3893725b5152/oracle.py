"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet bf16 gate-gradient fragment, including the returned bf16 `arg0 * 0.2 * 2.0` tensor, the returned bf16 sigmoid gate, the bf16-rounded spatial reduction epilogue, the returned per-`(N,C)` bf16 gate-gradient tensor, and the final returned bf16-rounded f32 channel vector. Inductor currently schedules the scaled full-tensor producer, sigmoid side output, spatial sum, sigmoid-derivative multiply, and channel sum as separate generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that preserves bf16 rounding boundaries while sharing the channels-last dense producer with the dependent reductions. The fix is SCHEDULER_FUSION: emit the returned side outputs and `(N,C)` partial reduction from one spatial loop nest, then finalize the small channel vector with the captured bf16 reduction boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _spatial_gate_kernel(
    arg0_ptr,
    arg1_ptr,
    gate_ptr,
    scaled_out_ptr,
    sigmoid_out_ptr,
    row_out_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    A0_S0: tl.constexpr,
    A0_S1: tl.constexpr,
    A0_S2: tl.constexpr,
    A0_S3: tl.constexpr,
    A1_S0: tl.constexpr,
    A1_S1: tl.constexpr,
    A1_S2: tl.constexpr,
    A1_S3: tl.constexpr,
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
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)[:, None]
    c_b = c[None, :]

    hw_size = H * W
    h = hw // W
    w = hw - h * W
    mask = (hw < hw_size) & (c_b < C) & (n < N)

    arg0_offsets = n * A0_S0 + c_b * A0_S1 + h * A0_S2 + w * A0_S3
    arg1_offsets = n * A1_S0 + c_b * A1_S1 + h * A1_S2 + w * A1_S3
    out_offsets = n * OUT_S0 + c_b * OUT_S1 + h * OUT_S2 + w * OUT_S3

    arg0 = tl.load(arg0_ptr + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
    arg1 = tl.load(arg1_ptr + arg1_offsets, mask=mask, other=0.0).to(tl.float32)

    mul0 = (arg0 * 0.2).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = (mul0.to(tl.float32) * 2.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    prod = (scaled.to(tl.float32) * arg1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(scaled_out_ptr + out_offsets, scaled, mask=mask)

    spatial = tl.sum(tl.where(mask, prod.to(tl.float32), 0.0), axis=0)
    spatial = spatial.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    c_mask = c < C
    gate_offsets = n * GATE_S0 + c * GATE_S1
    sigmoid_offsets = n * SIG_S0 + c * SIG_S1
    row_offsets = n * ROW_S0 + c * ROW_S1
    gate = tl.load(gate_ptr + gate_offsets, mask=c_mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16, fp_downcast_rounding="rtne")
    sigmoid_f32 = sigmoid.to(tl.float32)
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


# timm nfnet_l0 train, channels-last [128, 1536, 7, 7].
@oracle_impl(
    hardware="B200",
    point="9983a35a",
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
    arg0_1, arg1_1, arg2_1 = inputs
    n = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])

    scaled_out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    sigmoid_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    row_out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (channels,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _spatial_gate_kernel[(n, triton.cdiv(channels, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        scaled_out,
        sigmoid_out,
        row_out,
        N=n,
        C=channels,
        H=height,
        W=width,
        A0_S0=arg0_1.stride(0),
        A0_S1=arg0_1.stride(1),
        A0_S2=arg0_1.stride(2),
        A0_S3=arg0_1.stride(3),
        A1_S0=arg1_1.stride(0),
        A1_S1=arg1_1.stride(1),
        A1_S2=arg1_1.stride(2),
        A1_S3=arg1_1.stride(3),
        OUT_S0=scaled_out.stride(0),
        OUT_S1=scaled_out.stride(1),
        OUT_S2=scaled_out.stride(2),
        OUT_S3=scaled_out.stride(3),
        GATE_S0=arg2_1.stride(0),
        GATE_S1=arg2_1.stride(1),
        SIG_S0=sigmoid_out.stride(0),
        SIG_S1=sigmoid_out.stride(1),
        ROW_S0=row_out.stride(0),
        ROW_S1=row_out.stride(1),
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=spatial_warps,
        num_stages=4,
    )
    _channel_finalize_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
        row_out,
        final_out,
        N=n,
        C=channels,
        ROW_S0=row_out.stride(0),
        ROW_S1=row_out.stride(1),
        BLOCK_N=triton.next_power_of_2(n),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )

    return scaled_out, sigmoid_out, row_out, final_out
