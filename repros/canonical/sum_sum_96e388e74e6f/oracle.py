"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 affine/product reduction scope from Repro.forward, including the channels-last bf16 affine input, the f32 batch-norm affine producer, the explicit bf16 affine cast before the upstream-gradient product, the f32 spatial sum, the bf16 round-trip before the hard-sigmoid gate, the returned f32 gate tensor, the returned bf16 gated tensor, and the final bf16 channel sum converted to f32, whereas Inductor schedules the broadcast producer, spatial reduction, casts, gate, and sibling channel reduction through generic per-shape reduction kernels; Inductor cannot do this today because its scheduler/codegen does not expose this GhostNet batch-norm-affine reduction with visible cast side outputs as a reusable fused template across the 28x28 and 7x7 channel regimes; the fix is SCHEDULER_FUSION: add a guarded affine/product spatial-reduction template that preserves the bf16 rounding boundaries and finalizes the channel accumulator from the visible bf16 side output."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512


@triton.jit
def _spatial_row_gate_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    gate_ptr,
    full_ptr,
    gate_out_ptr,
    gated_bf16_out_ptr,
    C: tl.constexpr,
    N_: tl.constexpr,
    HW: tl.constexpr,
    X_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    xidx = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
    r = tl.arange(0, R_BLOCK)[None, :]
    c = xidx % C
    n = xidx // C
    x_mask = xidx < N_ * C
    r_mask = r < HW
    dense_offsets = n * C * HW + r * C + c
    mask = x_mask & r_mask

    x = tl.load(x_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=x_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=x_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=x_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=x_mask, other=0.0).to(tl.float32)

    affine = ((x - mean) * invstd) * weight + bias
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    product_bf16 = (grad * affine_bf16).to(tl.bfloat16).to(tl.float32)
    spatial = tl.sum(tl.where(mask, product_bf16, 0.0), axis=1)
    spatial_bf16 = spatial.to(tl.bfloat16).to(tl.float32)

    flat = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    flat_mask = flat < N_ * C
    gate = tl.load(gate_ptr + flat, mask=flat_mask, other=0.0).to(tl.float32)
    tl.store(gate_out_ptr + flat, gate, mask=flat_mask)
    active = (gate > -3.0) & (gate < 3.0)
    gated = tl.where(active, spatial_bf16 * 0.16666666666666666, tl.load(full_ptr).to(tl.float32))
    tl.store(gated_bf16_out_ptr + flat, gated.to(tl.bfloat16), mask=flat_mask)


@triton.jit
def _spatial_direct_gate_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    gate_ptr,
    full_ptr,
    gate_out_ptr,
    gated_bf16_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    c_mask = c < C
    hw_mask = hw < HW
    dense_offsets = n * C * HW + hw[:, None] * C + c[None, :]
    nc_offsets = n * C + c
    mask = hw_mask[:, None] & c_mask[None, :]

    x = tl.load(x_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + dense_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    affine = ((x - mean[None, :]) * invstd[None, :]) * weight[None, :] + bias[None, :]
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    product_bf16 = (grad * affine_bf16).to(tl.bfloat16).to(tl.float32)
    spatial = tl.sum(tl.where(hw_mask[:, None], product_bf16, 0.0), axis=0)
    spatial_bf16 = spatial.to(tl.bfloat16).to(tl.float32)

    gate = tl.load(gate_ptr + nc_offsets, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(gate_out_ptr + nc_offsets, gate, mask=c_mask)
    active = (gate > -3.0) & (gate < 3.0)
    gated = tl.where(active, spatial_bf16 * 0.16666666666666666, tl.load(full_ptr).to(tl.float32))
    tl.store(gated_bf16_out_ptr + nc_offsets, gated.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _channel_sum_finalize_kernel(
    gated_bf16_out_ptr,
    channel_out_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    c_mask = c < C
    offsets = n[:, None] * C + c[None, :]
    values = tl.load(
        gated_bf16_out_ptr + offsets,
        mask=c_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(channel_out_ptr + c, total, mask=c_mask)


@oracle_impl(
    hardware="B200",
    point="0e39883f",
    C=72,
    HW=784,
    USE_ROW=True,
    BLOCK_X=32,
    BLOCK_R=1024,
    FINAL_BLOCK_C=8,
    num_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="3bcfd222",
    C=672,
    HW=49,
    USE_ROW=True,
    BLOCK_X=32,
    BLOCK_R=64,
    FINAL_BLOCK_C=16,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    C: int,
    HW: int,
    USE_ROW: bool,
    BLOCK_X: int,
    BLOCK_R: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    x, mean, invstd, weight, bias, grad, gate_bf16, full = inputs

    gate_out = torch.empty_strided(
        (N, C, 1, 1),
        (C, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )
    gated_bf16 = torch.empty_strided(
        (N, C, 1, 1),
        (C, 1, 1, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    channel_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    if USE_ROW:
        _spatial_row_gate_kernel[(triton.cdiv(N * C, BLOCK_X),)](
            x,
            mean,
            invstd,
            weight,
            bias,
            grad,
            gate_bf16,
            full,
            gate_out,
            gated_bf16,
            C=C,
            N_=N,
            HW=HW,
            X_BLOCK=BLOCK_X,
            R_BLOCK=BLOCK_R,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _spatial_direct_gate_kernel[(N, triton.cdiv(C, BLOCK_X))](
            x,
            mean,
            invstd,
            weight,
            bias,
            grad,
            gate_bf16,
            full,
            gate_out,
            gated_bf16,
            C=C,
            HW=HW,
            BLOCK_C=BLOCK_X,
            BLOCK_HW=BLOCK_R,
            num_warps=num_warps,
            num_stages=1,
        )
    _channel_sum_finalize_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        gated_bf16,
        channel_out,
        C=C,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=8,
        num_stages=1,
    )
    return gate_out, gated_bf16, channel_out
