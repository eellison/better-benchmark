"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileNetV3 BatchNorm-affine h-swish gate-backward fragment, including the affine bf16 round-trip returned as f32, the returned bf16 gate tensor promoted to f32, the bf16-rounded spatial reduction gated by the captured hard-swish derivative predicate, and the final bf16 channel sum promoted to f32, whereas Inductor schedules the shared affine/h-swish producer, spatial sum, predicate epilogue, and dependent batch-channel sum as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned multi-output plan that preserves explicit bf16 cast boundaries while sharing a channels-last spatial producer with a rank-changing dependent reduction; the fix is SCHEDULER_FUSION: add a MobileNetV3 h-swish spatial-reduction template that emits sibling materialized outputs and finalizes the dependent bf16 channel reduction without re-reading the full producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


FINAL_BLOCK_N = 512


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _spatial_hswish_gate_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    gate_ptr,
    scalar_ptr,
    affine_out_ptr,
    gate_f32_out_ptr,
    gate_grad_out_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    G_STRIDE_N: tl.constexpr,
    G_STRIDE_C: tl.constexpr,
    G_STRIDE_H: tl.constexpr,
    G_STRIDE_W: tl.constexpr,
    GATE_STRIDE_N: tl.constexpr,
    GATE_STRIDE_C: tl.constexpr,
    OUT_GATE_STRIDE_N: tl.constexpr,
    OUT_GATE_STRIDE_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW)
    h = hw // W
    w = hw - h * W
    c_mask = c < C
    hw_mask = hw < H * W
    mask = hw_mask[:, None] & c_mask[None, :]

    x_offsets = (
        n * X_STRIDE_N
        + h[:, None] * X_STRIDE_H
        + w[:, None] * X_STRIDE_W
        + c[None, :] * X_STRIDE_C
    )
    grad_offsets = (
        n * G_STRIDE_N
        + h[:, None] * G_STRIDE_H
        + w[:, None] * G_STRIDE_W
        + c[None, :] * G_STRIDE_C
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    scaled = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    affine_f32 = affine.to(tl.bfloat16).to(tl.float32)
    tl.store(affine_out_ptr + x_offsets, affine_f32, mask=mask)

    hswish_arg = _f32_add(affine_f32, 3.0)
    hswish_min = tl.where(hswish_arg < 0.0, 0.0, hswish_arg)
    hswish_clamped = tl.where(hswish_min > 6.0, 6.0, hswish_min)
    hswish_mul = _f32_mul(affine_f32, hswish_clamped)
    hswish_bf16 = _f32_div(hswish_mul, 6.0).to(tl.bfloat16)
    product_bf16 = _f32_mul(grad, hswish_bf16.to(tl.float32)).to(tl.bfloat16)
    spatial_sum = tl.sum(tl.where(mask, product_bf16.to(tl.float32), 0.0), axis=0)
    spatial_sum_f32 = spatial_sum.to(tl.bfloat16).to(tl.float32)

    gate_offsets = n * GATE_STRIDE_N + c * GATE_STRIDE_C
    out_gate_offsets = n * OUT_GATE_STRIDE_N + c * OUT_GATE_STRIDE_C
    gate_f32 = tl.load(gate_ptr + gate_offsets, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(gate_f32_out_ptr + out_gate_offsets, gate_f32, mask=c_mask)

    active = (gate_f32 > -3.0) & (gate_f32 < 3.0)
    scaled_sum = _f32_mul(spatial_sum_f32, 0.16666666666666666)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    gate_grad = tl.where(active, scaled_sum, scalar).to(tl.bfloat16)
    tl.store(gate_grad_out_ptr + out_gate_offsets, gate_grad, mask=c_mask)


@triton.jit
def _final_channel_sum_kernel(
    gate_grad_ptr,
    out_ptr,
    C: tl.constexpr,
    GATE_STRIDE_N: tl.constexpr,
    GATE_STRIDE_C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    c_mask = c < C
    offsets = n[:, None] * GATE_STRIDE_N + c[None, :] * GATE_STRIDE_C
    vals = tl.load(gate_grad_ptr + offsets, mask=c_mask[None, :], other=0.0).to(tl.float32)
    reduced = tl.sum(vals, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + c, reduced, mask=c_mask)


@oracle_impl(hardware="B200", point="2d9d7e80", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=32, num_warps=4)
@oracle_impl(hardware="B200", point="ffea7dda", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=32, num_warps=4)
@oracle_impl(hardware="B200", point="3bcfd222", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64, num_warps=4)
@oracle_impl(hardware="B200", point="b1ab7efd", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, FINAL_BLOCK_C, num_warps):
    x, mean, invstd, weight, bias, grad, gate, scalar = inputs
    n, c, h, w = (int(dim) for dim in x.shape)

    affine_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    gate_f32_out = torch.empty_strided(
        tuple(gate.shape),
        tuple(gate.stride()),
        device=gate.device,
        dtype=torch.float32,
    )
    gate_grad_out = torch.empty_strided(
        tuple(gate.shape),
        tuple(gate.stride()),
        device=gate.device,
        dtype=torch.bfloat16,
    )
    channel_out = torch.empty((c,), device=x.device, dtype=torch.float32)

    _spatial_hswish_gate_kernel[(triton.cdiv(c, BLOCK_C), n)](
        x,
        mean,
        invstd,
        weight,
        bias,
        grad,
        gate,
        scalar,
        affine_out,
        gate_f32_out,
        gate_grad_out,
        C=c,
        H=h,
        W=w,
        X_STRIDE_N=x.stride(0),
        X_STRIDE_C=x.stride(1),
        X_STRIDE_H=x.stride(2),
        X_STRIDE_W=x.stride(3),
        G_STRIDE_N=grad.stride(0),
        G_STRIDE_C=grad.stride(1),
        G_STRIDE_H=grad.stride(2),
        G_STRIDE_W=grad.stride(3),
        GATE_STRIDE_N=gate.stride(0),
        GATE_STRIDE_C=gate.stride(1),
        OUT_GATE_STRIDE_N=gate_f32_out.stride(0),
        OUT_GATE_STRIDE_C=gate_f32_out.stride(1),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=4,
    )
    _final_channel_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        gate_grad_out,
        channel_out,
        C=c,
        GATE_STRIDE_N=gate_grad_out.stride(0),
        GATE_STRIDE_C=gate_grad_out.stride(1),
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return affine_out, gate_f32_out, gate_grad_out, channel_out
