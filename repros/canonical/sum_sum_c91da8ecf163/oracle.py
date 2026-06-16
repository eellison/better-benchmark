"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 EfficientNet BatchNorm-affine SiLU gate-backward fragment, including the bf16-rounded affine f32 output, returned bf16 sigmoid, returned bf16 per-(N,C) gate-gradient tensor, and final bf16 channel sum promoted to f32, whereas Inductor schedules the shared affine/SiLU producer, spatial sum, sigmoid-gradient epilogue, and dependent channel sum as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned multi-output plan that preserves the explicit bf16 cast boundaries while sharing a channels-last spatial producer with a rank-changing dependent reduction; the fix is SCHEDULER_FUSION: add a BN-affine SiLU spatial-reduction template that emits sibling materialized outputs and finalizes the dependent bf16 channel reduction without re-reading the full producer."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


FINAL_BLOCK_N = 128


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
def _spatial_affine_silu_gate_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    grad_ptr,
    gate_ptr,
    affine_out_ptr,
    sigmoid_out_ptr,
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
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.program_id(1)
    hw = tl.arange(0, BLOCK_HW)
    h = hw // W
    w = hw - h * W
    hw_mask = hw < H * W
    c_mask = c < C
    if TRANSPOSED:
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
    else:
        mask = c_mask[:, None] & hw_mask[None, :]

        x_offsets = (
            n * X_STRIDE_N
            + c[:, None] * X_STRIDE_C
            + h[None, :] * X_STRIDE_H
            + w[None, :] * X_STRIDE_W
        )
        grad_offsets = (
            n * G_STRIDE_N
            + c[:, None] * G_STRIDE_C
            + h[None, :] * G_STRIDE_H
            + w[None, :] * G_STRIDE_W
        )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    if TRANSPOSED:
        centered = _f32_sub(x, mean[None, :])
        normalized = _f32_mul(centered, invstd[None, :])
        scaled = _f32_mul(normalized, weight[None, :])
        affine = _f32_add(scaled, bias[None, :])
    else:
        centered = _f32_sub(x, mean[:, None])
        normalized = _f32_mul(centered, invstd[:, None])
        scaled = _f32_mul(normalized, weight[:, None])
        affine = _f32_add(scaled, bias[:, None])
    affine_bf16 = affine.to(tl.bfloat16)
    affine_f32 = affine_bf16.to(tl.float32)
    tl.store(affine_out_ptr + x_offsets, affine_f32, mask=mask)

    exp_neg = libdevice.exp(_f32_sub(0.0, affine_f32))
    silu = _f32_div(affine_f32, _f32_add(exp_neg, 1.0))
    silu_bf16 = silu.to(tl.bfloat16)
    product_bf16 = _f32_mul(grad, silu_bf16.to(tl.float32)).to(tl.bfloat16)
    if TRANSPOSED:
        spatial_sum = tl.sum(tl.where(mask, product_bf16.to(tl.float32), 0.0), axis=0)
    else:
        spatial_sum = tl.sum(tl.where(mask, product_bf16.to(tl.float32), 0.0), axis=1)
    spatial_sum_f32 = spatial_sum.to(tl.bfloat16).to(tl.float32)

    gate_offsets = n * C + c
    gate = tl.load(gate_ptr + gate_offsets, mask=c_mask, other=0.0).to(tl.float32)
    gate_sigmoid = _f32_div(1.0, _f32_add(libdevice.exp(_f32_sub(0.0, gate)), 1.0))
    gate_sigmoid_bf16 = gate_sigmoid.to(tl.bfloat16)
    gate_sigmoid_f32 = gate_sigmoid_bf16.to(tl.float32)
    one_minus = _f32_sub(1.0, gate_sigmoid_f32)
    sigmoid_grad = _f32_mul(gate_sigmoid_f32, one_minus)
    gate_grad = _f32_mul(spatial_sum_f32, sigmoid_grad).to(tl.bfloat16)

    tl.store(sigmoid_out_ptr + gate_offsets, gate_sigmoid_bf16, mask=c_mask)
    tl.store(gate_grad_out_ptr + gate_offsets, gate_grad, mask=c_mask)


@triton.jit
def _final_channel_sum_kernel(
    gate_grad_ptr,
    out_ptr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    mask = c[None, :] < C
    offsets = n[:, None] * C + c[None, :]
    vals = tl.load(gate_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    reduced = tl.sum(vals, axis=0).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + c, reduced, mask=c < C)


@oracle_impl(hardware="B200", point="eeebf109", BLOCK_C=2, BLOCK_HW=16384, FINAL_BLOCK_C=32, TRANSPOSED=True, num_warps=8)
@oracle_impl(hardware="B200", point="e7f464f6", BLOCK_C=4, BLOCK_HW=4096, FINAL_BLOCK_C=32, TRANSPOSED=False, num_warps=8)
@oracle_impl(hardware="B200", point="8b89e909", BLOCK_C=4, BLOCK_HW=4096, FINAL_BLOCK_C=32, TRANSPOSED=False, num_warps=8)
@oracle_impl(hardware="B200", point="bb3aa93a", BLOCK_C=8, BLOCK_HW=1024, FINAL_BLOCK_C=32, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="e177ef6b", BLOCK_C=8, BLOCK_HW=1024, FINAL_BLOCK_C=32, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="693f7c58", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=32, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="d58930d9", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=64, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="b8f8666d", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=64, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="dae1d1dd", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64, TRANSPOSED=False, num_warps=4)
@oracle_impl(hardware="B200", point="b90cb54e", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64, TRANSPOSED=False, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, FINAL_BLOCK_C, TRANSPOSED, num_warps):
    x, mean, invstd, weight, bias, grad, gate = inputs
    n, c, h, w = (int(dim) for dim in x.shape)

    affine_out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    sigmoid_out = torch.empty_strided(
        tuple(gate.shape),
        tuple(gate.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    gate_grad_out = torch.empty_strided(
        tuple(gate.shape),
        tuple(gate.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    channel_out = torch.empty((c,), device=x.device, dtype=torch.float32)

    _spatial_affine_silu_gate_kernel[(triton.cdiv(c, BLOCK_C), n)](
        x,
        mean,
        invstd,
        weight,
        bias,
        grad,
        gate,
        affine_out,
        sigmoid_out,
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
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        TRANSPOSED=TRANSPOSED,
        num_warps=num_warps,
    )
    _final_channel_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        gate_grad_out,
        channel_out,
        C=c,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=4,
    )
    return affine_out, sigmoid_out, gate_grad_out, channel_out
