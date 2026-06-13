"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 batch-norm-backward tail by sharing the captured ReLU-mask producer across the two per-channel reductions, preserving the bf16 `where` inputs and BN-backward arithmetic order, and using the finalized summaries immediately in the residual-add bf16 `[128,120,4,4]` epilogue whose first-16-channel slice is returned as an alias; Inductor currently schedules the mask producer, sibling reductions, dependent broadcast epilogue, bf16 residual add, and slice return as generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that shares the masked BN-backward producer and sinks the dependent full-output epilogue while preserving explicit bf16 casts and output aliasing; the fix is SCHEDULER_FUSION: add a guarded BN-backward multi-output reduction template that keeps compatible channel reductions and dependent epilogues in one plan with exact dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 120
H = 4
W = 4
HW = H * W
INV_COUNT = 0.00048828125


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
def _channel_reduce_residual_epilogue_kernel(
    residual_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_base_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_where_ptr,
    weight_grad_ptr,
    out_bf16_ptr,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)[:, None]
    hw = tl.arange(0, BLOCK_HW)[None, :]

    offsets = n * (120 * 16) + c * 16 + hw
    residual_offsets = n * (136 * 16) + (c + 16) * 16 + hw

    mask_input = tl.load(mask_input_ptr + offsets).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + offsets).to(tl.float32)
    centered_base = tl.load(centered_base_ptr + offsets).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, source)
    centered = _f32_sub(centered_base, mean)

    sum_where = tl.sum(tl.sum(where_value, axis=1), axis=0)
    sum_centered = tl.sum(tl.sum(_f32_mul(where_value, centered), axis=1), axis=0)

    mean_term = _f32_mul(sum_where, 0.00048828125)
    scaled_centered_sum = _f32_mul(sum_centered, 0.00048828125)
    variance_term = _f32_mul(scaled_centered_sum, _f32_mul(invstd, invstd))
    post_scale = _f32_mul(invstd, affine_weight)
    grad = _f32_mul(
        _f32_sub(_f32_sub(where_value, _f32_mul(centered, variance_term)), mean_term),
        post_scale,
    )
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + residual_offsets).to(tl.float32)
    out = _f32_add(residual, grad_bf16.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(sum_where_ptr + c, sum_where)
    tl.store(weight_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(out_bf16_ptr + offsets, out)


# (T([128,136,4,4], bf16), T([128,120,4,4], bf16), T([], bf16), T([128,120,4,4], bf16), T([128,120,4,4], bf16), T([1,120,1,1], f32), T([120], f32), T([120], f32))
@oracle_impl(hardware="B200", point="4055c4c5", BLOCK_N=128, BLOCK_HW=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs

    device = arg0_1.device
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    weight_grad = torch.empty((C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _channel_reduce_residual_epilogue_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        sum_where,
        weight_grad,
        out_bf16,
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )

    return sum_where, weight_grad, out_bf16, out_bf16[:, :16, :, :]
