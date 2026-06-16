"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 batch-norm-backward fragment by sharing the captured ReLU-mask producer across the two per-channel reductions, preserving the bf16 upstream-gradient divide-by-49 and activation rounding boundaries, using the finalized summaries immediately in the full bf16 `[4,1024,7,7]` epilogue, and returning the last-32-channel slice as an alias of that dense output. Inductor currently schedules the mask producer, sibling reductions, dependent broadcast epilogue, bf16 materialization, and slice view as generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that shares the masked BN-backward producer and sinks the dependent full-output epilogue while preserving the explicit bf16 casts and returned slice alias. The fix is SCHEDULER_FUSION: add a guarded BN-backward multi-output reduction template that keeps compatible channel reductions and dependent epilogues in one plan with exact dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 1024
H = 7
W = 7
HW = H * W
R = N * HW
INV_HW = 0.02040816326530612
INV_COUNT = 0.00510204081632653
SLICE_START = 992


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
def _channel_reduce_epilogue_kernel(
    grad_source_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    full_scalar_ptr,
    sum_where_ptr,
    weight_grad_ptr,
    out_bf16_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    R_: tl.constexpr,
    INV_HW_: tl.constexpr,
    INV_COUNT_: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_

    x_offsets = n * (C_ * HW_) + c * HW_ + spatial
    source_offsets = n * C_ + c

    source = tl.load(grad_source_ptr + source_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    source_div = _f32_mul(source, INV_HW_)
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    where_value = tl.where(active & (affine_bf16 > 0.0), source_div, 0.0)

    product = _f32_mul(where_value, centered)
    sum_where = tl.sum(where_value, axis=0)
    sum_centered = tl.sum(product, axis=0)

    mean_term = _f32_mul(sum_where, INV_COUNT_)
    dot_scaled = _f32_mul(sum_centered, INV_COUNT_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    after_variance = _f32_sub(where_value, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    grad = _f32_mul(after_mean, output_scale)

    tl.store(sum_where_ptr + c, sum_where)
    tl.store(weight_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(
        out_bf16_ptr + x_offsets,
        grad.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )

    if c == 0:
        tl.store(full_scalar_ptr, tl.full((), 0.0, tl.float32).to(tl.bfloat16))


# densenet121 train, expanded [4,1024] upstream grad over [4,1024,7,7].
@oracle_impl(hardware="B200", point="4d18b4bc", BLOCK_R=256, num_warps=2)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    device = arg0_1.device
    full_scalar = torch.empty((), device=device, dtype=torch.bfloat16)
    sum_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    weight_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _channel_reduce_epilogue_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        full_scalar,
        sum_where,
        weight_grad,
        out_bf16,
        C_=C,
        HW_=HW,
        R_=R,
        INV_HW_=INV_HW,
        INV_COUNT_=INV_COUNT,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return full_scalar, sum_where, weight_grad, out_bf16, out_bf16[:, SLICE_START:C, :, :]
