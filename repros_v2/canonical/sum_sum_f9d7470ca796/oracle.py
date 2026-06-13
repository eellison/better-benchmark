"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 batch-norm-backward fragment by sharing the captured ReLU-mask producer across the two per-channel reductions, preserving the bf16 upstream-gradient division and activation rounding boundaries, and using the finalized summaries immediately in the full bf16 `[128,184,4,4]` epilogue whose first-16-channel slice is returned as an alias; Inductor currently schedules the mask producer, sibling reductions, dependent broadcast epilogue, bf16 materialization, and slice as generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that shares the masked BN-backward producer and sinks the dependent full-output epilogue while preserving the explicit bf16 casts and returned slice alias; the fix is SCHEDULER_FUSION: add a guarded BN-backward multi-output reduction template that keeps compatible channel reductions and dependent epilogues in one plan with exact dtype boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 184
H = 4
W = 4
HW = H * W
TOTAL_SPATIAL = N * HW
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
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)[:, None]
    hw = tl.arange(0, BLOCK_HW)[None, :]

    x_offsets = n * (184 * 16) + c * 16 + hw
    source_offsets = n * 184 + c

    source = tl.load(grad_source_ptr + source_offsets).to(tl.float32)
    source_div = _f32_mul(source, 0.0625).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )
    x = tl.load(x_ptr + x_offsets).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    where_value = tl.where(affine_bf16 <= 0.0, 0.0, source_div)

    sum_where = tl.sum(tl.sum(where_value, axis=1), axis=0)
    sum_centered = tl.sum(tl.sum(_f32_mul(where_value, centered), axis=1), axis=0)

    mean_term = _f32_mul(sum_where, 0.00048828125)
    centered_scale = _f32_mul(
        _f32_mul(sum_centered, 0.00048828125),
        _f32_mul(invstd, invstd),
    )
    output_scale = _f32_mul(invstd, weight)
    grad = _f32_mul(
        _f32_sub(_f32_sub(where_value, _f32_mul(centered, centered_scale)), mean_term),
        output_scale,
    )

    tl.store(sum_where_ptr + c, sum_where)
    tl.store(weight_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(out_bf16_ptr + x_offsets, grad.to(tl.bfloat16, fp_downcast_rounding="rtne"))

    if c == 0:
        tl.store(full_scalar_ptr, tl.full((), 0.0, tl.float32).to(tl.bfloat16))


# (T([128,184], bf16), T([128,184,4,4], bf16), T([1,184,1,1], f32), T([1,184,1,1], f32), T([184], f32), T([184], f32), S([128,184,1,1]), S([128,184,4,4]))
@oracle_impl(hardware="B200", point="e3c25c9d", BLOCK_N=128, BLOCK_HW=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    device = arg0_1.device
    full_scalar = torch.empty((), device=device, dtype=torch.bfloat16)
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    weight_grad = torch.empty((C,), device=device, dtype=torch.float32)
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
        BLOCK_N=BLOCK_N,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )

    return full_scalar, sum_where, weight_grad, out_bf16, out_bf16[:, :16, :, :]
