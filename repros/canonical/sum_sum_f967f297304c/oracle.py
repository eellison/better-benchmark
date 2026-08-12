"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail by sharing the masked bf16 `where` producer across both per-channel reductions, writing the returned raw sum vector, scale-gradient vector, full bf16 dense gradient tensor, and preserving the ordered nine-input last-32-channel residual slice add; Inductor lowers the residual slice chain, masked producer, sibling reductions, dense dependent epilogue, and final slice add as generic reduction/pointwise regions around materialized or replayed intermediates; Inductor cannot do this today because scheduler/codegen has no guarded full-scope multi-output reduction plan that keeps compatible channel summaries, explicit bf16 rounding boundaries, a dependent dense epilogue, and a slice-limited DenseNet residual fan-in in one fused plan; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse masked BN-backward channel reductions with dependent dense and residual-slice epilogues while preserving captured dtype casts and output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 736
N = 4
SLICE_START = 704
SLICE_C = 32
SCALE = 0.0012755102040816326


@triton.jit
def _bf16_round_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _densenet_tail_kernel(
    residual0_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    dense_out_ptr,
    tail_out_ptr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK_K)
    active = r < K_TOTAL
    n = r // HW
    hw = r - n * HW
    base = n * (736 * HW) + c * HW + hw

    mask_value = tl.load(mask_ptr + base, mask=active, other=1.0).to(tl.float32)
    rhs = tl.load(source_ptr + base, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    source = tl.where(mask_value <= 0.0, fill, rhs).to(tl.float32)

    centered_src = tl.load(centered_src_ptr + base, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(centered_src, mean)

    source_for_sum = tl.where(active, source, 0.0)
    centered_for_sum = tl.where(active, centered, 0.0)
    sum_value = tl.sum(source_for_sum, axis=0)
    dot_value = tl.sum(_f32_mul(source_for_sum, centered_for_sum), axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    coeff = _f32_mul(_f32_mul(dot_value, SCALE_VALUE), invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    correction = _f32_mul(centered, coeff)
    corrected = _f32_sub(_f32_sub(source, correction), mean_term)
    dense_bf16 = _bf16_round_f32(_f32_mul(corrected, output_scale))

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(dense_out_ptr + base, dense_bf16, mask=active)

    tail_active = active & (c >= 704)
    tail_c = c - 704
    tail_offset = n * (32 * HW) + tail_c * HW + hw

    offset0 = n * (1024 * HW) + c * HW + hw
    offset1 = n * (992 * HW) + c * HW + hw
    offset2 = n * (960 * HW) + c * HW + hw
    offset3 = n * (928 * HW) + c * HW + hw
    offset4 = n * (896 * HW) + c * HW + hw
    offset5 = n * (864 * HW) + c * HW + hw
    offset6 = n * (832 * HW) + c * HW + hw
    offset7 = n * (800 * HW) + c * HW + hw
    offset8 = n * (768 * HW) + c * HW + hw

    residual = tl.load(residual0_ptr + offset0, mask=tail_active, other=0.0).to(
        tl.float32
    )
    next_value = tl.load(residual1_ptr + offset1, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual2_ptr + offset2, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual3_ptr + offset3, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual4_ptr + offset4, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual5_ptr + offset5, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual6_ptr + offset6, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual7_ptr + offset7, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    next_value = tl.load(residual8_ptr + offset8, mask=tail_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual, next_value))
    tail_value = _bf16_round_f32(_f32_add(residual, dense_bf16))
    tl.store(tail_out_ptr + tail_offset, tail_value, mask=tail_active)


@oracle_impl(
    hardware="B200",
    point="1bdfd7ec",
    HW=196,
    K_TOTAL=784,
    BLOCK_K=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="747f775b",
    HW=49,
    K_TOTAL=196,
    BLOCK_K=256,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    HW: int,
    K_TOTAL: int,
    BLOCK_K: int,
    num_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
    ) = inputs
    height = int(arg9_1.shape[2])
    width = int(arg9_1.shape[3])
    hw = height * width
    device = arg9_1.device

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, height, width),
        (C * hw, hw, width, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    tail_out = torch.empty_strided(
        (N, SLICE_C, height, width),
        (SLICE_C * hw, hw, width, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _densenet_tail_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        sum_out,
        scaled_dot_out,
        dense_out,
        tail_out,
        HW=HW,
        K_TOTAL=K_TOTAL,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scaled_dot_out, dense_out, tail_out
