"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BN-backward tail, including the six-input last-32-channel residual add, scalar-fill `where`, both fp32 `[0, 2, 3]` channel reductions, returned scale-gradient vector, full bf16 tensor epilogue, and sliced residual-add output in one channel-resident Triton kernel, whereas Inductor schedules the masked producer, sibling reductions, coefficient math, dense epilogue, and slice-add consumer through generic reduction/pointwise regions; Inductor cannot do this today because scheduler/codegen has no guarded full-scope multi-output reduction plan that keeps each small channel reduction resident while preserving bf16 cast boundaries for the dependent full tensor and residual slice output; the fix is SCHEDULER_FUSION: add a DenseNet BN-backward lowering that co-schedules same-axis channel reductions with their dense and sliced bf16 epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 832
N = 4
SLICE_START = 800
SLICE_C = 32
SCALE = 0.0012755102040816326


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
def _bf16_round_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _densenet_bn_tail_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    mask_ptr,
    fill_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    full_out_ptr,
    slice_out_ptr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK_K)
    active = r < K_TOTAL
    n = r // HW
    hw = r - n * HW
    base = n * (832 * HW) + c * HW + hw

    mask_value = tl.load(mask_ptr + base, mask=active, other=1.0).to(tl.float32)
    rhs = tl.load(where_rhs_ptr + base, mask=active, other=0.0).to(tl.float32)
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
    mean_term = _f32_mul(sum_value, 0.0012755102040816326)
    invstd_sq = _f32_mul(invstd, invstd)
    coeff = _f32_mul(_f32_mul(dot_value, 0.0012755102040816326), invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    correction = _f32_mul(centered, coeff)
    corrected = _f32_sub(_f32_sub(source, correction), mean_term)
    out_f32 = _f32_mul(corrected, output_scale)
    out_bf16_f32 = _bf16_round_f32(out_f32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(full_out_ptr + base, out_bf16_f32, mask=active)

    slice_active = active & (c >= 800)
    offset0 = n * (1024 * HW) + c * HW + hw
    offset1 = n * (992 * HW) + c * HW + hw
    offset2 = n * (960 * HW) + c * HW + hw
    offset3 = n * (928 * HW) + c * HW + hw
    offset4 = n * (896 * HW) + c * HW + hw
    offset5 = n * (864 * HW) + c * HW + hw
    residual0 = tl.load(arg0_ptr + offset0, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual1 = tl.load(arg1_ptr + offset1, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual2 = tl.load(arg2_ptr + offset2, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual3 = tl.load(arg3_ptr + offset3, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual4 = tl.load(arg4_ptr + offset4, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual5 = tl.load(arg5_ptr + offset5, mask=slice_active, other=0.0).to(
        tl.float32
    )
    residual = _bf16_round_f32(_f32_add(residual0, residual1))
    residual = _bf16_round_f32(_f32_add(residual, residual2))
    residual = _bf16_round_f32(_f32_add(residual, residual3))
    residual = _bf16_round_f32(_f32_add(residual, residual4))
    residual = _bf16_round_f32(_f32_add(residual, residual5))
    slice_c = c - 800
    slice_offset = n * (32 * HW) + slice_c * HW + hw
    slice_out = _bf16_round_f32(_f32_add(residual, out_bf16_f32))
    tl.store(slice_out_ptr + slice_offset, slice_out, mask=slice_active)


@oracle_impl(
    hardware="B200",
    point="100b69a7",
    HW=196,
    K_TOTAL=784,
    BLOCK_K=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="3982ddad",
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
    ) = inputs
    h = int(arg6_1.shape[2])
    w = int(arg6_1.shape[3])
    device = arg6_1.device
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    full_out = torch.empty_strided(
        (N, C, h, w),
        (C * HW, HW, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    slice_out = torch.empty_strided(
        (N, SLICE_C, h, w),
        (SLICE_C * HW, HW, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _densenet_bn_tail_kernel[(C,)](
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
        sum_out,
        scaled_dot_out,
        full_out,
        slice_out,
        HW=HW,
        K_TOTAL=K_TOTAL,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scaled_dot_out, full_out, slice_out
