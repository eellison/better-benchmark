"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail by sharing the bf16 `where` producer across both per-channel reductions, reusing the finalized channel summaries to emit the returned raw sum vector, scale-gradient vector, and full dense bf16 gradient tensor, then preserving the ordered bf16 residual-slice add chain for the returned `[4,32,H,W]` tail output; whereas Inductor lowers the sliced residual chain, masked producer, sibling reductions, dense dependent epilogue, and final slice add as generic reduction/pointwise regions. Inductor cannot do this today because its scheduler/codegen lacks a full-scope channel-reduction template that keeps compatible reductions, explicit bf16 rounding boundaries, a dependent dense epilogue, and a slice-limited DenseNet residual fan-in in one fused plan; the fix is SCHEDULER_FUSION: teach the reduction scheduler to fuse masked BN-backward channel reductions with dependent dense and residual-slice epilogues while preserving captured dtype casts and output strides."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 768
N = 4
SLICE_START = 736
SLICE_C = 32
SCALE = 0.0012755102040816326


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fused_channel_kernel(
    residual0_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    dense_out_ptr,
    tail_out_ptr,
    HW: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    total_spatial = 4 * HW
    active = k < total_spatial
    n = k // HW
    hw = k - n * HW
    offsets = n * (768 * HW) + c * HW + hw

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)

    where_value = tl.where(mask_value <= 0.0, fill, source)
    where_active = tl.where(active, where_value, 0.0)
    centered = _sub_f32(centered_source, mean)
    product = _mul_f32(where_value, centered)
    product = tl.where(active, product, 0.0)
    sum1 = tl.sum(where_active, axis=0)
    sum2 = tl.sum(product, axis=0)

    tl.store(sum_out_ptr + c, sum1)
    tl.store(vec_out_ptr + c, _mul_f32(sum2, invstd))

    mean_term = _mul_f32(sum1, SCALE_VALUE)
    sum2_scaled = _mul_f32(sum2, SCALE_VALUE)
    invstd_sq = _mul_f32(invstd, invstd)
    variance_term = _mul_f32(sum2_scaled, invstd_sq)
    affine = _mul_f32(invstd, affine_weight)
    centered_scaled = _mul_f32(centered, variance_term)
    sub1 = _sub_f32(where_value, centered_scaled)
    sub2 = _sub_f32(sub1, mean_term)
    dense_value = _mul_f32(sub2, affine)
    dense_bf16 = _round_bf16_to_f32(dense_value)
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)

    tail = c >= 736
    tail_active = active & tail
    tail_c = c - 736
    tail_offsets = n * (32 * HW) + tail_c * HW + hw

    residual_offsets0 = n * (1024 * HW) + c * HW + hw
    residual_offsets1 = n * (992 * HW) + c * HW + hw
    residual_offsets2 = n * (960 * HW) + c * HW + hw
    residual_offsets3 = n * (928 * HW) + c * HW + hw
    residual_offsets4 = n * (896 * HW) + c * HW + hw
    residual_offsets5 = n * (864 * HW) + c * HW + hw
    residual_offsets6 = n * (832 * HW) + c * HW + hw
    residual_offsets7 = n * (800 * HW) + c * HW + hw

    residual = tl.load(residual0_ptr + residual_offsets0, mask=tail_active, other=0.0).to(tl.float32)
    next_value = tl.load(residual1_ptr + residual_offsets1, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual2_ptr + residual_offsets2, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual3_ptr + residual_offsets3, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual4_ptr + residual_offsets4, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual5_ptr + residual_offsets5, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual6_ptr + residual_offsets6, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    next_value = tl.load(residual7_ptr + residual_offsets7, mask=tail_active, other=0.0).to(tl.float32)
    residual = _round_bf16_to_f32(_add_f32(residual, next_value))
    tail_value = _round_bf16_to_f32(_add_f32(residual, dense_bf16))
    tl.store(tail_out_ptr + tail_offsets, tail_value, mask=tail_active)


@oracle_impl(hardware="B200", point="50808975", BLOCK_K=1024, num_warps=8)
@oracle_impl(hardware="B200", point="13ede9b8", BLOCK_K=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    (
        residual0,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        mask,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        affine_weight,
    ) = inputs
    height = int(mask.shape[2])
    width = int(mask.shape[3])
    hw = height * width

    sum_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, height, width),
        (C * hw, hw, width, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )
    tail_out = torch.empty_strided(
        (N, SLICE_C, height, width),
        (SLICE_C * hw, hw, width, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )

    _fused_channel_kernel[(C,)](
        residual0,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        mask,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        affine_weight,
        sum_out,
        vec_out,
        dense_out,
        tail_out,
        HW=hw,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return sum_out, vec_out, dense_out, tail_out
