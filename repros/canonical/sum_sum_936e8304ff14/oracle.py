"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BN-backward tail by sharing the masked `where` producer across both f32 channel reductions, writing the returned full bf16 gradient tensor, and forming the returned channels 160:192 residual add with exact sequential bf16 slice-add rounding across ten residual inputs, whereas Inductor schedules the residual slice-add chain, sibling reductions, reduction-dependent BN epilogue, full gradient materialization, and final slice add as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel summaries available to dense and slice-limited consumers while preserving bf16/f32 cast boundaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to share the masked producer and sink finalized channel summaries into both the dense BN-backward store and the static residual-slice epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 192
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 160
SLICE_C = 32
SCALE = 0.00031887755102040814


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_add(a, b):
    return _add_rn(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )


@triton.jit
def _bn_tail_kernel(
    r0_ptr,
    r1_ptr,
    r2_ptr,
    r3_ptr,
    r4_ptr,
    r5_ptr,
    r6_ptr,
    r7_ptr,
    r8_ptr,
    r9_ptr,
    mask_ptr,
    fill_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_out_ptr,
    mul8_out_ptr,
    dense_out_ptr,
    add_out_ptr,
    c_size: tl.constexpr,
    hw: tl.constexpr,
    total_spatial: tl.constexpr,
    slice_start: tl.constexpr,
    slice_c: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < total_spatial
    n = rows // hw
    spatial = rows - n * hw
    dense_offsets = n * (c_size * hw) + c * hw + spatial

    mask_value = tl.load(mask_ptr + dense_offsets, mask=active, other=0.0).to(tl.bfloat16)
    source = tl.load(where_rhs_ptr + dense_offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_src = tl.load(centered_src_ptr + dense_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _sub_rn(centered_src, mean), 0.0)

    product = _mul_rn(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_mul = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _mul_rn(sum_where, scale)
    sum_mul_scaled = _mul_rn(sum_mul, scale)
    invstd_sq = _mul_rn(invstd, invstd)
    variance_term = _mul_rn(sum_mul_scaled, invstd_sq)
    out_weight = _mul_rn(invstd, weight)

    sub1 = _sub_rn(where_f32, _mul_rn(centered, variance_term))
    sub2 = _sub_rn(sub1, mean_term)
    grad = _mul_rn(sub2, out_weight).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(sum_where_out_ptr + c, sum_where)
    tl.store(mul8_out_ptr + c, _mul_rn(sum_mul, invstd))
    tl.store(dense_out_ptr + dense_offsets, grad, mask=active)

    in_slice = c >= slice_start
    slice_idx = c - slice_start
    add_mask = active & in_slice
    add_offsets = n * (slice_c * hw) + slice_idx * hw + spatial

    v0 = tl.load(r0_ptr + n * (512 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v1 = tl.load(r1_ptr + n * (480 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v2 = tl.load(r2_ptr + n * (448 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v3 = tl.load(r3_ptr + n * (416 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v4 = tl.load(r4_ptr + n * (384 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v5 = tl.load(r5_ptr + n * (352 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v6 = tl.load(r6_ptr + n * (320 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v7 = tl.load(r7_ptr + n * (288 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v8 = tl.load(r8_ptr + n * (256 * hw) + c * hw + spatial, mask=add_mask, other=0.0)
    v9 = tl.load(r9_ptr + n * (224 * hw) + c * hw + spatial, mask=add_mask, other=0.0)

    residual = _bf16_add(v0, v1)
    residual = _bf16_add(residual, v2)
    residual = _bf16_add(residual, v3)
    residual = _bf16_add(residual, v4)
    residual = _bf16_add(residual, v5)
    residual = _bf16_add(residual, v6)
    residual = _bf16_add(residual, v7)
    residual = _bf16_add(residual, v8)
    residual = _bf16_add(residual, v9)
    added = _bf16_add(residual, grad)
    tl.store(add_out_ptr + add_offsets, added, mask=add_mask)


@triton.jit
def _bn_dense_kernel(
    mask_ptr,
    fill_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_out_ptr,
    mul8_out_ptr,
    dense_out_ptr,
    c_size: tl.constexpr,
    hw: tl.constexpr,
    total_spatial: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < total_spatial
    n = rows // hw
    spatial = rows - n * hw
    dense_offsets = n * (c_size * hw) + c * hw + spatial

    mask_value = tl.load(mask_ptr + dense_offsets, mask=active, other=0.0).to(tl.bfloat16)
    source = tl.load(where_rhs_ptr + dense_offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_src = tl.load(centered_src_ptr + dense_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _sub_rn(centered_src, mean), 0.0)

    product = _mul_rn(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_mul = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _mul_rn(sum_where, scale)
    variance_term = _mul_rn(_mul_rn(sum_mul, scale), _mul_rn(invstd, invstd))
    out_weight = _mul_rn(invstd, weight)

    sub1 = _sub_rn(where_f32, _mul_rn(centered, variance_term))
    sub2 = _sub_rn(sub1, mean_term)
    grad = _mul_rn(sub2, out_weight).to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(sum_where_out_ptr + c, sum_where)
    tl.store(mul8_out_ptr + c, _mul_rn(sum_mul, invstd))
    tl.store(dense_out_ptr + dense_offsets, grad, mask=active)


@triton.jit
def _slice_add_kernel(
    r0_ptr,
    r1_ptr,
    r2_ptr,
    r3_ptr,
    r4_ptr,
    r5_ptr,
    r6_ptr,
    r7_ptr,
    r8_ptr,
    r9_ptr,
    dense_ptr,
    add_out_ptr,
    numel: tl.constexpr,
    c_size: tl.constexpr,
    hw: tl.constexpr,
    slice_start: tl.constexpr,
    slice_c_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < numel
    spatial = offsets % hw
    slice_c = (offsets // hw) % slice_c_size
    n = offsets // (slice_c_size * hw)
    c = slice_start + slice_c

    v0 = tl.load(r0_ptr + n * (512 * hw) + c * hw + spatial, mask=active, other=0.0)
    v1 = tl.load(r1_ptr + n * (480 * hw) + c * hw + spatial, mask=active, other=0.0)
    v2 = tl.load(r2_ptr + n * (448 * hw) + c * hw + spatial, mask=active, other=0.0)
    v3 = tl.load(r3_ptr + n * (416 * hw) + c * hw + spatial, mask=active, other=0.0)
    v4 = tl.load(r4_ptr + n * (384 * hw) + c * hw + spatial, mask=active, other=0.0)
    v5 = tl.load(r5_ptr + n * (352 * hw) + c * hw + spatial, mask=active, other=0.0)
    v6 = tl.load(r6_ptr + n * (320 * hw) + c * hw + spatial, mask=active, other=0.0)
    v7 = tl.load(r7_ptr + n * (288 * hw) + c * hw + spatial, mask=active, other=0.0)
    v8 = tl.load(r8_ptr + n * (256 * hw) + c * hw + spatial, mask=active, other=0.0)
    v9 = tl.load(r9_ptr + n * (224 * hw) + c * hw + spatial, mask=active, other=0.0)
    grad = tl.load(dense_ptr + n * (c_size * hw) + c * hw + spatial, mask=active, other=0.0)

    residual = _bf16_add(v0, v1)
    residual = _bf16_add(residual, v2)
    residual = _bf16_add(residual, v3)
    residual = _bf16_add(residual, v4)
    residual = _bf16_add(residual, v5)
    residual = _bf16_add(residual, v6)
    residual = _bf16_add(residual, v7)
    residual = _bf16_add(residual, v8)
    residual = _bf16_add(residual, v9)
    tl.store(add_out_ptr + offsets, _bf16_add(residual, grad), mask=active)


# (T([4,512,28,28], bf16), T([4,480,28,28], bf16), T([4,448,28,28], bf16), T([4,416,28,28], bf16), T([4,384,28,28], bf16), T([4,352,28,28], bf16), T([4,320,28,28], bf16), T([4,288,28,28], bf16), T([4,256,28,28], bf16), T([4,224,28,28], bf16), T([4,192,28,28], bf16), T([], bf16), T([4,192,28,28], bf16), T([4,192,28,28], bf16), T([1,192,1,1], f32), T([192], f32), T([192], f32))
@oracle_impl(hardware="B200", point="817f488a", BLOCK_R=4096, num_warps=8, SLICE_BLOCK=256, slice_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_R: int,
    num_warps: int,
    SLICE_BLOCK: int,
    slice_warps: int,
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
        arg16_1,
    ) = inputs

    sum_where = torch.empty_strided((C,), (1,), device=arg10_1.device, dtype=torch.float32)
    mul8 = torch.empty_strided((C,), (1,), device=arg10_1.device, dtype=torch.float32)
    dense = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=arg10_1.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=arg10_1.device,
        dtype=torch.bfloat16,
    )

    _bn_dense_kernel[(C,)](
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        sum_where,
        mul8,
        dense,
        c_size=C,
        hw=HW,
        total_spatial=TOTAL_SPATIAL,
        scale=SCALE,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    _slice_add_kernel[(triton.cdiv(N * SLICE_C * HW, SLICE_BLOCK),)](
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
        dense,
        add_out,
        numel=N * SLICE_C * HW,
        c_size=C,
        hw=HW,
        slice_start=SLICE_START,
        slice_c_size=SLICE_C,
        BLOCK=SLICE_BLOCK,
        num_warps=slice_warps,
        num_stages=3,
    )
    return sum_where, mul8, dense, add_out
