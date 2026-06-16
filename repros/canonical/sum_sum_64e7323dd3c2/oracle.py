"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BN-backward tail by sharing the masked bf16 `where` producer across both returned f32 channel reductions, writing the returned full bf16 gradient tensor, and forming the returned last-32-channel residual add with the required sequential bf16 slice-add rounding across four residual inputs. Inductor currently schedules the residual slice-add chain, sibling reductions, reduction-dependent BN epilogue, full gradient materialization, and final slice add as separate generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel summaries available to dense and slice-limited consumers while preserving bf16 cast boundaries. The fix is SCHEDULER_FUSION: teach reduction scheduling to share the masked producer and sink finalized channel summaries into both the dense BN-backward store and the static residual-slice epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _dense_kernel(
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    R: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    offsets = n * (C * HW) + c * HW + spatial

    mask_value = tl.load(mask_input_ptr + offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source_value = tl.load(source_ptr + offsets, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(
        centered_source_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, SCALE)
    dot_scaled = _f32_mul(sum_centered, SCALE)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    dense_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)


@triton.jit
def _slice_add_kernel(
    r0_ptr,
    r1_ptr,
    r2_ptr,
    r3_ptr,
    dense_out_ptr,
    add_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    R: tl.constexpr,
    SLICE_START: tl.constexpr,
    SLICE_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    slice_c = tl.program_id(0)
    c = SLICE_START + slice_c
    rows = tl.arange(0, BLOCK_R)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW

    dense_offsets = n * (C * HW) + c * HW + spatial
    add_offsets = n * (SLICE_C * HW) + slice_c * HW + spatial
    dense_bf16 = tl.load(dense_out_ptr + dense_offsets, mask=active, other=0.0)

    v0 = tl.load(r0_ptr + n * (512 * HW) + c * HW + spatial, mask=active, other=0.0)
    v1 = tl.load(r1_ptr + n * (480 * HW) + c * HW + spatial, mask=active, other=0.0)
    v2 = tl.load(r2_ptr + n * (448 * HW) + c * HW + spatial, mask=active, other=0.0)
    v3 = tl.load(r3_ptr + n * (416 * HW) + c * HW + spatial, mask=active, other=0.0)

    residual = _bf16_add(v0, v1)
    residual = _bf16_add(residual, v2)
    residual = _bf16_add(residual, v3)
    add_value = _bf16_add(residual, dense_bf16)
    tl.store(add_out_ptr + add_offsets, add_value, mask=active)


# 342bbb54: densenet121 train, N=4 C=384 H=W=28, residual slice channels 352:384.
@oracle_impl(hardware="B200", point="342bbb54", BLOCK_R=4096, num_warps=8, SIDE_WARPS=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int, SIDE_WARPS: int):
    (
        r0,
        r1,
        r2,
        r3,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
    ) = inputs

    c = 384
    hw = 28 * 28
    r = 4 * hw
    slice_start = 352
    slice_c = 32
    scale = 0.00031887755102040814

    sum_out = torch.empty_strided((c,), (1,), device=source.device, dtype=torch.float32)
    scale_grad = torch.empty_strided((c,), (1,), device=source.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (4, c, 28, 28),
        (c * hw, hw, 28, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (4, slice_c, 28, 28),
        (slice_c * hw, hw, 28, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )

    _dense_kernel[(c,)](
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
        sum_out,
        scale_grad,
        dense_out,
        C=c,
        HW=hw,
        R=r,
        SCALE=scale,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    _slice_add_kernel[(slice_c,)](
        r0,
        r1,
        r2,
        r3,
        dense_out,
        add_out,
        C=c,
        HW=hw,
        R=r,
        SLICE_START=slice_start,
        SLICE_C=slice_c,
        BLOCK_R=BLOCK_R,
        num_warps=SIDE_WARPS,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, add_out
