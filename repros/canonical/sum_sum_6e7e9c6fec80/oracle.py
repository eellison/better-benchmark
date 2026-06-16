"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DenseNet BN-backward tail in one per-channel Triton kernel, sharing the `where` producer and centered input across the sibling `sum(where)` and `sum(where * centered)` reductions, preserving the captured f32 scale constant, writing the full bf16 gradient tensor, and emitting the returned bf16 channel-928:960 residual add with both bf16 add rounding boundaries; whereas Inductor lowers the mask, paired reductions, dependent BN-backward epilogue, full tensor cast, and sliced side-output add as generic scheduled regions around materialized intermediates. Inductor cannot do this today because the reduction scheduler does not form one full-scope multi-output plan whose finalized channel reductions feed both a full-tensor bf16 epilogue and a slice-limited bf16 residual-add consumer. The fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse compatible sibling channel reductions with dependent full and sliced epilogues while preserving low-precision cast/add boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 960
SLICE_START = 928
SLICE_C = 32
CAPTURED_INV_NHW = 0.0012755102040816326


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_tail_kernel(
    residual0_ptr,
    residual1_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    scale_ptr,
    affine_ptr,
    sum_where_ptr,
    scaled_sum_centered_ptr,
    grad_ptr,
    add_out_ptr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    CAPTURED_INV_NHW_: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_SPATIAL: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    active = r < TOTAL_SPATIAL
    n = r // HW
    hw = r - n * HW
    offsets = n * (C_ * HW) + c * HW + hw

    mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)

    where_value = tl.where(mask_input <= 0.0, full_value, source)
    where_active = tl.where(active, where_value, 0.0)
    centered = _sub_rn_f32(centered_source, mean)
    product = _mul_rn_f32(where_value, centered)

    sum_where = tl.sum(where_active, axis=0)
    sum_centered = tl.sum(tl.where(active, product, 0.0), axis=0)
    scale = tl.load(scale_ptr + c).to(tl.float32)
    affine = tl.load(affine_ptr + c).to(tl.float32)

    mean_term = _mul_rn_f32(sum_where, CAPTURED_INV_NHW_)
    centered_scale = _mul_rn_f32(sum_centered, CAPTURED_INV_NHW_)
    scale_sq = _mul_rn_f32(scale, scale)
    variance_term = _mul_rn_f32(centered_scale, scale_sq)
    affine_term = _mul_rn_f32(scale, affine)
    scaled_sum_centered = _mul_rn_f32(sum_centered, scale)

    correction = _mul_rn_f32(centered, variance_term)
    tmp = _sub_rn_f32(where_value, correction)
    tmp = _sub_rn_f32(tmp, mean_term)
    grad = _mul_rn_f32(tmp, affine_term).to(tl.bfloat16)

    tl.store(sum_where_ptr + c, sum_where)
    tl.store(scaled_sum_centered_ptr + c, scaled_sum_centered)
    tl.store(grad_ptr + offsets, grad, mask=active)

    in_slice = c >= SLICE_START_
    slice_c = c - SLICE_START_
    residual0 = tl.load(residual0_ptr + n * (1024 * HW) + c * HW + hw, mask=active & in_slice, other=0.0).to(tl.float32)
    residual1 = tl.load(residual1_ptr + n * (992 * HW) + c * HW + hw, mask=active & in_slice, other=0.0).to(tl.float32)
    first_add = _add_rn_f32(residual0, residual1).to(tl.bfloat16)
    second_add = _add_rn_f32(first_add.to(tl.float32), grad.to(tl.float32)).to(tl.bfloat16)
    add_offsets = n * (SLICE_C_ * HW) + slice_c * HW + hw
    tl.store(add_out_ptr + add_offsets, second_add, mask=active & in_slice)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _launch(inputs, *, BLOCK_R: int, num_warps: int, num_stages: int):
    (
        residual0,
        residual1,
        mask_input,
        full,
        source,
        centered_source,
        mean,
        scale,
        affine,
    ) = inputs
    n = int(mask_input.shape[0])
    h = int(mask_input.shape[2])
    w = int(mask_input.shape[3])
    hw = h * w
    total_spatial = n * hw

    sum_where = torch.empty_strided((C,), (1,), device=mask_input.device, dtype=torch.float32)
    scaled_sum_centered = torch.empty_strided((C,), (1,), device=mask_input.device, dtype=torch.float32)
    grad = torch.empty_strided(
        (n, C, h, w),
        _contiguous_stride((n, C, h, w)),
        device=mask_input.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (n, SLICE_C, h, w),
        _contiguous_stride((n, SLICE_C, h, w)),
        device=mask_input.device,
        dtype=torch.bfloat16,
    )

    _bn_tail_kernel[(C,)](
        residual0,
        residual1,
        mask_input,
        full,
        source,
        centered_source,
        mean,
        scale,
        affine,
        sum_where,
        scaled_sum_centered,
        grad,
        add_out,
        C_=C,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        CAPTURED_INV_NHW_=CAPTURED_INV_NHW,
        HW=hw,
        TOTAL_SPATIAL=total_spatial,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return sum_where, scaled_sum_centered, grad, add_out


# 0b1ad29c: (T([4,1024,14,14], bf16), T([4,992,14,14], bf16), ...)
# 69c44c42: (T([4,1024,7,7], bf16), T([4,992,7,7], bf16), ...)
@oracle_impl(hardware="B200", point="0b1ad29c", BLOCK_R=1024, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="69c44c42", BLOCK_R=256, num_warps=4, num_stages=4)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK_R=BLOCK_R, num_warps=num_warps, num_stages=num_stages)
