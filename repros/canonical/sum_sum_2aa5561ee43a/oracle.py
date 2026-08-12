"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BN-backward tail with one channel-resident Triton program per output channel, sharing the masked bf16 `where(arg2 <= 0, arg3, arg4)` producer across both returned f32 channel reductions, writing the returned full bf16 gradient tensor, and forming the returned channels 416:448 residual add with the required two sequential bf16 add roundings. Inductor currently schedules the sibling reductions, reduction-dependent dense BN epilogue, full dense materialization, and final sliced add as separate generic regions around replayed or materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps the channel summaries available to dense and slice-limited consumers while preserving bf16/f32 cast boundaries. The fix is SCHEDULER_FUSION: teach reduction scheduling to share the conditional producer and sink the vector, dense, and static slice-add outputs into one coordinated full-scope plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 448
H = 28
W = 28
HW = H * W
R = N * HW
SLICE_START = 416
SLICE_C = C - SLICE_START
RESIDUAL0_C = 512
RESIDUAL1_C = 480
SCALE = 0.00031887755102040814


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
def _bn_tail_kernel(
    residual0_ptr,
    residual1_ptr,
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
    add_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    R_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    RESIDUAL0_C_: tl.constexpr,
    RESIDUAL1_C_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R_
    n = rows // HW_
    spatial = rows - n * HW_
    dense_offsets = n * (C_ * HW_) + c * HW_ + spatial

    mask_value = tl.load(mask_input_ptr + dense_offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source_value = tl.load(source_ptr + dense_offsets, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(
        centered_source_ptr + dense_offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, SCALE_)
    dot_scaled = _f32_mul(sum_centered, SCALE_)
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
    tl.store(dense_out_ptr + dense_offsets, dense_bf16, mask=active)

    in_slice = c >= SLICE_START_
    slice_c = c - SLICE_START_
    add_mask = active & in_slice
    add_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + spatial
    residual0 = tl.load(
        residual0_ptr + n * (RESIDUAL0_C_ * HW_) + c * HW_ + spatial,
        mask=add_mask,
        other=0.0,
    )
    residual1 = tl.load(
        residual1_ptr + n * (RESIDUAL1_C_ * HW_) + c * HW_ + spatial,
        mask=add_mask,
        other=0.0,
    )
    residual_sum = _bf16_add(residual0, residual1)
    add_value = _bf16_add(residual_sum, dense_bf16)
    tl.store(add_out_ptr + add_offsets, add_value, mask=add_mask)


# densenet121 train, N=4 C=448 H=W=28, residual slice channels 416:448.
@oracle_impl(hardware="B200", point="cbafcb75", BLOCK_R=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    residual0, residual1, mask_input, fill, source, centered_source, mean, invstd, weight = inputs
    device = source.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _bn_tail_kernel[(C,)](
        residual0,
        residual1,
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
        add_out,
        C_=C,
        HW_=HW,
        R_=R,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        RESIDUAL0_C_=RESIDUAL0_C,
        RESIDUAL1_C_=RESIDUAL1_C,
        SCALE_=SCALE,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, add_out
