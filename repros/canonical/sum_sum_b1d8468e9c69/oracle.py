"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet bf16 batch-norm-backward tail by sharing the masked bf16 where producer across both f32 channel reductions, immediately consuming the channel summaries in the returned full bf16 gradient tensor, and forming the returned last-32-channel residual add with the required bf16 cast boundary, whereas Inductor schedules the sibling reductions, reduction-dependent dense epilogue, and final slice add as separate generic regions around materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output reduction plan that keeps compatible channel summaries available to both dense and slice-limited consumers while preserving bf16 casts and output scope; the fix is SCHEDULER_FUSION: teach reduction scheduling to share the masked producer and sink finalized channel summaries into both the dense BN-backward store and the static residual-slice epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 992
ARG0_C = 1024
SLICE_START = 960
SLICE_C = C - SLICE_START
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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _densenet_bn_tail_kernel(
    residual_ptr,
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
    ARG0_C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    SCALE_: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    dense_offsets = n * (C_ * HW) + c * HW + spatial

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
    add_offsets = n * (SLICE_C_ * HW) + slice_c * HW + spatial
    residual_offsets = n * (ARG0_C_ * HW) + c * HW + spatial
    add_mask = active & in_slice
    residual_value = tl.load(residual_ptr + residual_offsets, mask=add_mask, other=0.0)
    add_value = _bf16_add(residual_value, dense_bf16)
    tl.store(add_out_ptr + add_offsets, add_value, mask=add_mask)


def _launch(inputs, *, H: int, W: int, BLOCK_R: int, num_warps: int):
    (
        residual,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
    ) = inputs
    hw = H * W
    r = N * hw
    device = source.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * hw, hw, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * hw, hw, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _densenet_bn_tail_kernel[(C,)](
        residual,
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
        ARG0_C_=ARG0_C,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        H=H,
        W=W,
        HW=hw,
        R=r,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, add_out


# densenet121 train, C=992, H=W=14; scale remains the captured graph literal.
@oracle_impl(hardware="B200", point="f1255d51", H=14, W=14, BLOCK_R=1024, num_warps=8)
# densenet121 train, C=992, H=W=7; scale remains the captured graph literal.
@oracle_impl(hardware="B200", point="6cbb3519", H=7, W=7, BLOCK_R=256, num_warps=4)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, **kwargs)
