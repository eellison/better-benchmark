"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete phlippe DenseNet bf16 BN-backward tail by split-K reducing the shared masked `where` producer into both f32 channel summaries, finalizing the returned raw-sum and scale-gradient vectors once, then sinking those summaries into the bf16 residual-slice add epilogue while returning the required `add[:, :16]` aliasing view; Inductor currently schedules the sibling reductions, broadcast BN-backward epilogue, bf16 cast/add, and returned slice as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope cooperative multi-output channel-reduction plan that preserves bf16/f32 cast boundaries, dependent dense epilogues, and output aliasing together; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to split compatible BN-backward channel reductions across N/H/W, combine partial summaries, and fuse the downstream residual-add epilogue plus alias-view return."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 48
INPUT_C = 64
SLICE_START = 16
SLICE_C = 16
H = 32
W = 32
HW = H * W
K_TOTAL = N * HW
SCALE = 7.62939453125e-06
BLOCK_HW = 1024
BLOCK_N = 128
BLOCK_C = 8


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
def _producer(
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    n_idx,
    c_offsets,
    hw_offsets,
    C_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
):
    c_mask = c_offsets < C_N
    hw_mask = hw_offsets < HW_N
    mask = c_mask[:, None] & hw_mask[None, :]
    offsets = n_idx * (C_N * HW_N) + c_offsets[:, None] * HW_N + hw_offsets[None, :]

    gate = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
    fill = tl.load(fill_ptr)
    selected = tl.where(gate <= 0.0, fill, rhs).to(tl.float32)

    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    centered = _f32_sub(activation, mean[:, None])
    selected = tl.where(mask, selected, 0.0)
    return selected, centered, mask, offsets


@triton.jit
def _partial_reduce_kernel(
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    hw_offsets = tl.arange(0, BLOCK_HW_N)
    selected, centered, _, _ = _producer(
        mask_ptr,
        fill_ptr,
        rhs_ptr,
        activation_ptr,
        mean_ptr,
        n_idx,
        c_offsets,
        hw_offsets,
        C_N,
        HW_N,
        BLOCK_C_N,
        BLOCK_HW_N,
    )
    partial_sum = tl.sum(selected, axis=1)
    partial_dot = tl.sum(_f32_mul(selected, centered), axis=1)
    offsets = n_idx * C_N + c_offsets
    c_mask = c_offsets < C_N
    tl.store(partial_sum_ptr + offsets, partial_sum, mask=c_mask)
    tl.store(partial_dot_ptr + offsets, partial_dot, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    stats_ptr,
    C_N: tl.constexpr,
    N_N: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_N_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_offsets = tl.arange(0, BLOCK_N_N)
    c_mask = c_offsets < C_N
    n_mask = n_offsets < N_N
    offsets = n_offsets[:, None] * C_N + c_offsets[None, :]
    mask = n_mask[:, None] & c_mask[None, :]

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    partial_dot = tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.sum(partial_sum, axis=0)
    dot_value = tl.sum(partial_dot, axis=0)

    invstd = tl.load(invstd_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c_offsets, sum_value, mask=c_mask)
    tl.store(scaled_dot_out_ptr + c_offsets, _f32_mul(dot_value, invstd), mask=c_mask)
    tl.store(stats_ptr + c_offsets, mean_term, mask=c_mask)
    tl.store(stats_ptr + C_N + c_offsets, correction_scale, mask=c_mask)
    tl.store(stats_ptr + 2 * C_N + c_offsets, output_scale, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    stats_ptr,
    add_out_ptr,
    C_N: tl.constexpr,
    INPUT_C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    HW_N: tl.constexpr,
    BLOCK_C_N: tl.constexpr,
    BLOCK_HW_N: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C_N + tl.arange(0, BLOCK_C_N)
    n_idx = tl.program_id(1)
    hw_offsets = tl.arange(0, BLOCK_HW_N)
    selected, centered, mask, compact_offsets = _producer(
        mask_ptr,
        fill_ptr,
        rhs_ptr,
        activation_ptr,
        mean_ptr,
        n_idx,
        c_offsets,
        hw_offsets,
        C_N,
        HW_N,
        BLOCK_C_N,
        BLOCK_HW_N,
    )

    mean_term = tl.load(stats_ptr + c_offsets, mask=c_offsets < C_N, other=0.0).to(
        tl.float32
    )
    correction_scale = tl.load(
        stats_ptr + C_N + c_offsets, mask=c_offsets < C_N, other=0.0
    ).to(tl.float32)
    output_scale = tl.load(
        stats_ptr + 2 * C_N + c_offsets, mask=c_offsets < C_N, other=0.0
    ).to(tl.float32)

    correction = _f32_mul(centered, correction_scale[:, None])
    after_correction = _f32_sub(selected, correction)
    after_mean = _f32_sub(after_correction, mean_term[:, None])
    grad_bf16 = _f32_mul(after_mean, output_scale[:, None]).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual_offsets = (
        n_idx * (INPUT_C_N * HW_N)
        + (c_offsets[:, None] + SLICE_START_N) * HW_N
        + hw_offsets[None, :]
    )
    residual = tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0)
    tl.store(add_out_ptr + compact_offsets, _bf16_add(residual, grad_bf16), mask=mask)


# a7975a32: phlippe_densenet train, N=128 C=48 H=W=32, residual slice channels 16:64.
@oracle_impl(hardware="B200", point="a7975a32")
def oracle_forward(inputs):
    residual, mask, fill, rhs, activation, mean, invstd, weight = inputs

    sum_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((N, C), device=mask.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    stats = torch.empty((3, C), device=mask.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), N)
    _partial_reduce_kernel[grid](
        mask,
        fill,
        rhs,
        activation,
        mean,
        partial_sum,
        partial_dot,
        C_N=C,
        HW_N=HW,
        BLOCK_C_N=BLOCK_C,
        BLOCK_HW_N=BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    _finalize_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        scaled_dot_out,
        stats,
        C_N=C,
        N_N=N,
        SCALE_VALUE=SCALE,
        BLOCK_N_N=BLOCK_N,
        BLOCK_C_N=BLOCK_C,
        num_warps=8,
        num_stages=4,
    )
    _epilogue_kernel[grid](
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        stats,
        add_out,
        C_N=C,
        INPUT_C_N=INPUT_C,
        SLICE_START_N=SLICE_START,
        HW_N=HW,
        BLOCK_C_N=BLOCK_C,
        BLOCK_HW_N=BLOCK_HW,
        num_warps=4,
        num_stages=4,
    )
    slice_out = torch.as_strided(
        add_out,
        (N, SLICE_C, H, W),
        (C * HW, HW, W, 1),
    )
    return sum_out, scaled_dot_out, add_out, slice_out
