"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail by split-K reducing the masked `where(arg0 <= 0, arg1, arg2)` producer into the two returned per-channel f32 summaries, finalizing the scale-gradient vector once, writing the full bf16 dense gradient, and returning the last-32-channel slice as an alias of that dense output, whereas Inductor schedules the sibling channel reductions, reduction-dependent dense epilogue, and slice return as separate generic regions around replayed or materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output split-K plan that keeps compatible channel summaries available to a dense epilogue while preserving bf16/f32 cast boundaries and aliasing output scope; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward split-K lowering that shares the masked producer, co-finalizes both channel reductions, and sinks the vector, dense, and slice-alias outputs into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 256
H = 56
W = 56
HW = H * W
K_TOTAL = N * HW
SCALE = 7.971938775510203e-05
SLICE_START = 224


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _nchw_offsets(
    c,
    k_offsets,
    C_: tl.constexpr,
    HW_: tl.constexpr,
):
    n = k_offsets // HW_
    spatial = k_offsets - n * HW_
    return n * C_ * HW_ + c * HW_ + spatial


@triton.jit
def _partial_reduce_kernel(
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    k_offsets = block * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL_
    offsets = _nchw_offsets(c, k_offsets, C_, HW_)

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    grad = tl.where(mask_value <= 0.0, fill, source).to(tl.float32)

    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(centered_source, mean)

    grad = tl.where(active, grad, 0.0)
    centered = tl.where(active, centered, 0.0)
    tl.store(partial_sum_ptr + block * C_ + c, tl.sum(grad, axis=0))
    tl.store(partial_dot_ptr + block * C_ + c, tl.sum(_f32_mul(grad, centered), axis=0))


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scale_grad_ptr,
    C_: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    c = tl.program_id(0)
    blocks = tl.arange(0, BLOCK_BLOCKS)
    active = blocks < NUM_BLOCKS
    offsets = blocks * C_ + c
    sum_value = tl.sum(tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    dot_value = tl.sum(tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    dense_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    k_offsets = block * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL_
    offsets = _nchw_offsets(c, k_offsets, C_, HW_)

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    grad = tl.where(mask_value <= 0.0, fill, source).to(tl.float32)

    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    sum_value = tl.load(sum_ptr + c).to(tl.float32)
    dot_value = tl.load(dot_ptr + c).to(tl.float32)

    centered = _f32_sub(centered_source, mean)
    mean_term = _f32_mul(sum_value, SCALE_)
    dot_scaled = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    corrected = _f32_sub(grad, _f32_mul(centered, variance_term))
    centered_grad = _f32_sub(corrected, mean_term)
    dense_bf16 = _f32_mul(centered_grad, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)


@oracle_impl(hardware="B200", point="878cf9f5", BLOCK_K=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    mask_input, fill, source, centered_source, mean, invstd, weight = inputs
    device = source.device
    num_blocks = triton.cdiv(K_TOTAL, BLOCK_K)
    block_blocks = _ceil_pow2(num_blocks)

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    grid = (C, num_blocks)
    _partial_reduce_kernel[grid](
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        partial_sum,
        partial_dot,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        invstd,
        sum_out,
        dot_tmp,
        scale_grad,
        C_=C,
        NUM_BLOCKS=num_blocks,
        BLOCK_BLOCKS=block_blocks,
        num_warps=8,
        num_stages=3,
    )
    _epilogue_kernel[grid](
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
        sum_out,
        dot_tmp,
        dense_out,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        SCALE_=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, dense_out[:, SLICE_START:C, :, :]
