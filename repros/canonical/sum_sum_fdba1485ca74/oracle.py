"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DenseNet bf16 BN-backward tail by splitting the shared N,H,W domain for the masked bf16 `where(arg2 <= 0, arg3, arg4)` producer, co-finalizing the two f32 channel reductions, writing the returned full bf16 gradient tensor, and forming the returned last-32-channel residual add with the required sequential bf16 slice-add rounding. Inductor currently schedules the sibling reductions, reduction-dependent dense BN epilogue, and final slice add as separate generic regions around replayed or materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output split-K plan that shares the conditional bf16 producer and keeps finalized channel summaries available to both dense and slice-limited consumers while preserving bf16 cast boundaries. The fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward split-K lowering that shares compatible channel reductions and sinks the vector, dense, and static residual-slice epilogues into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 7.971938775510203e-05


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _nchw_offsets(
    c,
    k_offsets,
    C: tl.constexpr,
    HW: tl.constexpr,
):
    n = k_offsets // HW
    hw = k_offsets - n * HW
    return n * C * HW + c * HW + hw


@triton.jit
def _partial_reduce_kernel(
    mask_src_ptr,
    fill_ptr,
    where_rhs_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    k_offsets = block * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL
    offsets = _nchw_offsets(c, k_offsets, C, HW)

    mask_src = tl.load(mask_src_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    where_rhs = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0)
    grad = tl.where(mask_src <= 0.0, fill, where_rhs).to(tl.float32)

    activation = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(activation, mean)

    partial_sum = tl.sum(tl.where(active, grad, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(active, _f32_mul(grad, centered), 0.0), axis=0)
    out_offset = block * C + c
    tl.store(partial_sum_ptr + out_offset, partial_sum)
    tl.store(partial_dot_ptr + out_offset, partial_dot)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_out_ptr,
    mul8_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    c = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = block_offsets < NUM_K_BLOCKS
    offsets = block_offsets * C + c

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_out_ptr + c, dot_value)
    tl.store(mul8_out_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    residual0_ptr,
    residual1_ptr,
    mask_src_ptr,
    fill_ptr,
    where_rhs_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    dense_out_ptr,
    add_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    k_offsets = block * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL
    offsets = _nchw_offsets(c, k_offsets, C, HW)

    mask_src = tl.load(mask_src_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    where_rhs = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0)
    grad = tl.where(mask_src <= 0.0, fill, where_rhs).to(tl.float32)

    activation = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    sum_value = tl.load(sum_ptr + c).to(tl.float32)
    dot_value = tl.load(dot_ptr + c).to(tl.float32)

    centered = _f32_sub(activation, mean)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_scaled = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    after_variance = _f32_sub(grad, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    dense_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)

    in_slice = c >= 160
    n = k_offsets // HW
    hw = k_offsets - n * HW
    slice_c = c - 160
    add_mask = active & in_slice
    residual0_offsets = n * (256 * HW) + c * HW + hw
    residual1_offsets = n * (224 * HW) + c * HW + hw
    add_offsets = n * (32 * HW) + slice_c * HW + hw
    residual0 = tl.load(residual0_ptr + residual0_offsets, mask=add_mask, other=0.0)
    residual1 = tl.load(residual1_ptr + residual1_offsets, mask=add_mask, other=0.0)
    base_add = _bf16_add(residual0, residual1)
    add_value = _bf16_add(base_add, dense_bf16)
    tl.store(add_out_ptr + add_offsets, add_value, mask=add_mask)


# 611efccd: densenet121 train, N=4 C=192 H=W=56, residual slice channels 160:192.
@oracle_impl(hardware="B200", point="611efccd", BLOCK_K=8192, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 = inputs
    c = 192
    hw = 56 * 56
    k_total = 4 * hw
    num_k_blocks = triton.cdiv(k_total, BLOCK_K)
    block_blocks = _ceil_pow2(num_k_blocks)

    sum_out = torch.empty_strided((c,), (1,), device=arg2.device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((c,), (1,), device=arg2.device, dtype=torch.float32)
    mul8_out = torch.empty_strided((c,), (1,), device=arg2.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (4, c, 56, 56),
        (c * hw, hw, 56, 1),
        device=arg2.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (4, 32, 56, 56),
        (32 * hw, hw, 56, 1),
        device=arg2.device,
        dtype=torch.bfloat16,
    )
    partial_sum = torch.empty((num_k_blocks, c), device=arg2.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    grid = (c, num_k_blocks)
    _partial_reduce_kernel[grid](
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg7,
        sum_out,
        dot_tmp,
        mul8_out,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_BLOCKS=block_blocks,
        num_warps=8,
    )
    _epilogue_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        sum_out,
        dot_tmp,
        dense_out,
        add_out,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return sum_out, mul8_out, dense_out, add_out
