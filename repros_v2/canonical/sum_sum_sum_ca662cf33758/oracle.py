"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 PyTorch-UNet BatchNorm-backward tail by splitting the shared `N,H,W` domain for the `where` producer, co-finalizing the two f32 channel summaries, using those summaries to write the full returned bf16 dense gradient, and accumulating the bf16-rounded dense output's returned channel sum from the same split domain, whereas Inductor schedules the sibling channel reductions, dependent dense epilogue, and final bf16-output reduction as separate generic reduction/pointwise regions over replayed or materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned cooperative split-K lowering that coordinates a conditional bf16 producer, multiple channel summaries, a dependent full-tensor epilogue, and a post-cast bf16 side reduction; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward split-K lowering that shares compatible channel reductions, preserves the explicit bf16 cast before the final sum, and sinks the vector and dense epilogues into one coordinated plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 1.6293013555787278e-06


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
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
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
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL
    offsets = _nchw_offsets(c, k_offsets, C, HW)

    mask_src = tl.load(mask_src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    where_rhs = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    grad = tl.where(mask_src <= 0.0, fill, where_rhs)

    activation = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(activation, mean)

    partial_sum = tl.sum(tl.where(active, grad, 0.0), axis=0)
    partial_dot = tl.sum(tl.where(active, _f32_mul(grad, centered), 0.0), axis=0)
    out_offset = tl.program_id(1) * C + c
    tl.store(partial_sum_ptr + out_offset, partial_sum)
    tl.store(partial_dot_ptr + out_offset, partial_dot)


@triton.jit
def _finalize_first_kernel(
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
    partial_dense_sum_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k_offsets < K_TOTAL
    offsets = _nchw_offsets(c, k_offsets, C, HW)

    mask_src = tl.load(mask_src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    where_rhs = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    grad = tl.where(mask_src <= 0.0, fill, where_rhs)

    activation = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    sum_value = tl.load(sum_ptr + c).to(tl.float32)
    dot_value = tl.load(dot_ptr + c).to(tl.float32)

    centered = _f32_sub(activation, mean)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(grad, correction)
    centered_grad = _f32_sub(after_correction, mean_term)
    output_scale = _f32_mul(invstd, weight)
    dense_f32 = _f32_mul(centered_grad, output_scale)
    dense_bf16_f32 = _round_to_bf16_f32(dense_f32)

    tl.store(dense_out_ptr + offsets, dense_bf16_f32, mask=active)

    partial_sum = tl.sum(tl.where(active, dense_bf16_f32, 0.0), axis=0)
    tl.store(partial_dense_sum_ptr + tl.program_id(1) * C + c, partial_sum)


@triton.jit
def _finalize_dense_sum_kernel(
    partial_dense_sum_ptr,
    dense_sum_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_BLOCKS: tl.constexpr,
):
    c = tl.program_id(0)
    block_offsets = tl.arange(0, BLOCK_BLOCKS)
    active = block_offsets < NUM_K_BLOCKS
    offsets = block_offsets * C + c
    total = tl.sum(
        tl.load(partial_dense_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    tl.store(dense_sum_out_ptr + c, _round_to_bf16_f32(total))


# 39b5812f: (T([1,64,640,959], bf16), T([], bf16), ...)
@oracle_impl(hardware="B200", point="39b5812f", BLOCK_K=8192, num_warps=8)
# 9309c5e2: (T([1,128,320,479], bf16), T([], bf16), ...)
@oracle_impl(hardware="B200", point="9309c5e2", BLOCK_K=4096, num_warps=8)
# 0439afe2: (T([1,256,160,239], bf16), T([], bf16), ...)
@oracle_impl(hardware="B200", point="0439afe2", BLOCK_K=2048, num_warps=8)
# 457eec03: (T([1,512,80,119], bf16), T([], bf16), ...)
@oracle_impl(hardware="B200", point="457eec03", BLOCK_K=1024, num_warps=4)
# f63148ef: (T([1,512,40,59], bf16), T([], bf16), ...)
@oracle_impl(hardware="B200", point="f63148ef", BLOCK_K=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    n = int(arg0.shape[0])
    c = int(arg0.shape[1])
    h = int(arg0.shape[2])
    w = int(arg0.shape[3])
    hw = h * w
    k_total = n * hw
    num_k_blocks = triton.cdiv(k_total, BLOCK_K)
    block_blocks = _ceil_pow2(num_k_blocks)

    sum_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    mul8_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    dense_sum_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    partial_sum = torch.empty((num_k_blocks, c), device=arg0.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    partial_dense_sum = torch.empty_like(partial_sum)

    grid = (c, num_k_blocks)
    _partial_reduce_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    _finalize_first_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg5,
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
        sum_out,
        dot_tmp,
        dense_out,
        partial_dense_sum,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    _finalize_dense_sum_kernel[(c,)](
        partial_dense_sum,
        dense_sum_out,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_BLOCKS=block_blocks,
        num_warps=8,
    )
    return sum_out, mul8_out, dense_out, dense_sum_out
