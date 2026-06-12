"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 MobileNet-style BatchNorm-backward tail by materializing the returned bf16 residual add, split-K reducing the two compatible per-channel summaries from that rounded producer, and using the finalized summaries to emit both fp32 side vectors and the dependent bf16 dense gradient tensor, whereas Inductor schedules the returned add producer, sibling `sum([0, 2, 3])` reductions, and broadcast epilogue as generic reduction/pointwise work; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates a visible producer, compatible channel reductions, and dependent vector/tensor epilogues while preserving bf16 rounding boundaries; the fix is COOPERATIVE_SPLIT_K: add a channel split-K reduction plan that co-finalizes shared summaries and fuses the full BN-backward epilogue around returned side outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 9.964923469387754e-06
EPILOGUE_BLOCK = 1024


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
def _partial_reduce_add_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    add_out_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K_TOTAL)
    offsets = k_offsets * C + c_offsets

    x0 = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x1 = tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg3_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    add_bf16 = _f32_add(x0, x1).to(tl.bfloat16)
    add_f32 = add_bf16.to(tl.float32)
    centered = _f32_sub(x2, mean)
    dot = _f32_mul(add_f32, centered)

    tl.store(add_out_ptr + offsets, add_bf16, mask=active)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.program_id(1) * C + c_vec
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(tl.where(active, add_f32, 0.0), axis=0),
        mask=c_vec < C,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(tl.where(active, dot, 0.0), axis=0),
        mask=c_vec < C,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    arg4_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    dot_scaled_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    p_offsets = tl.arange(0, BLOCK_P)[:, None]
    active = (c_offsets < C) & (p_offsets < NUM_K_BLOCKS)
    offsets = p_offsets * C + c_offsets

    partial_sum = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_dot = tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    total_sum = tl.sum(partial_sum, axis=0)
    total_dot = tl.sum(partial_dot, axis=0)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    weight = tl.load(arg4_ptr + c_vec, mask=c_vec < C, other=0.0).to(tl.float32)
    tl.store(sum_out_ptr + c_vec, total_sum, mask=c_vec < C)
    tl.store(dot_tmp_ptr + c_vec, total_dot, mask=c_vec < C)
    tl.store(dot_scaled_out_ptr + c_vec, _f32_mul(total_dot, weight), mask=c_vec < C)


@triton.jit
def _epilogue_kernel(
    add_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % C

    add_value = tl.load(add_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    x2 = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg3_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(arg4_ptr + c, mask=active, other=0.0).to(tl.float32)
    grad_weight = tl.load(arg5_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x2, mean)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    weight_sq = _f32_mul(weight, weight)
    correction_scale = _f32_mul(dot_mean, weight_sq)
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(add_value, correction)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    centered_grad = _f32_sub(after_correction, mean_term)
    output_scale = _f32_mul(weight, grad_weight)
    out = _f32_mul(centered_grad, output_scale)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="65b876e3", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="1b9feebb", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="3726f4ca", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="9793b43e", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="399aa3e2", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="39a9326e", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="b6f518ab", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="a45e6340", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="6f1023fc", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="b5264010", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="315c2b3e", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="0dc5b6bd", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="864b3c6f", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="cf15f756", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="727b7028", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="3edd6c00", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="ee318906", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="1592ce3d", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="ebb56431", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="f35ade00", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    num_warps: int,
):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    n, c, h, w = (int(dim) for dim in arg0.shape)
    total = int(arg0.numel())
    k_total = n * h * w
    num_k_blocks = triton.cdiv(k_total, BLOCK_K)
    block_p = _ceil_pow2(num_k_blocks)

    out_stride = tuple(int(s) for s in arg0.stride())
    add_out = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    dense_out = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    dot_scaled_out = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    partial_sum = torch.empty((num_k_blocks, c), device=arg0.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)

    grid = (triton.cdiv(c, BLOCK_C), num_k_blocks)
    _partial_reduce_add_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        add_out,
        partial_sum,
        partial_dot,
        C=c,
        K_TOTAL=k_total,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(triton.cdiv(c, BLOCK_C),)](
        partial_sum,
        partial_dot,
        arg4,
        sum_out,
        dot_tmp,
        dot_scaled_out,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_P=block_p,
        BLOCK_C=BLOCK_C,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        add_out,
        arg2,
        arg3,
        arg4,
        arg5,
        sum_out,
        dot_tmp,
        dense_out,
        TOTAL=total,
        C=c,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return add_out, sum_out, dot_scaled_out, dense_out
