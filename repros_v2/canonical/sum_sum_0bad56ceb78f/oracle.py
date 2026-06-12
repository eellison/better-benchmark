"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileNetV2 hardtanh-masked BatchNorm-backward tail by sharing the bf16-rounded affine comparison producer across both channel reductions and then using the finalized per-channel sums to emit the raw sum vector, scale-gradient vector, and dense bf16 gradient tensor, whereas Inductor currently schedules the mask-producing affine, sibling `sum([0, 2, 3])` reductions, and dependent dense epilogue as separate generic reduction/pointwise work; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned full-scope multi-output reduction template that keeps compatible masked channel reductions and their finalized BN-backward epilogue in one fused plan for channels-last NCHW tensors; the fix is SCHEDULER_FUSION: add scheduler/codegen support for hardtanh-masked BN-backward channel reductions with dependent vector and dense bf16 epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 6.228077168367346e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
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
def _partial_reduce_nhwc_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (k_offsets[:, None] < K_TOTAL) & (c_offsets[None, :] < C)

    x_bf16 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0)
    grad_bf16 = tl.load(arg6_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(arg1_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    invstd = tl.load(arg2_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    weight = tl.load(arg3_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    bias = tl.load(arg4_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    fill = tl.load(arg5_ptr)

    x_f32 = x_bf16.to(tl.float32)
    centered = _f32_sub(x_f32, mean[None, :])
    mul0 = _f32_mul(centered, invstd[None, :])
    mul1 = _f32_mul(mul0, weight[None, :])
    affine_bf16 = _f32_add(mul1, bias[None, :]).to(tl.bfloat16)
    where_value = tl.where((affine_bf16 <= 0.0) | (affine_bf16 >= 6.0), fill, grad_bf16).to(tl.float32)

    active_where = tl.where(mask, where_value, 0.0)
    active_dot = tl.where(mask, where_value * centered, 0.0)
    out_offsets = c_offsets * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + out_offsets,
        tl.sum(active_where, axis=0),
        mask=c_offsets < C,
    )
    tl.store(
        partial_dot_ptr + out_offsets,
        tl.sum(active_dot, axis=0),
        mask=c_offsets < C,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    arg2_ptr,
    out_sum_ptr,
    sum2_ptr,
    out_vec_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    partial_offsets = c * NUM_K_TILES + tile_offsets

    sum1 = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum2 = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(arg2_ptr + c).to(tl.float32)

    tl.store(out_sum_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(out_vec_ptr + c, sum2 * invstd)


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % C

    x_bf16 = tl.load(arg0_ptr + offsets, mask=active, other=0.0)
    grad_bf16 = tl.load(arg6_ptr + offsets, mask=active, other=0.0)
    mean = tl.load(arg1_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg2_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(arg3_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(arg4_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(arg5_ptr)

    x_f32 = x_bf16.to(tl.float32)
    centered = _f32_sub(x_f32, mean)
    mul0 = _f32_mul(centered, invstd)
    mul1 = _f32_mul(mul0, weight)
    affine_bf16 = _f32_add(mul1, bias).to(tl.bfloat16)
    where_value = tl.where((affine_bf16 <= 0.0) | (affine_bf16 >= 6.0), fill, grad_bf16).to(tl.float32)

    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = sum1 * SCALE_VALUE
    correction = sum2 * SCALE_VALUE * invstd * invstd
    output_scale = invstd * weight
    result = (where_value - centered * correction - mean_term) * output_scale
    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="8693ecd8", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="e4902295", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="cea53ab1", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="d3515732", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="72305b66", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="ddbc642b", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="20e7cc5d", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="8fbbedd9", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="cfc1c8b3", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="425108c3", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="47a686d2", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="1e7379f7", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="a5d13e74", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="aad9cbeb", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="2f19ce6a", BLOCK_K=2048, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="aba87b74", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="3dce5391", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="0a38f4eb", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="5951d630", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="45529174", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="4785ab0c", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
@oracle_impl(hardware="B200", point="7bc04cf2", BLOCK_K=1024, BLOCK_C=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, num_warps):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    n, c, h, w = arg0.shape
    k_total = n * h * w
    total = arg0.numel()
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    partial_sum = torch.empty((c, num_k_tiles), device=arg0.device, dtype=torch.float32)
    partial_dot = torch.empty((c, num_k_tiles), device=arg0.device, dtype=torch.float32)
    out_sum = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out_vec = torch.empty((c,), device=arg0.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=arg0.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_nhwc_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        partial_sum,
        partial_dot,
        C=c,
        K_TOTAL=k_total,
        NUM_K_TILES=num_k_tiles,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        arg2,
        out_sum,
        sum2,
        out_vec,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        out_sum,
        sum2,
        out,
        TOTAL=total,
        C=c,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return out_sum, out_vec, out
