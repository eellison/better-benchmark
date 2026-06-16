"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 SiLU-backward plus BatchNorm-backward fragment by sharing the bf16-rounded SiLU-gradient producer across both channel reductions, then using the finalized per-channel summaries to emit the returned raw sum vector, scale-gradient vector, and dense bf16 gradient tensor, whereas Inductor currently schedules the broadcasted affine/sigmoid producer, sibling `sum([0, 2, 3])` reductions, and dependent full-tensor epilogue as separate generic regions over large intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned full-scope multi-output reduction template that keeps compatible channel reductions, explicit bf16 conversion boundaries, and their dependent materializing epilogue in one fused plan for channels-last NCHW tensors; the fix is SCHEDULER_FUSION: add scheduler/codegen support for shared SiLU-BN-backward channel reductions with finalized-scalar epilogues that write both vector and dense outputs while preserving low-precision casts and natural-exp sigmoid lowering."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


REDUCE_SCALE = 4.76837158203125e-07
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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _silu_bn_producer(grad_bf16, x_bf16, mean, invstd, weight, bias):
    grad = grad_bf16.to(tl.float32)
    x = x_bf16.to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine_scaled = _f32_mul(normalized, weight)
    affine = _f32_add(affine_scaled, bias)
    rounded_affine = affine.to(tl.bfloat16).to(tl.float32)

    exp_neg = libdevice.exp(_f32_sub(0.0, rounded_affine))
    sigmoid = _f32_div(1.0, _f32_add(exp_neg, 1.0))
    grad_sigmoid = _f32_mul(grad, sigmoid)
    one_minus_sigmoid = _f32_sub(1.0, sigmoid)
    affine_tail = _f32_add(_f32_mul(rounded_affine, one_minus_sigmoid), 1.0)
    producer = _f32_mul(grad_sigmoid, affine_tail).to(tl.bfloat16).to(tl.float32)
    return producer, centered


@triton.jit
def _partial_reduce_kernel(
    grad_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    producer_ptr,
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

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    producer, centered = _silu_bn_producer(
        grad,
        x,
        mean[None, :],
        invstd[None, :],
        weight[None, :],
        bias[None, :],
    )
    active_producer = tl.where(mask, producer, 0.0)
    active_dot = tl.where(mask, _f32_mul(producer, centered), 0.0)

    tl.store(producer_ptr + offsets, producer, mask=mask)

    partial_offsets = c_offsets * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(active_producer, axis=0),
        mask=c_offsets < C,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(active_dot, axis=0),
        mask=c_offsets < C,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scaled_dot_out_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    partial_offsets = c * NUM_K_TILES + tile_offsets

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    producer_ptr,
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

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    producer = tl.load(producer_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_scaled = _f32_mul(dot_value, SCALE_VALUE)
    correction = _f32_mul(dot_scaled, _f32_mul(invstd, invstd))
    output_scale = _f32_mul(invstd, weight)
    centered_correction = _f32_mul(centered, correction)
    without_var = _f32_sub(producer, centered_correction)
    without_mean = _f32_sub(without_var, mean_term)
    result = _f32_mul(without_mean, output_scale)

    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="57aa57bf", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="4a93118a", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="f444ad5f", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="0312171c", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="65ca7fc0", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="315c8b2f", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="f11d7f2c", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="04100efe", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="9fc1d6b0", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="a6ded16e", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="0466d5cd", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="6ef366bb", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="def98aed", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="d6507bc1", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="d6acaf93", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="787e6d78", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="1bde3bc1", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="14e1266b", BLOCK_K=256, BLOCK_C=16, num_warps=8)
@oracle_impl(hardware="B200", point="168f8cc2", BLOCK_K=256, BLOCK_C=16, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_C: int, num_warps: int):
    grad, x, mean, invstd, weight, bias = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    k_total = n * h * w
    total = grad.numel()
    num_k_tiles = triton.cdiv(k_total, BLOCK_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    out_sum = torch.empty((c,), device=grad.device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=grad.device, dtype=torch.float32)
    out_vec = torch.empty((c,), device=grad.device, dtype=torch.float32)
    partial_sum = torch.empty((c, num_k_tiles), device=grad.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    producer = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        grad,
        x,
        mean,
        invstd,
        weight,
        bias,
        producer,
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
        invstd,
        out_sum,
        dot_tmp,
        out_vec,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        x,
        mean,
        invstd,
        weight,
        producer,
        out_sum,
        dot_tmp,
        out,
        TOTAL=total,
        C=c,
        SCALE_VALUE=REDUCE_SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return out_sum, out_vec, out
