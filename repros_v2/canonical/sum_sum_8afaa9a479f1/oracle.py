"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileViT SiLU-backward plus BatchNorm-backward fragment by sharing the bf16-rounded SiLU-gradient producer across both returned f32 channel reductions, then using the finalized per-channel summaries to write the returned channels-last bf16 input-gradient tensor. Inductor currently schedules the sliced channels-last gradient producer, affine/sigmoid SiLU derivative, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as separate generic regions over materialized intermediates; it cannot do this today because scheduler/codegen lacks a full-scope multi-output reduction plan that keeps compatible channel reductions, exact bf16/f32 cast boundaries, and the dependent dense epilogue together. The fix is SCHEDULER_FUSION: add a guarded channels-last SiLU-BN-backward template that shares the sliced producer and sinks finalized reduction scalars into the full output store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


N = 128
C = 128
IN_C = 256
SLICE_OFFSET = 128
H = 16
W = 16
HW = H * W
K_TOTAL = N * HW
TOTAL = K_TOTAL * C
REDUCE_SCALE = 3.0517578125e-05
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
def _silu_bn_producer(upstream_bf16, x_bf16, mean, invstd, weight, bias):
    upstream = upstream_bf16.to(tl.float32)
    x = x_bf16.to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    affine_scaled = _f32_mul(normalized, weight)
    affine = _f32_add(affine_scaled, bias)
    rounded_affine = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(
        tl.float32
    )

    exp_neg = libdevice.exp(_f32_sub(0.0, rounded_affine))
    sigmoid = _f32_div(1.0, _f32_add(exp_neg, 1.0))
    grad_sigmoid = _f32_mul(upstream, sigmoid)
    one_minus = _f32_sub(1.0, sigmoid)
    affine_tail = _f32_add(_f32_mul(rounded_affine, one_minus), 1.0)
    producer = _f32_mul(grad_sigmoid, affine_tail).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    return producer.to(tl.float32), centered


@triton.jit
def _partial_reduce_kernel(
    grad_pair_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    producer_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    NUM_K_TILES: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_lanes = tl.arange(0, BLOCK_K)
    k = tl.program_id(1) * GROUP_K + k_lanes
    active = (
        (k_lanes[:, None] < GROUP_K)
        & (k[:, None] < 32768)
        & (c[None, :] < 128)
    )

    x_offsets = k[:, None] * 128 + c[None, :]
    grad_offsets = k[:, None] * 256 + (128 + c[None, :])

    upstream = tl.load(grad_pair_ptr + grad_offsets, mask=active, other=0.0)
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0)
    mean = tl.load(mean_ptr + c, mask=c < 128, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < 128, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < 128, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < 128, other=0.0).to(tl.float32)

    producer, centered = _silu_bn_producer(
        upstream,
        x,
        mean[None, :],
        invstd[None, :],
        weight[None, :],
        bias[None, :],
    )
    active_producer = tl.where(active, producer, 0.0)
    active_dot = tl.where(active, _f32_mul(producer, centered), 0.0)

    tl.store(producer_ptr + x_offsets, producer.to(tl.bfloat16), mask=active)

    partial_offsets = c * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(active_producer, axis=0),
        mask=c < 128,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(active_dot, axis=0),
        mask=c < 128,
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
        tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
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
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < 4194304
    c = offsets % 128

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    producer = tl.load(producer_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(x, mean)
    mean_term = _f32_mul(sum_value, 3.0517578125e-05)
    dot_scaled = _f32_mul(dot_value, 3.0517578125e-05)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    after_variance = _f32_sub(producer, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    out = _f32_mul(after_mean, output_scale)

    tl.store(out_ptr + offsets, out.to(tl.bfloat16), mask=active)


# timm_mobilevit_s train, channels-last [128,128,16,16] SiLU-BN backward fed by arg0[:, 128:256].
@oracle_impl(
    hardware="B200",
    point="e23458e3",
    GROUP_K=256,
    BLOCK_K=256,
    BLOCK_C=16,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    num_k_tiles = triton.cdiv(K_TOTAL, GROUP_K)

    sum_out = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)
    scaled_dot = torch.empty_strided((C,), (1,), device=arg1_1.device, dtype=torch.float32)
    partial_sum = torch.empty_strided(
        (C, num_k_tiles),
        (num_k_tiles, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    partial_dot = torch.empty_like(partial_sum)
    producer = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(C, BLOCK_C), num_k_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        producer,
        partial_sum,
        partial_dot,
        NUM_K_TILES=num_k_tiles,
        GROUP_K=GROUP_K,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=4,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        arg3_1,
        sum_out,
        dot_tmp,
        scaled_dot,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=_next_power_of_2(num_k_tiles),
        num_warps=8,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        producer,
        sum_out,
        dot_tmp,
        out,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    return sum_out, scaled_dot, out
