"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 bf16 hard-swish BatchNorm-backward fragment by sharing the rounded residual-add and hard-swish-gradient producer across both `[16]` channel reductions, then reusing the finalized per-channel summaries to emit the returned raw sum vector, scale-gradient vector, and dense channels-last bf16 input-gradient tensor, whereas Inductor schedules the broadcast affine/hard-swish producer, sibling `sum([0,2,3])` reductions, and dependent full-tensor epilogue as generic multi-output reduction work over large channels-last intermediates; Inductor cannot do this today because its scheduler/codegen lacks a full-scope template that keeps compatible channel reductions, explicit bf16 rounding boundaries, and their finalized BN-backward epilogue in one fused plan; the fix is SCHEDULER_FUSION: add scheduler/codegen support for hard-swish-gradient channel reductions with dependent tensor/vector epilogues over NHWC physical layouts."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 16
H = 112
W = 112
HW = H * W
SCALE = 1.5570192920918366e-07


@triton.jit
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _producer_values(
    grad0,
    grad1,
    x,
    mean,
    invstd,
    weight,
    bias,
    fill,
):
    grad_add = _add_f32(grad0.to(tl.float32), grad1.to(tl.float32))
    grad = _round_bf16_to_f32(grad_add)
    centered = _sub_f32(x.to(tl.float32), mean)
    affine_mul0 = _mul_f32(centered, invstd)
    affine_mul1 = _mul_f32(affine_mul0, weight)
    affine = _add_f32(affine_mul1, bias)
    affine = _round_bf16_to_f32(affine)
    affine_div = affine / 3.0
    affine_add = _add_f32(affine_div, 0.5)
    middle = _mul_f32(grad, affine_add)
    value = tl.where(affine < 3.0, middle, grad)
    value = tl.where(affine <= -3.0, fill, value)
    return _round_bf16_to_f32(value), centered


@triton.jit
def _partial_sums_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    K_TOTAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    c = tl.arange(0, BLOCK_C)[None, :]
    c_vec = tl.arange(0, BLOCK_C)
    mask = k < K_TOTAL
    offsets = k * CHANNELS + c

    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    grad0 = tl.load(grad0_ptr + offsets, mask=mask, other=0.0)
    grad1 = tl.load(grad1_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    value, centered = _producer_values(grad0, grad1, x, mean, invstd, weight, bias, fill)

    active_value = tl.where(mask, value, 0.0)
    active_dot = tl.where(mask, _mul_f32(value, centered), 0.0)
    partial_base = tl.program_id(0) * CHANNELS + c_vec
    tl.store(partial_sum_ptr + partial_base, tl.sum(active_value, axis=0))
    tl.store(partial_dot_ptr + partial_base, tl.sum(active_dot, axis=0))


@triton.jit
def _finalize_sums_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    out_sum_ptr,
    dot_sum_ptr,
    out_vec_ptr,
    NUM_TILES: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tiles = tl.arange(0, BLOCK_TILES)[:, None]
    c = tl.arange(0, BLOCK_C)[None, :]
    c_vec = tl.arange(0, BLOCK_C)
    active = tiles < NUM_TILES
    offsets = tiles * CHANNELS + c

    sum_partials = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    dot_partials = tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_partials, axis=0)
    dot_value = tl.sum(dot_partials, axis=0)
    invstd = tl.load(invstd_ptr + c_vec).to(tl.float32)

    tl.store(out_sum_ptr + c_vec, sum_value)
    tl.store(dot_sum_ptr + c_vec, dot_value)
    tl.store(out_vec_ptr + c_vec, _mul_f32(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum_ptr,
    dot_sum_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    CHANNELS: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % CHANNELS

    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    grad0 = tl.load(grad0_ptr + offsets, mask=active, other=0.0)
    grad1 = tl.load(grad1_ptr + offsets, mask=active, other=0.0)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0)
    value, centered = _producer_values(grad0, grad1, x, mean, invstd, weight, bias, fill)

    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = _mul_f32(sum_value, SCALE_VALUE)
    dot_scaled = _mul_f32(dot_value, SCALE_VALUE)
    invstd_sq = _mul_f32(invstd, invstd)
    centered_term = _mul_f32(dot_scaled, invstd_sq)
    output_scale = _mul_f32(invstd, weight)
    centered_scaled = _mul_f32(centered, centered_term)
    sub0 = _sub_f32(value, centered_scaled)
    sub1 = _sub_f32(sub0, mean_term)
    output = _mul_f32(sub1, output_scale)
    output = _round_bf16_to_f32(output)
    tl.store(out_ptr + offsets, output, mask=active)

@oracle_impl(hardware="B200", point="44b468fa", BLOCK_K=2048, EPILOGUE_BLOCK=1024, num_warps=8)
@oracle_impl(hardware="B200", point="7088b6e2", BLOCK_K=1024, EPILOGUE_BLOCK=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, EPILOGUE_BLOCK: int, num_warps: int):
    grad0, grad1, x, mean, invstd, weight, bias, fill = inputs
    n = int(x.shape[0])
    total = n * C * HW
    k_total = n * HW
    num_tiles = triton.cdiv(k_total, BLOCK_K)

    partial_sum = torch.empty_strided((num_tiles, C), (C, 1), device=x.device, dtype=torch.float32)
    partial_dot = torch.empty_strided((num_tiles, C), (C, 1), device=x.device, dtype=torch.float32)
    out_sum = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    dot_sum = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    out_vec = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (n, C, H, W),
        (C * HW, 1, W * C, C),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _partial_sums_kernel[(num_tiles,)](
        grad0,
        grad1,
        x,
        mean,
        invstd,
        weight,
        bias,
        fill,
        partial_sum,
        partial_dot,
        K_TOTAL=k_total,
        CHANNELS=C,
        BLOCK_K=BLOCK_K,
        BLOCK_C=C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_sums_kernel[(1,)](
        partial_sum,
        partial_dot,
        invstd,
        out_sum,
        dot_sum,
        out_vec,
        NUM_TILES=num_tiles,
        CHANNELS=C,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        BLOCK_C=C,
        num_warps=8,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        grad0,
        grad1,
        x,
        mean,
        invstd,
        weight,
        bias,
        fill,
        out_sum,
        dot_sum,
        out,
        TOTAL=total,
        CHANNELS=C,
        SCALE_VALUE=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out_sum, out_vec, out
