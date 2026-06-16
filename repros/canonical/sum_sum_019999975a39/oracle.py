"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete structured average-pool-backward, bf16-rounded SiLU-gradient producer, both channel reductions, and the dependent bf16 BN-backward tensor directly from the pooled `[128, C]` gradient and channels-last activation, whereas Inductor lowers the zero-fill as_strided_scatter/expand/div producer and then schedules the sibling reductions plus dense epilogue as separate generic regions over large materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model this zero-fill scatter followed by broadcast average-pool backward as a structured scatter-reduce producer feeding both reductions and a layout-preserving dependent epilogue; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that shares the source pooled gradient, preserves the bf16 rounding boundaries, and emits the channel reductions plus full channels-last output from one planned template."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partial_reduce_kernel(
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    DIVISOR: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_lanes = tl.arange(0, BLOCK_K)
    k = tl.program_id(1) * GROUP_K + k_lanes
    mask = (
        (k_lanes[:, None] < GROUP_K)
        & (k[:, None] < K_TOTAL)
        & (c[None, :] < C)
    )
    n = k // SPATIAL
    offsets = k[:, None] * C + c[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.load(pooled_ptr + n[:, None] * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < C, other=0.0).to(tl.float32)

    divisor = tl.full((BLOCK_K, BLOCK_C), DIVISOR, tl.float32)
    pool_grad = _f32_div(pooled, divisor).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    centered = _f32_sub(x, mean[None, :])
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd[None, :]), weight[None, :]), bias[None, :])
    affine = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    sigmoid = tl.sigmoid(affine)
    grad_silu = _f32_mul(
        _f32_mul(pool_grad, sigmoid),
        _f32_add(_f32_mul(affine, _f32_sub(1.0, sigmoid)), 1.0),
    )
    grad_silu = grad_silu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    active = tl.where(mask, grad_silu, 0.0)
    dot = tl.where(mask, _f32_mul(grad_silu, centered), 0.0)
    partial_offsets = c * NUM_K_TILES + tl.program_id(1)
    tl.store(partial_sum_ptr + partial_offsets, tl.sum(active, axis=0), mask=c < C)
    tl.store(partial_dot_ptr + partial_offsets, tl.sum(dot, axis=0), mask=c < C)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    dot_tmp_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_K_TILES
    offsets = c * NUM_K_TILES + tiles
    sum_value = tl.sum(tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    dot_value = tl.sum(tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    sum_ptr,
    dot_ptr,
    grad_input_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    DIVISOR: tl.constexpr,
    REDUCE_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = offsets % C
    k = offsets // C
    n = k // SPATIAL

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.load(pooled_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=mask, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=mask, other=0.0).to(tl.float32)

    divisor = tl.full((BLOCK,), DIVISOR, tl.float32)
    pool_grad = _f32_div(pooled, divisor).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    centered = _f32_sub(x, mean)
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)
    affine = affine.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    sigmoid = tl.sigmoid(affine)
    grad_silu = _f32_mul(
        _f32_mul(pool_grad, sigmoid),
        _f32_add(_f32_mul(affine, _f32_sub(1.0, sigmoid)), 1.0),
    )
    grad_silu = grad_silu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    reduce_scale = tl.full((), REDUCE_SCALE, tl.float32)
    mean_term = _f32_mul(sum_value, reduce_scale)
    var_term = _f32_mul(_f32_mul(_f32_mul(dot_value, reduce_scale), invstd), invstd)
    input_scale = _f32_mul(invstd, weight)
    correction = _f32_sub(_f32_sub(grad_silu, _f32_mul(centered, var_term)), mean_term)
    grad_input = _f32_mul(correction, input_scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _run_oracle(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
):
    pooled, x, mean, invstd, weight, bias = inputs[:6]
    sum_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    scaled_dot_out = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    grad_input = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16)

    spatial = H * W
    k_total = 128 * spatial
    total = k_total * C
    num_k_tiles = triton.cdiv(k_total, GROUP_K)
    partial_sum = torch.empty_strided((C, num_k_tiles), (num_k_tiles, 1), device=x.device, dtype=torch.float32)
    partial_dot = torch.empty_strided((C, num_k_tiles), (num_k_tiles, 1), device=x.device, dtype=torch.float32)

    _partial_reduce_kernel[(triton.cdiv(C, BLOCK_C), num_k_tiles)](
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        partial_sum,
        partial_dot,
        C=C,
        SPATIAL=spatial,
        DIVISOR=64,
        K_TOTAL=k_total,
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
        invstd,
        sum_out,
        scaled_dot_out,
        dot_tmp,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=_next_power_of_2(num_k_tiles),
        num_warps=8,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        sum_out,
        dot_tmp,
        grad_input,
        TOTAL=total,
        C=C,
        SPATIAL=spatial,
        DIVISOR=64,
        REDUCE_SCALE=0.0001220703125,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return sum_out, scaled_dot_out, grad_input


@oracle_impl(hardware="B200", point="79fb3aff", C=640, H=8, W=8, GROUP_K=256, BLOCK_K=256, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=8)
@oracle_impl(hardware="B200", point="a44186d7", C=1280, H=7, W=7, GROUP_K=256, BLOCK_K=256, BLOCK_C=16, EPILOGUE_BLOCK=256, num_warps=8)
def oracle_forward(
    inputs,
    *,
    C: int,
    H: int,
    W: int,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
):
    return _run_oracle(
        inputs,
        C=C,
        H=H,
        W=W,
        GROUP_K=GROUP_K,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
        num_warps=num_warps,
    )
