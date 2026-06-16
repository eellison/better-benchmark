"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bf16 MobileNetV3 average-pool-backward, hard-sigmoid derivative mask, and BatchNorm-backward return tuple directly from the pooled `[512, 960, 1, 1]` gradient and channels-last `[512, 960, 7, 7]` activation. Inductor currently materializes the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor, then schedules the hard-sigmoid mask, sibling channel reductions, and dependent BN-backward epilogue as generic consumers; it cannot do this today because scheduler/codegen does not represent this structured scatter/expand producer as a source-space gather-mask-reduce feeding both reductions and the full tensor epilogue while preserving bf16 cast boundaries. The fix is SCATTER_REDUCE: add a structured average-pool-backward lowering that shares the pooled source across channel reductions and the channels-last BN-backward store without materializing the scatter intermediate."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C = 960
H = 7
W = 7
HW = H * W
K_TOTAL = N * HW
TOTAL = K_TOTAL * C
DIVISOR = 49.0
REDUCE_SCALE = 3.985969387755102e-05
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
def _hsigmoid_bn_producer(pooled_bf16, x_bf16, mean, invstd, weight, bias, fill):
    pooled = pooled_bf16.to(tl.float32)
    pool_grad = _f32_div(pooled, 49.0).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    pool_grad_f32 = pool_grad.to(tl.float32)

    x = x_bf16.to(tl.float32)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    weighted = _f32_mul(normalized, weight)
    affine = _f32_add(weighted, bias).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    affine_f32 = affine.to(tl.float32)

    affine_third = _f32_div(affine_f32, 3.0)
    gate = _f32_add(affine_third, 0.5)
    middle = _f32_mul(pool_grad_f32, gate)
    producer = tl.where(affine_f32 < 3.0, middle, pool_grad_f32)
    producer = tl.where(affine_f32 <= -3.0, fill, producer)
    producer_bf16 = producer.to(tl.bfloat16, fp_downcast_rounding="rtne")
    return producer_bf16.to(tl.float32), centered


@triton.jit
def _partial_reduce_kernel(
    pooled_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
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
    mask = (
        (k_lanes[:, None] < GROUP_K)
        & (k[:, None] < 25088)
        & (c[None, :] < 960)
    )
    n = k // 49
    offsets = k[:, None] * 960 + c[None, :]

    pooled = tl.load(pooled_ptr + n[:, None] * 960 + c[None, :], mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=c < 960, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c < 960, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c < 960, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c < 960, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    producer, centered = _hsigmoid_bn_producer(
        pooled,
        x,
        mean[None, :],
        invstd[None, :],
        weight[None, :],
        bias[None, :],
        fill,
    )
    active = tl.where(mask, producer, 0.0)
    active_dot = tl.where(mask, _f32_mul(producer, centered), 0.0)

    partial_offsets = c * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(active, axis=0),
        mask=c < 960,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(active_dot, axis=0),
        mask=c < 960,
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
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_K_TILES
    offsets = c * NUM_K_TILES + tiles

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
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
    fill_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < 24084480
    c = offsets % 960
    k = offsets // 960
    n = k // 49

    pooled = tl.load(pooled_ptr + n * 960 + c, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=mask, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=mask, other=0.0).to(tl.float32)

    producer, centered = _hsigmoid_bn_producer(
        pooled,
        x,
        mean,
        invstd,
        weight,
        bias,
        fill,
    )
    mean_term = _f32_mul(sum_value, 3.985969387755102e-05)
    dot_scaled = _f32_mul(dot_value, 3.985969387755102e-05)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(dot_scaled, invstd_sq)
    input_scale = _f32_mul(invstd, weight)
    without_var = _f32_sub(producer, _f32_mul(centered, var_term))
    without_mean = _f32_sub(without_var, mean_term)
    out = _f32_mul(without_mean, input_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(out_ptr + offsets, out, mask=mask)


# timm_mobilenetv3_large_100 train, structured avg-pool backward into hard-sigmoid BN-backward.
@oracle_impl(
    hardware="B200",
    point="d90f6100",
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs[:7]
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
    grad_input = torch.empty_strided(
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
        arg6_1,
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
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        sum_out,
        dot_tmp,
        grad_input,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    return sum_out, scaled_dot, grad_input
