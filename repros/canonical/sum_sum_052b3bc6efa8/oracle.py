"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Inception max-pool-backward scatter, bf16 BN/ReLU gate, two channel reductions, and dependent BN-backward tensor epilogue by building the structured scatter source once and sharing it across the reductions and final output, whereas Inductor materializes the generic scatter_add/view/bf16/channels-last clone and then schedules the masked reductions and epilogue as separate generic regions; Inductor cannot do this today because its scheduler/codegen does not recognize low-memory max-pool offsets as a structured scatter-reduce producer that can feed channel reductions while preserving bf16 cast and ReLU-mask boundaries; the fix is SCATTER_REDUCE: add a guarded max-pool-backward scatter-reduce template that fuses the scatter source with BN-backward reductions and the dependent output epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 64
POOL_H = 73
POOL_W = 73
POOL_HW = 5329
H = 147
W = 147
HW = 21609
SPATIAL = 2765952
SCALE = 3.6153917349252627e-07


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
def _zero_kernel(
    source_ptr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    tl.store(source_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@triton.jit
def _scatter_source_kernel(
    pooled_ptr,
    offsets_ptr,
    source_ptr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < NUMEL
    c = linear % 64
    ow = (linear // 64) % 73
    oh = (linear // (64 * 73)) % 73
    n = linear // (64 * 5329)

    local = tl.load(offsets_ptr + linear, mask=mask, other=0).to(tl.int32)
    kh = local // 3
    kw = local - kh * 3
    h = oh * 2 + kh
    w = ow * 2 + kw

    pooled_offsets = n * (64 * 5329) + oh * (73 * 64) + ow * 64 + c
    source_offsets = n * (64 * 21609) + h * (147 * 64) + w * 64 + c
    value = tl.load(pooled_ptr + pooled_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(source_ptr + source_offsets, value, sem="relaxed", mask=mask)


@triton.jit
def _partial_reduce_kernel(
    source_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    partial_sum0_ptr,
    partial_sum1_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < 64
    k_mask = k < 2765952
    n = k // 21609
    spatial = k - n * 21609
    h = spatial // 147
    w = spatial - h * 147
    active = k_mask[:, None] & c_mask[None, :]

    c_mat = c[None, :]
    n_mat = n[:, None]
    h_mat = h[:, None]
    w_mat = w[:, None]
    offsets = n_mat * (64 * 21609) + h_mat * (147 * 64) + w_mat * 64 + c_mat

    scatter_f32 = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    scatter = scatter_f32.to(tl.bfloat16).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    scaled = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(affine_bf16 <= 0.0, fill, scatter)
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)

    sum0 = tl.sum(where_value, axis=0)
    sum1 = tl.sum(_f32_mul(where_value, centered), axis=0)
    partial_offsets = c * NUM_TILES + tl.program_id(1)
    tl.store(partial_sum0_ptr + partial_offsets, sum0, mask=c_mask)
    tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)


@triton.jit
def _finalize_kernel(
    partial_sum0_ptr,
    partial_sum1_ptr,
    invstd_ptr,
    sum0_ptr,
    sum1_ptr,
    out0_ptr,
    out1_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles
    parts0 = tl.load(partial_sum0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    parts1 = tl.load(partial_sum1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total0 = tl.sum(parts0, axis=0)
    total1 = tl.sum(parts1, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    tl.store(sum0_ptr + c, total0)
    tl.store(sum1_ptr + c, total1)
    tl.store(out0_ptr + c, total0)
    tl.store(out1_ptr + c, _f32_mul(total1, invstd))


@triton.jit
def _epilogue_kernel(
    source_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    sum0_ptr,
    sum1_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < 64
    k_mask = k < 2765952
    n = k // 21609
    spatial = k - n * 21609
    h = spatial // 147
    w = spatial - h * 147
    active = k_mask[:, None] & c_mask[None, :]

    c_mat = c[None, :]
    n_mat = n[:, None]
    h_mat = h[:, None]
    w_mat = w[:, None]
    offsets = n_mat * (64 * 21609) + h_mat * (147 * 64) + w_mat * 64 + c_mat

    scatter_f32 = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    scatter = scatter_f32.to(tl.bfloat16).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    total0 = tl.load(sum0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    total1 = tl.load(sum1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    centered = _f32_sub(x, mean[None, :])
    normalized = _f32_mul(centered, invstd[None, :])
    scaled = _f32_mul(normalized, weight[None, :])
    affine = _f32_add(scaled, bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(affine_bf16 <= 0.0, fill, scatter)

    mean_term = _f32_mul(total0[None, :], 3.6153917349252627e-07)
    var_scaled = _f32_mul(_f32_mul(total1[None, :], 3.6153917349252627e-07), _f32_mul(invstd[None, :], invstd[None, :]))
    without_var = _f32_sub(where_value, _f32_mul(centered, var_scaled))
    without_mean = _f32_sub(without_var, mean_term)
    output_scale = _f32_mul(invstd[None, :], weight[None, :])
    out = _f32_mul(without_mean, output_scale).to(tl.bfloat16)
    tl.store(out_ptr + offsets, out, mask=active)


@oracle_impl(
    hardware="B200",
    point="f00fbabc",
    BLOCK_C=32,
    BLOCK_K=256,
    ZERO_BLOCK=1024,
    SCATTER_BLOCK=256,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_K: int,
    ZERO_BLOCK: int,
    SCATTER_BLOCK: int,
    num_warps: int,
):
    pooled, offsets, x, mean, invstd, weight, bias, fill = inputs[:8]
    num_tiles = triton.cdiv(SPATIAL, BLOCK_K)
    block_tiles = _next_power_of_2(num_tiles)

    source = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=pooled.device,
        dtype=torch.float32,
    )
    partial_sum0 = torch.empty((C, num_tiles), device=pooled.device, dtype=torch.float32)
    partial_sum1 = torch.empty((C, num_tiles), device=pooled.device, dtype=torch.float32)
    sum0 = torch.empty((C,), device=pooled.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=pooled.device, dtype=torch.float32)
    out0 = torch.empty((C,), device=pooled.device, dtype=torch.float32)
    out1 = torch.empty((C,), device=pooled.device, dtype=torch.float32)
    out2 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=pooled.device,
        dtype=torch.bfloat16,
    )

    _zero_kernel[(triton.cdiv(source.numel(), ZERO_BLOCK),)](
        source,
        NUMEL=source.numel(),
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _scatter_source_kernel[(triton.cdiv(N * C * POOL_HW, SCATTER_BLOCK),)](
        pooled,
        offsets,
        source,
        NUMEL=N * C * POOL_HW,
        BLOCK=SCATTER_BLOCK,
        num_warps=4,
    )
    _partial_reduce_kernel[(triton.cdiv(C, BLOCK_C), num_tiles)](
        source,
        x,
        mean,
        invstd,
        weight,
        bias,
        fill,
        partial_sum0,
        partial_sum1,
        NUM_TILES=num_tiles,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    _finalize_kernel[(C,)](
        partial_sum0,
        partial_sum1,
        invstd,
        sum0,
        sum1,
        out0,
        out1,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(C, BLOCK_C), num_tiles)](
        source,
        x,
        mean,
        invstd,
        weight,
        bias,
        fill,
        sum0,
        sum1,
        out2,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return out0, out1, out2
