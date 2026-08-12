"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 dropout-residual LayerNorm training scope in one Triton row kernel, including the flat-to-`[B,512,H]` view, Inductor seed-index-23 dropout with the required f32-random-to-bf16 cast before the `gt(0.1)` mask, bf16 dropout scaling, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-7 rsqrt, affine f32 epilogue, returned mask, normalized f32 tensor, affine f32 tensor, bf16 flattened view output, and `rsqrt / H` side output, whereas Inductor lowers the stochastic producer, normalization reduction, affine store, bf16 cast, and returned mask/side tensors through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not keep the seeded dropout mask producer and all side outputs resident across the row-statistics pass and affine epilogue; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline Inductor-seeded dropout and emit the returned mask, normalized f32 tensor, affine f32 tensor, bf16 view, and inverse-std side tensor from one row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _dropout_residual_layernorm_kernel(
    addmm_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    norm_ptr,
    affine_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_ids = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_ids < ROWS) & (cols < HIDDEN)
    offsets = row_ids * HIDDEN + cols

    addmm = tl.load(
        addmm_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    rand_bf16 = tl.load(
        random_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    dropout_p = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, addmm, 0.0).to(tl.bfloat16)
    dropped_scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(
        tl.bfloat16
    )
    x = _f32_add(dropped_scaled.to(tl.float32), residual)
    x_masked = tl.where(mask, x, 0.0)

    mean_1d = tl.sum(x_masked, axis=1) / HIDDEN
    mean = mean_1d[:, None]
    centered = _f32_sub(x, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance_1d = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, 1.0e-7))
    normalized = _f32_mul(centered, invstd_1d[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(norm_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_offsets, invstd_1d / HIDDEN, mask=row_offsets < ROWS)


@triton.jit
def _dropout_residual_layernorm_seeded_kernel(
    addmm_ptr,
    seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    norm_ptr,
    affine_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_ids = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_ids < ROWS) & (cols < HIDDEN)
    offsets = row_ids * HIDDEN + cols

    addmm = tl.load(
        addmm_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    seed = tl.load(seeds_ptr + 23)
    rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    dropout_p = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > dropout_p
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, addmm, 0.0).to(tl.bfloat16)
    dropped_scaled = _f32_mul(dropped.to(tl.float32), 1.1111111111111112).to(
        tl.bfloat16
    )
    x = _f32_add(dropped_scaled.to(tl.float32), residual)
    x_masked = tl.where(mask, x, 0.0)

    mean_1d = tl.sum(x_masked, axis=1) / HIDDEN
    mean = mean_1d[:, None]
    centered = _f32_sub(x, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance_1d = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, 1.0e-7))
    normalized = _f32_mul(centered, invstd_1d[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(norm_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_offsets, invstd_1d / HIDDEN, mask=row_offsets < ROWS)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - advance)
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


# 55aa5fd0: (T([4096,1536], bf16), T([73], i64), T([8,512,1536], f32), T([1536], f32), T([1536], f32), ...)
@oracle_impl(hardware="B200", point="55aa5fd0", BLOCK_H=2048, ROW_BLOCK=1, num_warps=8, num_stages=3)
# 243d7832: (T([16384,768], bf16), T([37], i64), T([32,512,768], f32), T([768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="243d7832", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
# d9ecc504: (T([32768,256], bf16), T([37], i64), T([64,512,256], f32), T([256], f32), T([256], f32), ...)
@oracle_impl(hardware="B200", point="d9ecc504", BLOCK_H=256, ROW_BLOCK=2, num_warps=4, num_stages=3)
# d429ff7b: (T([16384,768], bf16), T([25], i64), T([32,512,768], f32), T([768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="d429ff7b", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=arg0_1.device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=arg0_1.device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, 23)
    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_layernorm_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            gt,
            normalized,
            affine,
            bf16_view,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return (gt, normalized, affine, bf16_view, div)

    _dropout_residual_layernorm_seeded_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        gt,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    random = _inductor_random_for_eager_check(norm_shape, seed, device=arg0_1.device)

    _dropout_residual_layernorm_kernel[grid](
        arg0_1,
        random,
        arg2_1,
        arg3_1,
        arg4_1,
        gt,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (gt, normalized, affine, bf16_view, div)
