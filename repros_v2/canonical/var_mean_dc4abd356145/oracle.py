"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 seeded-dropout residual LayerNorm training scope in one Triton row kernel, including internally generated `inductor_seeds`, seed-index-0 RNG with f32-to-bf16 rounding before `gt(0.1)`, the returned bool mask, bf16 dropout multiply and scale rounding, fp32 residual add, population `var_mean(correction=0, keepdim=True)`, eps=1e-5 `rsqrt`, returned normalized and affine fp32 tensors, final bf16 flattened view, and `rsqrt / hidden` side output, whereas Inductor lowers the stochastic producer, residual add, row-normalization reduction, affine epilogue, bf16 cast/view, and sibling outputs through generic scheduler boundaries; Inductor cannot do this today because its norm-template scheduler does not keep an internally seeded RNG producer and all observable side outputs resident across the fixed-hidden row reduction while preserving the required bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline `prims.inductor_seeds`/seeded dropout and emit the mask, normalized tensor, affine tensor, bf16 view, and inverse-std side output from one guarded row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 2
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


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
    src_ptr,
    rng_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    norm_ptr,
    affine_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]

    src = tl.load(
        src_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_SEEDED_RNG:
        seed = tl.load(rng_ptr + SEED_IDX)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        rand_bf16 = tl.load(
            rng_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)

    threshold = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_C, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > threshold
    tl.store(mask_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, src, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(tl.bfloat16)
    x = _f32_add(residual, scaled.to(tl.float32))

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_reduce = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_reduce, centered_for_reduce), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight[None, :]), bias[None, :])

    tl.store(norm_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_ids, invstd / HIDDEN, mask=row_mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


def _launch(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    batch = int(norm_shape[0])
    tokens = int(norm_shape[1])
    norm_stride = _contiguous_stride(norm_shape)

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape,
        norm_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape,
        norm_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    bf16_view = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (batch, tokens, 1),
        (tokens, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, arg0_1.device)
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            seeds,
            arg1_1,
            arg2_1,
            arg3_1,
            gt,
            normalized,
            affine,
            bf16_view,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            SEED_IDX=SEED_INDEX,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return seeds, gt, normalized, affine, bf16_view, div

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=arg0_1.device)
    _dropout_residual_layernorm_kernel[grid](
        arg0_1,
        seeds,
        arg1_1,
        arg2_1,
        arg3_1,
        gt,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        SEED_IDX=SEED_INDEX,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_SEEDED_RNG=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _dropout_residual_layernorm_kernel[grid](
        arg0_1,
        random,
        arg1_1,
        arg2_1,
        arg3_1,
        gt,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        SEED_IDX=SEED_INDEX,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_SEEDED_RNG=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return seeds, gt, normalized, affine, bf16_view, div


@oracle_impl(hardware="B200", point="a47484d2", BLOCK_H=1024, ROW_BLOCK=1, num_warps=8, num_stages=3)
@oracle_impl(hardware="B200", point="02ced5d2", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="042c38d1", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    return _launch(
        inputs,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
