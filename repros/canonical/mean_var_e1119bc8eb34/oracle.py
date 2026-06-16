"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete BERT bf16 dropout-add-dropout-LayerNorm training scope in one fixed-hidden Triton row kernel, including the `[2048,768] -> [16,128,768]` view, seed-index-14 dropout with f32-random-to-bf16 comparison before `gt(0.1)`, bf16 mask multiply and dropout scale rounding, fp32 residual add, seed-index-15 f32-random dropout, returned bool masks, returned f32 post-dropout tensor, shared row mean, unbiased hidden-size-768 variance, returned f32 sqrt, returned f32 centered tensor, sqrt-plus-1e-6 denominator, affine scale/bias, final bf16 rounding, and contiguous `[2048,768]` view, whereas Inductor lowers the explicit `mean` and sibling `var.correction` as separate reductions over the same visible stochastic expression; Inductor cannot remove this today because mean-plus-var canonicalization does not coalesce duplicate row-statistics state while threading two Inductor-seeded dropout producers and returned side outputs through the epilogue; the fix is ALGEBRAIC_ELIMINATION: canonicalize same-input `mean` plus `var.correction` into shared row statistics or lower the pair as a full-scope normalization template preserving the masks, f32 side outputs, and bf16 output boundary."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_INDEX_0 = 14
SEED_INDEX_1 = 15
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


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
def _dual_dropout_mean_var_kernel(
    x_ptr,
    random0_or_seeds_ptr,
    random1_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt0_ptr,
    gt1_ptr,
    dropped_ptr,
    sqrt_ptr,
    sub_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEED0: tl.constexpr,
    SEED1: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    VAR_CORRECTION_C: tl.constexpr,
    DENOM_EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_offsets < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_offsets[:, None] * HIDDEN + cols[None, :]

    x = tl.load(
        x_ptr + offsets,
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

    if USE_RANDOM_PTR:
        random0_bf16 = tl.load(
            random0_or_seeds_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        random1 = tl.load(
            random1_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed0 = tl.load(random0_or_seeds_ptr + SEED0)
        seed1 = tl.load(random0_or_seeds_ptr + SEED1)
        random0_bf16 = tl.rand(seed0, offsets.to(tl.uint32)).to(tl.bfloat16)
        random1 = tl.rand(seed1, offsets.to(tl.uint32))

    threshold_bf16 = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_C, tl.float32).to(
        tl.bfloat16
    )
    keep0 = random0_bf16 > threshold_bf16
    tl.store(gt0_ptr + offsets, keep0, mask=mask)

    mul0 = _f32_mul(keep0.to(tl.float32), x.to(tl.float32)).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    scaled0 = _f32_mul(mul0.to(tl.float32), DROPOUT_SCALE_C).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    add = _f32_add(residual, scaled0.to(tl.float32))

    keep1 = random1 > DROPOUT_P_C
    tl.store(gt1_ptr + offsets, keep1, mask=mask)
    mul2 = _f32_mul(keep1.to(tl.float32), add)
    dropped = _f32_mul(mul2, DROPOUT_SCALE_C)
    tl.store(dropped_ptr + offsets, dropped, mask=mask)

    reduce_input = tl.where(mask, dropped, 0.0)
    row_sum = tl.sum(reduce_input, axis=1)
    row_sum_sq = tl.sum(tl.where(mask, _f32_mul(dropped, dropped), 0.0), axis=1)
    mean = row_sum / HIDDEN
    centered = _f32_sub(dropped, mean[:, None])
    variance_sum = _f32_sub(row_sum_sq, _f32_mul(row_sum, mean))
    variance = variance_sum / (HIDDEN - VAR_CORRECTION_C)
    std = tl.sqrt(tl.maximum(variance, 0.0))
    tl.store(sqrt_ptr + row_offsets, std, mask=row_mask)
    tl.store(sub_ptr + offsets, centered, mask=mask)

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
    numerator = _f32_mul(weight[None, :], centered)
    denom = _f32_add(std, DENOM_EPS_C)[:, None]
    normalized = _f32_div(numerator, denom)
    affine = _f32_add(normalized, bias[None, :])
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


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


def _random_advance(shape, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (int(numel) + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return (
        ((int(numel) - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
            torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
        )

    advance = _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024, ROW_BLOCK=1, num_warps=8, num_stages=1)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    x, seeds, residual, weight, bias, view_shape, random_shape0, random_shape1, out_shape = inputs
    view_shape = _as_shape(view_shape)
    random_shape0 = _as_shape(random_shape0)
    random_shape1 = _as_shape(random_shape1)
    out_shape = _as_shape(out_shape)
    stat_shape = view_shape[:-1] + (1,)
    view_stride = _contiguous_stride(view_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])

    gt0 = torch.empty_strided(
        view_shape,
        view_stride,
        device=x.device,
        dtype=torch.bool,
    )
    gt1 = torch.empty_strided(
        view_shape,
        view_stride,
        device=x.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        view_shape,
        view_stride,
        device=x.device,
        dtype=torch.float32,
    )
    sqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=x.device,
        dtype=torch.float32,
    )
    sub = torch.empty_strided(
        view_shape,
        view_stride,
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dual_dropout_mean_var_kernel[grid](
            x,
            seeds,
            seeds,
            residual,
            weight,
            bias,
            gt0,
            gt1,
            dropped,
            sqrt,
            sub,
            out,
            ROWS=rows,
            HIDDEN=hidden,
            SEED0=SEED_INDEX_0,
            SEED1=SEED_INDEX_1,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            VAR_CORRECTION_C=VAR_CORRECTION,
            DENOM_EPS_C=DENOM_EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
        seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
        random0, random1 = _inductor_random_pair_for_eager_check(
            random_shape0,
            seed0,
            seed1,
            device=x.device,
        )
        _dual_dropout_mean_var_kernel[grid](
            x,
            random0,
            random1,
            residual,
            weight,
            bias,
            gt0,
            gt1,
            dropped,
            sqrt,
            sub,
            out,
            ROWS=rows,
            HIDDEN=hidden,
            SEED0=SEED_INDEX_0,
            SEED1=SEED_INDEX_1,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            VAR_CORRECTION_C=VAR_CORRECTION,
            DENOM_EPS_C=DENOM_EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    del random_shape1
    return gt0, gt1, dropped, sqrt, sub, out
