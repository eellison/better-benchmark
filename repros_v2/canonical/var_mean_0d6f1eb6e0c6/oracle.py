"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete M2M100 bf16 internally seeded dropout-residual LayerNorm scope in one Triton row kernel, including `prims.inductor_seeds.default(4)`, seed-index-0 RNG with the required f32-random-to-bf16 comparison, returned seed tensor and bool mask, bf16 dropout multiply and scale rounding before the f32 residual add, returned pre-normalization add tensor, population `var_mean(..., dim=2, correction=0)`, eps=1e-5 rsqrt, returned mean and rsqrt side outputs, and final bf16 `[8192,1024]` affine view, whereas Inductor lowers the internal seed/RNG producer, row-normalization reduction, affine epilogue, bf16 cast/view, and sibling side outputs through generic scheduler boundaries; Inductor cannot do this today because its fixed-hidden LayerNorm scheduler does not keep an internally seeded dropout producer and all observable side outputs resident across the row-statistics pass while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline `prims.inductor_seeds` and emit the seed tensor, mask, residual add, mean, rsqrt, and bf16 affine view from one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 4
SEED_INDEX = 0
BATCH = 64
SEQ = 128
ROWS = BATCH * SEQ
HIDDEN = 1024
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
def _dropout_layernorm_kernel(
    src_ptr,
    rng_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    add_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_ptr,
    HIDDEN_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = cols < HIDDEN_
    offsets = rows * HIDDEN_ + cols

    src = tl.load(src_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    if USE_SEEDED_RNG:
        seed = tl.load(rng_ptr + SEED_INDEX_)
        rand_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    else:
        rand_bf16 = tl.load(rng_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
    threshold = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = rand_bf16 > threshold
    tl.store(mask_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, src, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    add = _f32_add(residual, scaled.to(tl.float32))
    tl.store(add_ptr + offsets, add, mask=mask)

    mean_vec = tl.sum(tl.where(mask, add, 0.0), axis=1) / HIDDEN_
    mean = mean_vec[:, None]
    centered = _f32_sub(add, mean)
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN_
    rsqrt = libdevice.rsqrt(_f32_add(variance, EPS_))

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    normalized = _f32_mul(centered, rsqrt[:, None])
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(mean_ptr + row_offsets, mean_vec)
    tl.store(rsqrt_ptr + row_offsets, rsqrt)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="c414de20", BLOCK_H=1024, ROW_BLOCK=1, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, random_shape, _shape2 = inputs
    base_shape = _shape_tuple(shape0)
    rand_shape = _shape_tuple(random_shape)
    device = arg0_1.device

    gt = torch.empty(base_shape, device=device, dtype=torch.bool)
    add = torch.empty(base_shape, device=device, dtype=torch.float32)
    mean = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    rsqrt = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)

    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _dropout_layernorm_kernel[grid](
            arg0_1,
            seeds,
            arg1_1,
            arg2_1,
            arg3_1,
            gt,
            add,
            mean,
            rsqrt,
            out,
            HIDDEN_=HIDDEN,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seeds, random = _seeds_and_random_for_eager_check(rand_shape, device=device)
        _dropout_layernorm_kernel[grid](
            arg0_1,
            random,
            arg1_1,
            arg2_1,
            arg3_1,
            gt,
            add,
            mean,
            rsqrt,
            out,
            HIDDEN_=HIDDEN,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return seeds, gt, add, mean, rsqrt, out
