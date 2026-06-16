"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MegatronBERT seeded-dropout residual LayerNorm forward scope in one Triton row-normalization kernel, including the seed-index-4 Inductor RNG with the required f32-random-to-bf16 comparison, the returned bool dropout mask, bf16 dropout scaling before the f32 residual add, population `var_mean(..., dim=2, correction=0)`, eps=1e-12 rsqrt, returned residual-add tensor, returned normalized tensor, bf16 affine output view, and returned `rsqrt / 1024` side output, whereas Inductor lowers the stochastic dropout producer, residual add, row statistics, affine epilogue, and visible side outputs through generic RNG, normalization, and pointwise scheduling regions; Inductor cannot do this today because its normalization pattern library does not recognize a seeded-dropout residual LayerNorm template that preserves all observable RNG, dtype-rounding, and side-output boundaries; the fix is NEW_PATTERN: add a guarded stochastic LayerNorm lowering that threads Inductor RNG through the row reduction and emits the mask, residual add, normalized tensor, bf16 affine view, and inverse-std side output from one full-scope plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 4
BATCH = 16
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 1024
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


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
def _dropout_layernorm_random_kernel(
    x_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    out_ptr,
    div_ptr,
    HIDDEN_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN_ + cols

    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.bfloat16)
    random_bf16 = tl.load(
        random_ptr + offsets,
        eviction_policy="evict_first",
    ).to(tl.bfloat16)
    threshold = tl.full((BLOCK_H,), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(gt_ptr + offsets, keep)

    dropped = tl.where(keep, x, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    residual = tl.load(residual_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    add = _f32_add(residual, scaled.to(tl.float32))
    tl.store(add_ptr + offsets, add)

    mean = tl.sum(add, axis=0) / HIDDEN_
    centered = _f32_sub(add, mean)
    variance = tl.sum(_f32_mul(centered, centered), axis=0) / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))

    normalized = _f32_mul(centered, invstd)
    tl.store(norm_ptr + offsets, normalized)

    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"))
    tl.store(div_ptr + row, invstd / HIDDEN_)


@triton.jit
def _dropout_layernorm_seeded_kernel(
    x_ptr,
    seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    norm_ptr,
    out_ptr,
    div_ptr,
    HIDDEN_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN_ + cols

    x = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.bfloat16)
    seed = tl.load(seeds_ptr + SEED_INDEX_)
    random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)
    threshold = tl.full((BLOCK_H,), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(gt_ptr + offsets, keep)

    dropped = tl.where(keep, x, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    residual = tl.load(residual_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    add = _f32_add(residual, scaled.to(tl.float32))
    tl.store(add_ptr + offsets, add)

    mean = tl.sum(add, axis=0) / HIDDEN_
    centered = _f32_sub(add, mean)
    variance = tl.sum(_f32_mul(centered, centered), axis=0) / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))

    normalized = _f32_mul(centered, invstd)
    tl.store(norm_ptr + offsets, normalized)

    weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"))
    tl.store(div_ptr + row, invstd / HIDDEN_)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_H=1024, num_warps=8, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, _shape2 = inputs
    random_shape = _shape_tuple(shape1)
    device = arg0_1.device

    gt = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.bool)
    add = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    normalized = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    grid = (ROWS,)
    if torch.cuda.is_current_stream_capturing():
        _dropout_layernorm_seeded_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            gt,
            add,
            normalized,
            out,
            div,
            HIDDEN_=HIDDEN,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(
            random_shape,
            seed,
            device=device,
        )
        _dropout_layernorm_random_kernel[grid](
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            arg4_1,
            gt,
            add,
            normalized,
            out,
            div,
            HIDDEN_=HIDDEN,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, add, normalized, out, div
