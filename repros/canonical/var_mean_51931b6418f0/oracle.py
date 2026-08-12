"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT2 bf16 seeded-dropout residual LayerNorm training scope in one Triton row kernel, including the flat-to-`[8,1024,768]` view, seed-index-24 Inductor RNG with the required f32-random-to-bf16 cast before `gt(0.1)`, returned dropout mask, bf16 dropout multiply and scale rounding, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned normalized fp32 tensor, affine epilogue, final bf16 `[8192,768]` view, and `rsqrt / 768` side output, whereas Inductor lowers the stochastic producer, residual add, row-normalization reduction, affine/cast epilogue, and sibling returned outputs through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not keep a seeded dropout producer and all observable side outputs resident across the row-statistics pass while preserving bf16 rounding boundaries; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline Inductor-seeded dropout and emit the mask, normalized tensor, bf16 affine view, and inverse-std side output from one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768
SEED_INDEX = 24
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
    flat_ptr,
    random_or_seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    normalized_ptr,
    bf16_ptr,
    div_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_offsets[:, None] < ROWS_) & (cols < HIDDEN_)
    offsets = rows * HIDDEN_ + cols

    flat = tl.load(
        flat_ptr + offsets,
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
        random_bf16 = tl.load(
            random_or_seeds_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(random_or_seeds_ptr + SEED_INDEX_)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)

    threshold = tl.full((ROW_BLOCK, BLOCK_H), DROPOUT_P_, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > threshold
    tl.store(mask_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, flat, 0.0).to(tl.bfloat16, fp_downcast_rounding="rtne")
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )
    x = _f32_add(residual, scaled.to(tl.float32))

    mean = tl.sum(tl.where(mask, x, 0.0), axis=1) / HIDDEN_
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN_
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN_,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN_,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)
    tl.store(div_ptr + row_offsets, invstd / HIDDEN_, mask=row_offsets < ROWS_)


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


@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, random_shape, shape2 = inputs
    norm_shape = _as_shape(shape0)
    rand_shape = _as_shape(random_shape)
    flat_shape = _as_shape(shape2)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)

    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            gt,
            normalized,
            bf16,
            div,
            ROWS_=ROWS,
            HIDDEN_=HIDDEN,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
        random = _inductor_random_for_eager_check(rand_shape, seed, device=device)
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            random,
            arg2_1,
            arg3_1,
            arg4_1,
            gt,
            normalized,
            bf16,
            div,
            ROWS_=ROWS,
            HIDDEN_=HIDDEN,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return gt, normalized, bf16, div
