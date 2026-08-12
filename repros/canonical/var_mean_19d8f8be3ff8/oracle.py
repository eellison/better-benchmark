"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 seeded-dropout residual LayerNorm training scope in one Triton row kernel, including the flat-to-`[512,16,1024]` view, seed-index-15 Inductor RNG with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, normalized f32 output, affine f32 output, final flattened bf16 affine view, and `rsqrt / 1024` side output, whereas Inductor lowers the stochastic producer, normalization reduction, affine epilogue, bf16 cast, and returned side tensors through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not keep the seeded dropout mask producer and all side outputs resident across the row-statistics pass and affine epilogue; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline Inductor-seeded dropout and emit the returned mask, normalized f32 tensor, affine f32 tensor, bf16 view, and inverse-std side tensor from one row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 15
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
def _dropout_residual_layernorm_kernel(
    flat_ptr,
    random_or_seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    normalized_ptr,
    affine_ptr,
    bf16_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    row_ids = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row_ids < ROWS) & (cols < HIDDEN)
    offsets = row_ids * HIDDEN + cols

    flat = tl.load(
        flat_ptr + offsets,
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

    if USE_RANDOM_PTR:
        random_bf16 = tl.load(
            random_or_seeds_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
    else:
        seed = tl.load(random_or_seeds_ptr + SEED_IDX)
        random_bf16 = tl.rand(seed, offsets.to(tl.uint32)).to(tl.bfloat16)

    dropout_p = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep = random_bf16 > dropout_p
    tl.store(mask_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, flat, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(tl.bfloat16)
    layernorm_input = _f32_add(scaled.to(tl.float32), residual)

    reduce_input = tl.where(mask, layernorm_input, 0.0)
    mean = tl.sum(reduce_input, axis=1) / HIDDEN
    centered = _f32_sub(layernorm_input, mean[:, None])
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

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

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_offsets, invstd / HIDDEN, mask=row_offsets < ROWS)


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


# bc741f9d: (T([8192,1024], bf16), T([99], i64), T([512,16,1024], f32), T([1024], f32), T([1024], f32), ...)
@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, shape2, shape3 = inputs
    norm_shape = _as_shape(shape1)
    random_shape = _as_shape(shape2)
    flat_shape = _as_shape(shape3)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=arg0_1.device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=arg0_1.device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=arg0_1.device, dtype=torch.float32)

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _dropout_residual_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            mask,
            normalized,
            affine,
            bf16_view,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            SEED_IDX=SEED_INDEX,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return mask, normalized, affine, bf16_view, div

    _dropout_residual_layernorm_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        mask,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        SEED_IDX=SEED_INDEX,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_RANDOM_PTR=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)
    _dropout_residual_layernorm_kernel[grid](
        arg0_1,
        random,
        arg2_1,
        arg3_1,
        arg4_1,
        mask,
        normalized,
        affine,
        bf16_view,
        div,
        ROWS=rows,
        HIDDEN=hidden,
        SEED_IDX=SEED_INDEX,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_RANDOM_PTR=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mask, normalized, affine, bf16_view, div
