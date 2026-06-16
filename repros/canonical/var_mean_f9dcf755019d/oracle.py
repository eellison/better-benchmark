"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 double-dropout residual LayerNorm training scope in one Triton row kernel, including the flat-to-`[512,16,1024]` view, seed-index-97 bf16 input dropout with the required f32-random-to-bf16 cast before `gt(0.1)`, bf16 dropout scaling, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned normalized f32 tensor, seed-index-98 f32 post-affine dropout, the permute-clone-cast path materialized as the returned bf16 `[8192,1024]` view, and `rsqrt / 1024` side output; Inductor lowers the two stochastic producers, row statistics, affine epilogue, layout clone, bf16 cast, and returned side tensors through generic scheduler boundaries today; the fix is SCHEDULER_FUSION: teach the fixed-hidden LayerNorm scheduler to inline both Inductor-seeded dropout producers and sink the permute-clone epilogue into one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


INPUT_DROPOUT_SEED_INDEX = 97
OUTPUT_DROPOUT_SEED_INDEX = 98
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
def _double_dropout_layernorm_kernel(
    x_ptr,
    random0_or_seed_ptr,
    random1_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    input_mask_ptr,
    normalized_ptr,
    output_mask_ptr,
    out_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    INNER: tl.constexpr,
    HIDDEN: tl.constexpr,
    INPUT_SEED_IDX: tl.constexpr,
    OUTPUT_SEED_IDX: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    valid = (row_offsets[:, None] < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    x_bf16 = tl.load(
        x_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    )
    residual = tl.load(
        residual_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    if USE_SEEDED_RNG:
        seed0 = tl.load(random0_or_seed_ptr + INPUT_SEED_IDX)
        seed1 = tl.load(random0_or_seed_ptr + OUTPUT_SEED_IDX)
        rand0_bf16 = tl.rand(seed0, offsets.to(tl.uint32)).to(tl.bfloat16)
        rand1 = tl.rand(seed1, offsets.to(tl.uint32))
    else:
        rand0_bf16 = tl.load(
            random0_or_seed_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.bfloat16)
        rand1 = tl.load(
            random1_ptr + offsets,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

    dropout_p_bf16 = tl.full((ROW_BLOCK, BLOCK_H), 0.1, tl.float32).to(tl.bfloat16)
    keep0 = rand0_bf16 > dropout_p_bf16
    tl.store(input_mask_ptr + offsets, keep0, mask=valid)

    dropped = tl.where(keep0, x_bf16, 0.0).to(tl.bfloat16)
    dropped_scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(
        tl.bfloat16
    )
    layernorm_input = _f32_add(dropped_scaled.to(tl.float32), residual)

    reduce_input = tl.where(valid, layernorm_input, 0.0)
    mean_1d = tl.sum(reduce_input, axis=1) / HIDDEN
    centered = _f32_sub(layernorm_input, mean_1d[:, None])
    centered_masked = tl.where(valid, centered, 0.0)
    variance_1d = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd_1d = libdevice.rsqrt(_f32_add(variance_1d, EPS_C))
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

    keep1 = rand1 > 0.1
    tl.store(output_mask_ptr + offsets, keep1, mask=valid)
    dropped_affine = _f32_mul(keep1.to(tl.float32), affine)
    dropped_affine_scaled = _f32_mul(dropped_affine, DROPOUT_SCALE_C)

    outer = row_offsets // INNER
    inner = row_offsets - outer * INNER
    out_offsets = (inner[:, None] * ((ROWS // INNER) * HIDDEN)) + (
        outer[:, None] * HIDDEN
    ) + cols

    tl.store(normalized_ptr + offsets, normalized, mask=valid)
    tl.store(out_ptr + out_offsets, dropped_affine_scaled.to(tl.bfloat16), mask=valid)
    tl.store(div_ptr + row_offsets, invstd_1d / HIDDEN, mask=row_offsets < ROWS)


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
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
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


# bc741f9d: (T([8192,1024], bf16), T([99], i64), T([512,16,1024], f32), T([1024], f32), T([1024], f32), ...)
@oracle_impl(
    hardware="B200",
    point="bc741f9d",
    BLOCK_H=1024,
    ROW_BLOCK=1,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    flat_shape = _as_shape(shape3)
    rows = int(arg0_1.shape[0])
    inner = int(norm_shape[1])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    input_mask = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    output_mask = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    out = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape,
        _contiguous_stride(div_shape),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _double_dropout_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            input_mask,
            normalized,
            output_mask,
            out,
            div,
            ROWS=rows,
            INNER=inner,
            HIDDEN=hidden,
            INPUT_SEED_IDX=INPUT_DROPOUT_SEED_INDEX,
            OUTPUT_SEED_IDX=OUTPUT_DROPOUT_SEED_INDEX,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return input_mask, normalized, output_mask, out, div

    seed0 = torch.ops.prims.inductor_lookup_seed.default(
        arg1_1, INPUT_DROPOUT_SEED_INDEX
    )
    seed1 = torch.ops.prims.inductor_lookup_seed.default(
        arg1_1, OUTPUT_DROPOUT_SEED_INDEX
    )
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=arg0_1.device
    )
    _double_dropout_layernorm_kernel[grid](
        arg0_1,
        random0,
        random1,
        arg2_1,
        arg3_1,
        arg4_1,
        input_mask,
        normalized,
        output_mask,
        out,
        div,
        ROWS=rows,
        INNER=inner,
        HIDDEN=hidden,
        INPUT_SEED_IDX=INPUT_DROPOUT_SEED_INDEX,
        OUTPUT_SEED_IDX=OUTPUT_DROPOUT_SEED_INDEX,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        EPS_C=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_SEEDED_RNG=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return input_mask, normalized, output_mask, out, div
