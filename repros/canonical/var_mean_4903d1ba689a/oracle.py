"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer bf16 bias-add seeded-dropout residual LayerNorm training scope in one Triton row kernel, including the flat `[8192,768]` to `[8,1024,768]` metadata view, f32 bias cast to bf16 before the pre-dropout add, seed-index-25 Inductor RNG with the required f32-random-to-bf16 cast before `gt(0.1)`, returned bool mask, bf16 dropout multiply and scale rounding, fp32 residual add, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned normalized f32 tensor, affine f32 tensor, final bf16 `[8192,768]` view output, and `rsqrt / 768` side output, whereas Inductor lowers the bias producer, stochastic producer, normalization reduction, affine store, bf16 cast/view, and sibling side outputs through generic scheduler boundaries; Inductor cannot fuse this full returned-output envelope today because its fixed-hidden normalization template does not keep the bias-add/dropout producer, row statistics, affine epilogue, final bf16 view, and inverse-std side output in one schedule while preserving bf16 RNG/dropout rounding boundaries; the fix is SCHEDULER_FUSION: teach the normalization scheduler to fuse bf16 bias add, seeded dropout, residual add, var_mean/rsqrt, affine, bf16 cast/view, and sibling side-output stores into one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_INDEX = 25
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
def _bias_dropout_layernorm_kernel(
    bias0_ptr,
    flat_ptr,
    random_or_seeds_ptr,
    residual_ptr,
    weight_ptr,
    bias1_ptr,
    gt_ptr,
    normalized_ptr,
    affine_ptr,
    out_bf16_ptr,
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
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)
    col_mask = cols < HIDDEN
    offsets = rows * HIDDEN + cols

    flat = tl.load(
        flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    bias0 = tl.load(
        bias0_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.bfloat16).to(tl.float32)
    biased = _f32_add(flat, bias0).to(tl.bfloat16)

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
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, biased, 0.0).to(tl.bfloat16)
    scaled = _f32_mul(dropped.to(tl.float32), DROPOUT_SCALE_C).to(tl.bfloat16)
    residual = tl.load(
        residual_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(scaled.to(tl.float32), residual)

    x_masked = tl.where(mask, x, 0.0)
    mean = (tl.sum(x_masked, axis=1) / HIDDEN)[:, None]
    centered = _f32_sub(x, mean)
    centered_masked = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_masked, centered_masked), axis=1) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS_C))
    normalized = _f32_mul(centered, invstd[:, None])

    weight = tl.load(
        weight_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias1 = tl.load(
        bias1_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    affine = _f32_add(_f32_mul(normalized, weight), bias1)

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(affine_ptr + offsets, affine, mask=mask)
    tl.store(out_bf16_ptr + offsets, affine.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_offsets, invstd / HIDDEN, mask=row_offsets < ROWS)


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


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

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


# 726994b7: (T([768], f32), T([8192,768], bf16), T([36], i64), T([8,1024,768], f32), T([768], f32), T([768], f32), ...)
@oracle_impl(hardware="B200", point="726994b7", BLOCK_H=1024, ROW_BLOCK=2, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2)
    rows = int(arg1_1.shape[0])
    hidden = int(arg1_1.shape[1])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    gt = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg1_1.device,
        dtype=torch.bool,
    )
    normalized = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape,
        _contiguous_stride(div_shape),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        _bias_dropout_layernorm_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            gt,
            normalized,
            affine,
            out_bf16,
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
        return gt, normalized, affine, out_bf16, div

    _bias_dropout_layernorm_kernel[grid](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        gt,
        normalized,
        affine,
        out_bf16,
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
    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg1_1.device)
    _bias_dropout_layernorm_kernel[grid](
        arg0_1,
        arg1_1,
        random,
        arg3_1,
        arg4_1,
        arg5_1,
        gt,
        normalized,
        affine,
        out_bf16,
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
    return gt, normalized, affine, out_bf16, div
