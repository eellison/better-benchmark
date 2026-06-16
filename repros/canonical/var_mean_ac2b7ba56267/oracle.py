"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer training embedding-LayerNorm-dropout scope in one Triton row kernel, including the returned zero `full` tensor, mask-derived position ids, word/position/global embedding gathers, fp32 hidden-size-768 population `var_mean(correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned pre-affine normalized tensor, affine dropout with internally generated `prims.inductor_seeds` seed-index-0 RNG, returned bool mask, returned scaled f32 dropout tensor, and `rsqrt / 768` side output, whereas Inductor lowers the integer position-id construction, multiple embedding gathers, row normalization, stochastic epilogue, and sibling output stores through generic scheduler fragments; Inductor cannot do this today because its norm-template matcher does not canonicalize Longformer dynamic position embedding assembly with internally seeded dropout and all observable side outputs as one fixed-hidden semantic lowering; the fix is NEW_PATTERN: add a guarded Longformer embedding-LayerNorm-dropout template that folds position-id construction, indexed embedding loads, row statistics, affine dropout, RNG mask materialization, and inverse-std side storage into one full-scope row plan."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


SEED_COUNT = 1
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
def _longformer_embedding_layernorm_dropout_kernel(
    cumsum_ptr,
    position_mask_ptr,
    word_table_ptr,
    word_ids_ptr,
    position_table_ptr,
    global_table_ptr,
    weight_ptr,
    bias_ptr,
    rng_ptr,
    full_ptr,
    position_ids_ptr,
    normalized_ptr,
    gt_ptr,
    dropout_ptr,
    div_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    USE_RANDOM_PTR: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    cumsum_i32 = tl.load(cumsum_ptr + row_ids, mask=row_mask, other=0).to(tl.int32)
    position_mask = tl.load(position_mask_ptr + row_ids, mask=row_mask, other=0)
    position_id = (cumsum_i32 * position_mask).to(tl.int64) + 1
    word_id = tl.load(word_ids_ptr + row_ids, mask=row_mask, other=0)

    tl.store(full_ptr + row_ids, tl.full((ROW_BLOCK,), 0, tl.int64), mask=row_mask)
    tl.store(position_ids_ptr + row_ids, position_id, mask=row_mask)

    word = tl.load(
        word_table_ptr + word_id[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    global_token = tl.load(
        global_table_ptr + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x = _f32_add(_f32_add(word, position), global_token[None, :])
    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1) / HIDDEN
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN
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

    if USE_RANDOM_PTR:
        random = tl.load(
            rng_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed = tl.load(rng_ptr + SEED_IDX)
        random = tl.rand(seed, offsets.to(tl.uint32))

    keep = random > DROPOUT_P_C
    dropped = _f32_mul(keep.to(tl.float32), affine)
    scaled = _f32_mul(dropped, DROPOUT_SCALE_C)

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(gt_ptr + offsets, keep, mask=mask)
    tl.store(dropout_ptr + offsets, scaled, mask=mask)
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


def _random_for_eager_check(shape, *, device):
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
    return random


# 496feaec: Longformer train embedding + LayerNorm + generated dropout, [8,1024,768]
@oracle_impl(
    hardware="B200",
    point="496feaec",
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
        arg5_1,
        arg6_1,
        arg7_1,
        shape0,
        shape1,
    ) = inputs
    full_shape = _as_shape(shape0)
    tensor_shape = _as_shape(shape1)
    div_shape = full_shape + (1,)
    rows = int(full_shape[0] * full_shape[1])
    hidden = int(tensor_shape[2])
    device = arg2_1.device

    full = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=device,
        dtype=torch.int64,
    )
    position_ids = torch.empty_strided(
        full_shape,
        _contiguous_stride(full_shape),
        device=device,
        dtype=torch.int64,
    )
    normalized = torch.empty_strided(
        tensor_shape,
        _contiguous_stride(tensor_shape),
        device=device,
        dtype=torch.float32,
    )
    gt = torch.empty_strided(
        tensor_shape,
        _contiguous_stride(tensor_shape),
        device=device,
        dtype=torch.bool,
    )
    dropout = torch.empty_strided(
        tensor_shape,
        _contiguous_stride(tensor_shape),
        device=device,
        dtype=torch.float32,
    )
    div = torch.empty_strided(
        div_shape,
        _contiguous_stride(div_shape),
        device=device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(rows, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _longformer_embedding_layernorm_dropout_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            seeds,
            full,
            position_ids,
            normalized,
            gt,
            dropout,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            SEED_IDX=SEED_INDEX,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        random = _random_for_eager_check(tensor_shape, device=device)
        _longformer_embedding_layernorm_dropout_kernel[grid](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            arg4_1,
            arg5_1,
            arg6_1,
            arg7_1,
            random,
            full,
            position_ids,
            normalized,
            gt,
            dropout,
            div,
            ROWS=rows,
            HIDDEN=hidden,
            SEED_IDX=SEED_INDEX,
            DROPOUT_P_C=DROPOUT_P,
            DROPOUT_SCALE_C=DROPOUT_SCALE,
            EPS_C=EPS,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            USE_RANDOM_PTR=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return full, position_ids, normalized, gt, dropout, div
