"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Electra f32 embedding LayerNorm plus generated-dropout training scope in one Triton row kernel, including the gathered token-type id output, word/token-type/position embedding gathers, fp32 hidden-size-128 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, returned normalized f32 tensor, affine epilogue, internally generated `prims.inductor_seeds.default(37)` seed-index-0 f32 dropout, returned seed tensor, returned bool mask, final bf16 `[32768,128]` view, and `rsqrt / 128` side output, whereas Inductor lowers the gather/embedding producers, row-statistics reduction, generated RNG/dropout, bf16 cast/view, and sibling outputs through generic scheduler fragments; Inductor cannot fuse this full returned-output envelope today because its normalization template does not recognize Electra's gathered token-type embedding assembly with internally seeded dropout and all observable side outputs as one fixed-hidden row plan while preserving f32 RNG and dtype boundaries; the fix is NEW_PATTERN: add a guarded Electra embedding-LayerNorm-dropout lowering that folds gathered embedding producers, row statistics, affine dropout, seed/mask materialization, bf16 view storage, and inverse-std side storage into one full-scope schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 64
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 128
SEED_COUNT = 37
SEED_INDEX = 0
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
def _electra_embedding_layernorm_dropout_kernel(
    token_type_source_ptr,
    position_ids_ptr,
    word_table_ptr,
    word_ids_ptr,
    token_type_table_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    seeds_or_random_ptr,
    gathered_ptr,
    normalized_ptr,
    keep_ptr,
    bf16_view_ptr,
    div_ptr,
    ROWS_C: tl.constexpr,
    SEQ_LEN_C: tl.constexpr,
    HIDDEN_C: tl.constexpr,
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
    row_mask = row_ids < ROWS_C
    col_mask = cols < HIDDEN_C
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = row_ids[:, None] * HIDDEN_C + cols[None, :]

    seq = row_ids - (row_ids // SEQ_LEN_C) * SEQ_LEN_C
    position_id = tl.load(position_ids_ptr + seq, mask=row_mask, other=0)
    token_type_id = tl.load(token_type_source_ptr + position_id, mask=row_mask, other=0)
    word_id = tl.load(word_ids_ptr + row_ids, mask=row_mask, other=0)

    tl.store(gathered_ptr + seq, token_type_id, mask=row_ids < SEQ_LEN_C)

    word = tl.load(
        word_table_ptr + word_id[:, None] * HIDDEN_C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + token_type_id[:, None] * HIDDEN_C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_id[:, None] * HIDDEN_C + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    x = _f32_add(word, token_type)
    x = _f32_add(x, position)
    reduce_values = tl.where(mask, x, 0.0)
    mean = tl.sum(reduce_values, axis=1) / HIDDEN_C
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) / HIDDEN_C
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
            seeds_or_random_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
    else:
        seed = tl.load(seeds_or_random_ptr + SEED_IDX)
        random = tl.rand(seed, offsets.to(tl.uint32))

    keep = random > DROPOUT_P_C
    dropped = _f32_mul(keep.to(tl.float32), affine)
    scaled = _f32_mul(dropped, DROPOUT_SCALE_C)

    tl.store(normalized_ptr + offsets, normalized, mask=mask)
    tl.store(keep_ptr + offsets, keep, mask=mask)
    tl.store(bf16_view_ptr + offsets, scaled.to(tl.bfloat16), mask=mask)
    tl.store(div_ptr + row_ids, invstd / HIDDEN_C, mask=row_mask)


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


# d211188d: Electra train embedding LayerNorm + generated dropout, hidden=128.
@oracle_impl(
    hardware="B200",
    point="d211188d",
    BLOCK_H=128,
    ROW_BLOCK=8,
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
        token_type_source,
        position_ids,
        word_table,
        word_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        _expand_shape,
        random_shape,
        flat_shape,
    ) = inputs
    random_shape = _as_shape(random_shape)
    flat_shape = _as_shape(flat_shape)
    device = word_table.device

    gathered = torch.empty_strided(
        (1, SEQ_LEN),
        (SEQ_LEN, 1),
        device=device,
        dtype=torch.int64,
    )
    normalized = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.float32,
    )
    keep = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    bf16_view = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _electra_embedding_layernorm_dropout_kernel[grid](
            token_type_source,
            position_ids,
            word_table,
            word_ids,
            token_type_table,
            position_table,
            weight,
            bias,
            seeds,
            gathered,
            normalized,
            keep,
            bf16_view,
            div,
            ROWS_C=ROWS,
            SEQ_LEN_C=SEQ_LEN,
            HIDDEN_C=HIDDEN,
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
        seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _electra_embedding_layernorm_dropout_kernel[grid](
            token_type_source,
            position_ids,
            word_table,
            word_ids,
            token_type_table,
            position_table,
            weight,
            bias,
            random,
            gathered,
            normalized,
            keep,
            bf16_view,
            div,
            ROWS_C=ROWS,
            SEQ_LEN_C=SEQ_LEN,
            HIDDEN_C=HIDDEN,
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

    return gathered, normalized, seeds, keep, bf16_view, div
