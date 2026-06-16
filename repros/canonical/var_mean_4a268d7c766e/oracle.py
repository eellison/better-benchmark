"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistillGPT2 training embedding/dropout/LayerNorm scope in one shape-specialized Triton row kernel, including token embedding, generated position ids and position embedding, the all-false adjacent-position bool side output, internally generated `prims.inductor_seeds.default(13)`, seed-index-0 f32 dropout comparison, returned mask and scaled pre-normalization f32 tensor, population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt, returned mean and rsqrt tensors, final affine f32 computation, bf16 `[16384,768]` conversion, and returned `[768,16384]` permute alias, whereas Inductor lowers the indexed embedding producers, deterministic mask construction, generated RNG/dropout, row reduction, affine epilogue, cast, and alias-producing layout return through generic scheduler boundaries; Inductor cannot do this today because its normalization templates do not canonicalize DistillGPT2 token-plus-position embedding with internally seeded pre-norm dropout and multiple observable side outputs into one fixed-hidden row plan; the fix is NEW_PATTERN: add a GPT-style embedding-dropout-LayerNorm template that folds the gathers, RNG mask, row statistics, affine/cast epilogue, side-output stores, and layout alias return into one full-scope schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
SEED_COUNT = 13
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
def _distillgpt2_embedding_dropout_layernorm_kernel(
    word_table_ptr,
    token_ids_ptr,
    position_table_ptr,
    weight_ptr,
    bias_ptr,
    rng_ptr,
    position_ids_ptr,
    embedding_ptr,
    position_embedding_ptr,
    ne_ptr,
    mask_ptr,
    dropped_ptr,
    mean_ptr,
    rsqrt_ptr,
    bf16_ptr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    SEED_INDEX_: tl.constexpr,
    DROPOUT_P_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    EPS_: tl.constexpr,
    BLOCK_H: tl.constexpr,
    USE_SEEDED_RNG: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    col_mask = cols < HIDDEN_
    seq = row % SEQ_
    offsets = row * HIDDEN_ + cols
    position_offsets = seq * HIDDEN_ + cols

    token_id = tl.load(token_ids_ptr + row)
    token = tl.load(
        word_table_ptr + token_id * HIDDEN_ + cols,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + position_offsets,
        mask=col_mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)

    if USE_SEEDED_RNG:
        seed = tl.load(rng_ptr + SEED_INDEX_)
        random = tl.rand(seed, offsets.to(tl.uint32))
    else:
        random = tl.load(
            rng_ptr + offsets,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

    embedded = _f32_add(token, position)
    keep = random > DROPOUT_P_
    masked = tl.where(keep, embedded, 0.0)
    dropped = _f32_mul(masked, DROPOUT_SCALE_)

    tl.store(embedding_ptr + offsets, token, mask=col_mask)
    tl.store(mask_ptr + offsets, keep, mask=col_mask)
    tl.store(dropped_ptr + offsets, dropped, mask=col_mask)
    tl.store(ne_ptr + row, False)
    if row < SEQ_:
        tl.store(position_ids_ptr + seq, seq.to(tl.int64))
        tl.store(position_embedding_ptr + position_offsets, position, mask=col_mask)

    reduce_values = tl.where(col_mask, dropped, 0.0)
    mean = tl.sum(reduce_values, axis=0) / HIDDEN_
    centered = _f32_sub(dropped, mean)
    centered_for_var = tl.where(col_mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=0) / HIDDEN_
    rsqrt = libdevice.rsqrt(_f32_add(variance, EPS_))

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
    normalized = _f32_mul(centered, rsqrt)
    affine = _f32_add(_f32_mul(normalized, weight), bias)

    tl.store(mean_ptr + row, mean)
    tl.store(rsqrt_ptr + row, rsqrt)
    tl.store(
        bf16_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=col_mask,
    )


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


def _resolve_shape(shape, numel):
    resolved = []
    known = 1
    unknown = -1
    for idx, dim in enumerate(shape):
        dim = int(dim)
        if dim == -1:
            unknown = idx
            resolved.append(1)
        else:
            known *= dim
            resolved.append(dim)
    if unknown >= 0:
        resolved[unknown] = int(numel) // known
    return tuple(resolved)


@oracle_impl(hardware="B200", point="5ffeb263", BLOCK_H=1024, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    num_warps: int,
    num_stages: int,
):
    word_table, token_ids, position_table, weight, bias, _shape0, random_shape, flat_shape = inputs
    rand_shape = _shape_tuple(random_shape)
    flat_shape = _resolve_shape(flat_shape, ROWS * HIDDEN)
    device = word_table.device

    position_ids = torch.empty((1, SEQ), device=device, dtype=torch.int64)
    embedding = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    position_embedding = torch.empty((1, SEQ, HIDDEN), device=device, dtype=torch.float32)
    ne = torch.empty((BATCH, SEQ), device=device, dtype=torch.bool)
    gt = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.bool)
    dropped = torch.empty((BATCH, SEQ, HIDDEN), device=device, dtype=torch.float32)
    mean = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    rsqrt = torch.empty((BATCH, SEQ, 1), device=device, dtype=torch.float32)
    bf16 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    grid = (ROWS,)
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _distillgpt2_embedding_dropout_layernorm_kernel[grid](
            word_table,
            token_ids,
            position_table,
            weight,
            bias,
            seeds,
            position_ids,
            embedding,
            position_embedding,
            ne,
            gt,
            dropped,
            mean,
            rsqrt,
            bf16,
            HIDDEN_=HIDDEN,
            SEQ_=SEQ,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            USE_SEEDED_RNG=True,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seeds, random = _seeds_and_random_for_eager_check(rand_shape, device=device)
        _distillgpt2_embedding_dropout_layernorm_kernel[grid](
            word_table,
            token_ids,
            position_table,
            weight,
            bias,
            random,
            position_ids,
            embedding,
            position_embedding,
            ne,
            gt,
            dropped,
            mean,
            rsqrt,
            bf16,
            HIDDEN_=HIDDEN,
            SEQ_=SEQ,
            SEED_INDEX_=SEED_INDEX,
            DROPOUT_P_=DROPOUT_P,
            DROPOUT_SCALE_=DROPOUT_SCALE,
            EPS_=EPS,
            BLOCK_H=BLOCK_H,
            USE_SEEDED_RNG=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return (
        embedding,
        position_ids,
        position_embedding,
        ne,
        seeds,
        gt,
        dropped,
        mean,
        rsqrt,
        bf16,
        bf16.permute(1, 0),
    )
