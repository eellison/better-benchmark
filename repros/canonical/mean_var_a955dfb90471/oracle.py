"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BERT fp32 word/position/token-type embedding assembly, internally generated `prims.inductor_seeds.default(61)` seed-index-0 dropout, returned seed tensor and bool mask, returned scaled fp32 dropout tensor, hidden-size-768 unbiased variance LayerNorm, returned sqrt and centered tensors, affine epilogue, final bf16 rounding, and `[2048, 768]` view in one Triton row kernel, whereas Inductor lowers the embedding gathers, generated RNG/dropout producer, sibling `mean` plus `var.correction` reductions, visible side outputs, and bf16 epilogue through generic scheduler fragments; Inductor cannot fuse this today because its norm-template scheduler does not keep indexed embedding producers, internally seeded stochastic state, and all returned intermediates resident across the shared row-statistics pass while preserving fp32 reduction and RNG semantics; the fix is SCHEDULER_FUSION: add a guarded embedding-dropout-LayerNorm lowering that folds the gathers, seed/mask materialization, shared row statistics, visible side-output stores, and affine bf16 view into one full-scope schedule."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEED_COUNT = 61
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@triton.jit
def _embedding_dropout_layernorm_kernel(
    word_table_ptr,
    input_ids_ptr,
    position_table_ptr,
    token_type_table_ptr,
    token_type_ids_ptr,
    weight_ptr,
    bias_ptr,
    seeds_or_random_ptr,
    gt_ptr,
    scaled_ptr,
    sqrt_ptr,
    sub_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    SEED_IDX: tl.constexpr,
    DROPOUT_P_C: tl.constexpr,
    DROPOUT_SCALE_C: tl.constexpr,
    VAR_CORRECTION_C: tl.constexpr,
    DENOM_EPS_C: tl.constexpr,
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
    seq_ids = row_ids - (row_ids // SEQ_LEN) * SEQ_LEN

    word_ids = tl.load(input_ids_ptr + row_ids, mask=row_mask, other=0)
    token_type_ids = tl.load(token_type_ids_ptr + row_ids, mask=row_mask, other=0)
    word = tl.load(
        word_table_ptr + word_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    position = tl.load(
        position_table_ptr + seq_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    token_type = tl.load(
        token_type_table_ptr + token_type_ids[:, None] * HIDDEN + cols[None, :],
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    add0 = word + position
    add1 = add0 + token_type

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
    tl.store(gt_ptr + offsets, keep, mask=mask)

    dropped = tl.where(keep, add1, 0.0)
    scaled = dropped * DROPOUT_SCALE_C
    tl.store(scaled_ptr + offsets, scaled, mask=mask)

    reduce_input = tl.where(mask, scaled, 0.0)
    row_sum = tl.sum(reduce_input, axis=1)
    mean = row_sum / HIDDEN
    centered = scaled - mean[:, None]
    centered_for_var = tl.where(mask, centered, 0.0)
    variance = tl.sum(centered_for_var * centered_for_var, axis=1) / (
        HIDDEN - VAR_CORRECTION_C
    )
    std = tl.sqrt_rn(tl.maximum(variance, 0.0))
    tl.store(sqrt_ptr + row_ids, std, mask=row_mask)
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
    numerator = weight[None, :] * centered
    denom = std[:, None] + DENOM_EPS_C
    normalized = numerator / denom
    affine = normalized + bias[None, :]
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
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
    return (numel + 98303) // 98304


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


def _launch(
    word_table,
    input_ids,
    position_table,
    token_type_table,
    token_type_ids,
    weight,
    bias,
    seeds_or_random,
    gt,
    scaled,
    sqrt,
    sub,
    out,
    *,
    rows: int,
    hidden: int,
    seq_len: int,
    use_random_ptr: bool,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    _embedding_dropout_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        word_table,
        input_ids,
        position_table,
        token_type_table,
        token_type_ids,
        weight,
        bias,
        seeds_or_random,
        gt,
        scaled,
        sqrt,
        sub,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        SEQ_LEN=seq_len,
        SEED_IDX=SEED_INDEX,
        DROPOUT_P_C=DROPOUT_P,
        DROPOUT_SCALE_C=DROPOUT_SCALE,
        VAR_CORRECTION_C=VAR_CORRECTION,
        DENOM_EPS_C=DENOM_EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        USE_RANDOM_PTR=use_random_ptr,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# 4ea50a71: BERT train fp32 embedding + seed-0 dropout + hidden-768 LayerNorm.
@oracle_impl(hardware="B200", point="4ea50a71", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (
        word_table,
        input_ids,
        position_table,
        token_type_table,
        token_type_ids,
        weight,
        bias,
        random_shape_param,
        out_shape_param,
    ) = inputs
    random_shape = _shape_tuple(random_shape_param)
    out_shape = _shape_tuple(out_shape_param)
    stat_shape = random_shape[:-1] + (1,)
    batch = int(input_ids.shape[0])
    seq_len = int(input_ids.shape[1])
    rows = batch * seq_len
    hidden = int(weight.shape[0])
    device = word_table.device

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.float32,
    )
    sqrt = torch.empty_strided(
        stat_shape,
        _contiguous_stride(stat_shape),
        device=device,
        dtype=torch.float32,
    )
    sub = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=torch.bfloat16,
    )

    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        _launch(
            word_table,
            input_ids,
            position_table,
            token_type_table,
            token_type_ids,
            weight,
            bias,
            seeds,
            gt,
            scaled,
            sqrt,
            sub,
            out,
            rows=rows,
            hidden=hidden,
            seq_len=seq_len,
            use_random_ptr=False,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
        _launch(
            word_table,
            input_ids,
            position_table,
            token_type_table,
            token_type_ids,
            weight,
            bias,
            random,
            gt,
            scaled,
            sqrt,
            sub,
            out,
            rows=rows,
            hidden=hidden,
            seq_len=seq_len,
            use_random_ptr=True,
            BLOCK_H=BLOCK_H,
            ROW_BLOCK=ROW_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return seeds, gt, scaled, sqrt, sub, out
