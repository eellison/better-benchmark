"""cuTile port of var_mean_341e6399899d: Electra embedding LayerNorm + dropout.

Pre-generates seeds/random via torch.ops.prims.inductor_random OUTSIDE the kernel
(since cuTile lacks tl.rand). Then a single cuTile row kernel does the embedding
gathers, LayerNorm, and dropout scaling. HIDDEN=128 is a power of 2.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

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


@ct.kernel
def _electra_embedding_layernorm_dropout_kernel(
    token_type_source_ptr,  # i64 [SEQ_LEN]  (arg0_1 squeezed)
    position_ids_ptr,       # i64 [SEQ_LEN]  (arg1_1 squeezed)
    word_table_ptr,         # f32 [30522, HIDDEN]
    word_ids_ptr,           # i64 [BATCH, SEQ_LEN]  (arg3_1)
    token_type_table_ptr,   # f32 [2, HIDDEN]
    position_table_ptr,     # f32 [512, HIDDEN]
    weight_ptr,             # f32 [HIDDEN]
    bias_ptr,               # f32 [HIDDEN]
    random_ptr,             # f32 [BATCH, SEQ_LEN, HIDDEN]  (precomputed)
    gathered_ptr,           # i64 [SEQ_LEN]
    normalized_ptr,         # f32 [BATCH, SEQ_LEN, HIDDEN]
    keep_ptr,               # b8  [BATCH, SEQ_LEN, HIDDEN]
    bf16_view_ptr,          # bf16 [BATCH*SEQ_LEN, HIDDEN]
    div_ptr,                # f32 [BATCH, SEQ_LEN, 1]
    ROWS_C: ct.Constant[int],
    SEQ_LEN_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    DROPOUT_P_C: ct.Constant[float],
    DROPOUT_SCALE_C: ct.Constant[float],
    EPS_C: ct.Constant[float],
):
    """Grid: (ROWS, 1, 1). One row per program."""
    row = ct.bid(0)
    seq = row - (row // SEQ_LEN_C) * SEQ_LEN_C
    batch = row // SEQ_LEN_C

    position_id = ct.load(position_ids_ptr, index=(seq,), shape=(1,))
    token_type_id = ct.gather(token_type_source_ptr, position_id)
    word_id = ct.load(word_ids_ptr, index=(batch, seq), shape=(1, 1))
    word_id_1d = ct.reshape(word_id, (1,))

    # Store gathered token_type_id at gathered_ptr[seq] but only for batch == 0.
    # We do this branch-free by writing on every batch (idempotent, same value).
    ct.store(gathered_ptr, index=(seq,), tile=token_type_id)

    # Gather embeddings using 2D gather.
    cols = ct.arange(HIDDEN_C, dtype=ct.int64)
    # word_id: (1,) -> broadcast to (1, HIDDEN)
    word_id_2d = ct.reshape(word_id_1d, (1, 1))
    cols_2d = ct.reshape(cols, (1, HIDDEN_C))
    word_row_idx = ct.broadcast_to(word_id_2d, (1, HIDDEN_C))
    col_idx = ct.broadcast_to(cols_2d, (1, HIDDEN_C))

    word_val = ct.gather(word_table_ptr, (word_row_idx, col_idx))
    tt_row_idx = ct.broadcast_to(ct.reshape(token_type_id, (1, 1)), (1, HIDDEN_C))
    tt_val = ct.gather(token_type_table_ptr, (tt_row_idx, col_idx))
    pos_row_idx = ct.broadcast_to(ct.reshape(position_id, (1, 1)), (1, HIDDEN_C))
    pos_val = ct.gather(position_table_ptr, (pos_row_idx, col_idx))

    x = word_val + tt_val + pos_val  # (1, HIDDEN)
    x_f = ct.astype(x, ct.float32)

    mean = ct.sum(x_f) / HIDDEN_C  # scalar
    centered = x_f - mean
    variance = ct.sum(centered * centered) / HIDDEN_C
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(HIDDEN_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    bias_2d = ct.reshape(bias, (1, HIDDEN_C))
    affine = normalized * weight_2d + bias_2d

    random = ct.load(random_ptr, index=(batch, seq, 0), shape=(1, 1, HIDDEN_C))
    random_2d = ct.reshape(random, (1, HIDDEN_C))
    keep = random_2d > DROPOUT_P_C
    keep_f = ct.astype(keep, ct.float32)
    dropped = keep_f * affine
    scaled = dropped * DROPOUT_SCALE_C
    scaled_bf = ct.astype(scaled, ct.bfloat16)

    # Store to (BATCH, SEQ_LEN, HIDDEN)
    ct.store(normalized_ptr, index=(batch, seq, 0), tile=ct.reshape(normalized, (1, 1, HIDDEN_C)))
    ct.store(keep_ptr, index=(batch, seq, 0), tile=ct.reshape(keep, (1, 1, HIDDEN_C)))
    ct.store(bf16_view_ptr, index=(row, 0), tile=scaled_bf)
    ct.store(div_ptr, index=(batch, seq, 0),
             tile=ct.reshape(invstd / HIDDEN_C, (1, 1, 1)))


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
    return int.from_bytes(bytes(state[start:start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start:start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _random_advance(shape):
    numel = 1
    for d in shape:
        numel *= int(d)
    return (numel + 131071) // 131072


def _seeds_and_random(shape, *, device):
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


@oracle_impl(hardware="B200", point="d211188d", BLOCK_H=128, ROW_BLOCK=8)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    (
        token_type_source, position_ids, word_table, word_ids,
        token_type_table, position_table, weight, bias,
        _expand_shape, random_shape, flat_shape,
    ) = inputs

    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = _shape_tuple(random_shape)  # (64, 512, 128)
    flat_shape = _shape_tuple(flat_shape)  # (32768, 128)
    device = word_table.device

    gathered = torch.empty_strided(
        (1, SEQ_LEN), (SEQ_LEN, 1), device=device, dtype=torch.int64
    )
    normalized = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        random_shape, _contiguous_stride(random_shape),
        device=device, dtype=torch.bool,
    )
    bf16_view = torch.empty_strided(
        flat_shape, _contiguous_stride(flat_shape),
        device=device, dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1), (SEQ_LEN, 1, 1),
        device=device, dtype=torch.float32,
    )

    seeds, random = _seeds_and_random(random_shape, device=device)

    tt_src_1d = token_type_source.view(SEQ_LEN)
    pos_ids_1d = position_ids.view(SEQ_LEN)
    gathered_1d = gathered.view(SEQ_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _electra_embedding_layernorm_dropout_kernel,
        (
            tt_src_1d, pos_ids_1d, word_table, word_ids,
            token_type_table, position_table, weight, bias, random,
            gathered_1d, normalized, keep, bf16_view, div,
            ROWS, SEQ_LEN, HIDDEN, DROPOUT_P, DROPOUT_SCALE, EPS,
        ),
    )

    return gathered, normalized, seeds, keep, bf16_view, div
