"""cuTile port of pointwise_4254ac4c0d96: XLNet token+positional dropout fanout.

Uses eager pre-generated random via torch.ops.prims.inductor_random.
Two kernels: token-embedding dropout, and positional sin/cos + dropout.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ = 512
POS_SEQ = 1024
HIDDEN = 1024
HALF_HIDDEN = 512
TOKEN_ROWS = SEQ * BATCH
POS_ROWS = POS_SEQ * BATCH
TOKEN_NUMEL = TOKEN_ROWS * HIDDEN
POS_NUMEL = POS_ROWS * HIDDEN
SEED_COUNT = 99
SEED_INDEX_TOKEN = 0
SEED_INDEX_POS = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _token_embed_dropout_kernel(
    ids_ptr,      # i64 flat [BATCH*SEQ]
    table_ptr,    # f32 flat [VOCAB*HIDDEN]
    rand_ptr,     # f32 flat [N]
    clone_ptr,    # i64 flat [SEQ*BATCH]
    mask_ptr,     # bool flat [N]
    scaled_ptr,   # f32 flat [N]
    bf16_ptr,     # bf16 flat [N]
    N: ct.Constant[int],
    SEQ_C: ct.Constant[int],
    BATCH_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # row = pid = seq*BATCH + batch (matches (SEQ, BATCH, HIDDEN) contiguous layout)
    row = pid
    batch = row % BATCH_C
    seq = row // BATCH_C
    # Fetch token id from ids[batch, seq] = ids_flat[batch*SEQ + seq]
    ids_1d_idx = ct.full((1,), batch * SEQ_C + seq, dtype=ct.int64)
    token_id_1 = ct.gather(ids_ptr, ids_1d_idx)
    # Load embedding row: table[tok, :] contiguous of length HIDDEN
    cols = ct.arange(BLOCK, dtype=ct.int64)
    tok = token_id_1  # shape (1,)
    # Broadcast tok*HIDDEN across cols
    tok_bcast = ct.full((BLOCK,), 0, dtype=ct.int64) + tok  # broadcast: shape (BLOCK,)
    emb_idx = tok_bcast * HIDDEN_C + cols
    embedding = ct.gather(table_ptr, emb_idx)
    embedding_f = ct.astype(embedding, ct.float32)

    random = ct.load(rand_ptr, index=(pid,), shape=(BLOCK,))
    keep = random > DROPOUT_P_
    keep_f = ct.astype(keep, ct.float32)
    scaled = keep_f * embedding_f * DROPOUT_SCALE_

    ct.store(mask_ptr, index=(pid,), tile=keep)
    ct.store(scaled_ptr, index=(pid,), tile=scaled)
    ct.store(bf16_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))
    # Clone: write token id at position row = seq*BATCH + batch = pid.
    scatter_idx = ct.full((1,), pid, dtype=ct.int64)
    ct.scatter(clone_ptr, scatter_idx, token_id_1)


@ct.kernel
def _positional_dropout_kernel(
    rand_ptr,        # f32 flat [POS_NUMEL]
    out_ptr,         # bf16 flat [POS_NUMEL]
    HIDDEN_C: ct.Constant[int],
    HALF_HIDDEN_C: ct.Constant[int],
    BATCH_C: ct.Constant[int],
    DROPOUT_P_: ct.Constant[float],
    DROPOUT_SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    # row = pid ; each row is HIDDEN wide
    # row = pos_index * BATCH + batch
    row = pid
    pos_index = row // BATCH_C
    # For each column in [0, HIDDEN), compute:
    # freq_index = col if col < HALF_HIDDEN else col - HALF_HIDDEN
    # pos = 512 - pos_index (bf16)
    # exponent = (freq_index * 2) / 1024
    # denom = 10000 ** exponent
    # inv_freq = (1/denom) -> bf16
    # phase = pos * inv_freq -> bf16
    # trig = sin(phase) if col < HALF else cos(phase); rounded to bf16
    # random -> bf16, keep = random > 0.1 (bf16), scaled = keep*trig*1.1111 -> bf16
    cols = ct.arange(BLOCK, dtype=ct.int32)
    freq_index = ct.where(cols < HALF_HIDDEN_C, cols, cols - HALF_HIDDEN_C)
    freq_f = ct.astype(freq_index, ct.float32)

    pos_val_f = ct.astype(ct.full((BLOCK,), 512, dtype=ct.int32) - pos_index, ct.float32)
    pos_bf16 = ct.astype(pos_val_f, ct.bfloat16)
    even_dim = freq_f * 2.0
    exponent = even_dim / 1024.0
    denom = ct.exp(exponent * ct.log(ct.full((BLOCK,), 10000.0, dtype=ct.float32)))
    inv_freq_bf16 = ct.astype(1.0 / denom, ct.bfloat16)
    phase_bf16 = ct.astype(ct.astype(pos_bf16, ct.float32) * ct.astype(inv_freq_bf16, ct.float32), ct.bfloat16)
    phase_f = ct.astype(phase_bf16, ct.float32)
    sin_v = ct.sin(phase_f)
    cos_v = ct.cos(phase_f)
    trig_f = ct.where(cols < HALF_HIDDEN_C, sin_v, cos_v)
    trig_bf16 = ct.astype(trig_f, ct.bfloat16)

    random = ct.load(rand_ptr, index=(pid,), shape=(BLOCK,))
    random_bf16 = ct.astype(random, ct.bfloat16)
    threshold_bf16 = ct.astype(ct.full((BLOCK,), DROPOUT_P_, dtype=ct.float32), ct.bfloat16)
    keep = random_bf16 > threshold_bf16
    keep_f = ct.astype(keep, ct.float32)
    dropped_bf16 = ct.astype(keep_f * ct.astype(trig_bf16, ct.float32), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE_, ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=scaled_bf16)


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


def _seeds_and_randoms_for_eager_check(shape0, shape1, *, device):
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_TOKEN)
        random0 = torch.ops.prims.inductor_random.default(shape0, seed0, "rand")
        seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_POS)
        random1 = torch.ops.prims.inductor_random.default(shape1, seed1, "rand")
        return seeds, random0, random1

    total_advance = 8 + _random_advance(shape0) + _random_advance(shape1)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_TOKEN)
    random0 = torch.ops.prims.inductor_random.default(shape0, seed0, "rand")
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_POS)
    random1 = torch.ops.prims.inductor_random.default(shape1, seed1, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random0, random1


def _allocate_outputs(device):
    clone = torch.empty_strided(
        (SEQ, BATCH),
        (BATCH, 1),
        device=device,
        dtype=torch.int64,
    )
    mask = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.bool,
    )
    scaled = torch.empty_strided(
        (SEQ, BATCH, HIDDEN),
        (BATCH * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    token_bf16 = torch.empty_strided(
        (TOKEN_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    pos_bf16 = torch.empty_strided(
        (POS_ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    return clone, mask, scaled, token_bf16, pos_bf16


# 4ae3e770: XLNetLMHeadModel train embedding + dual dropout positional fanout.
@oracle_impl(hardware="B200", point="4ae3e770", TOKEN_BLOCK=1024, POS_BLOCK=1024)
def oracle_forward(inputs, *, TOKEN_BLOCK: int, POS_BLOCK: int):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, _shape3, _shape4 = inputs
    del _shape0, _shape1, _shape2, _shape3, _shape4
    device = arg1_1.device

    clone, mask, scaled, token_bf16, pos_bf16 = _allocate_outputs(device)

    seeds, random0, random1 = _seeds_and_randoms_for_eager_check(
        (SEQ, BATCH, HIDDEN),
        (POS_SEQ, BATCH, HIDDEN),
        device=device,
    )

    random0_flat = random0.reshape(-1).contiguous()
    random1_flat = random1.reshape(-1).contiguous()

    # Views for kernel
    mask_flat = mask.view(-1)
    scaled_flat = scaled.view(-1)
    token_bf16_flat = token_bf16.view(-1)
    pos_bf16_flat = pos_bf16.view(-1)
    clone_flat = clone.view(-1)

    ids_flat = arg0_1.reshape(-1).contiguous()
    table_flat = arg1_1.reshape(-1).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOKEN_ROWS, 1, 1),
        _token_embed_dropout_kernel,
        (ids_flat, table_flat, random0_flat, clone_flat, mask_flat, scaled_flat, token_bf16_flat,
         TOKEN_NUMEL, SEQ, BATCH, HIDDEN, DROPOUT_P, DROPOUT_SCALE, TOKEN_BLOCK),
    )
    ct.launch(
        stream,
        (POS_ROWS, 1, 1),
        _positional_dropout_kernel,
        (random1_flat, pos_bf16_flat, HIDDEN, HALF_HIDDEN, BATCH, DROPOUT_P, DROPOUT_SCALE, POS_BLOCK),
    )
    return (
        clone,
        seeds,
        mask,
        scaled,
        token_bf16,
        pos_bf16,
        pos_bf16.t(),
        token_bf16.t(),
    )
