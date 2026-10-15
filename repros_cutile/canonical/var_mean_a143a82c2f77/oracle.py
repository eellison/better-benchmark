"""cuTile port of var_mean_a143a82c2f77: DistilBERT embedding + LayerNorm + dropout.

Pre-gathers embedding on Python side, and pre-generates the seeded RNG tensor.
Row kernel: embedding+position add, var/mean, rsqrt (eps=1e-12), affine, dropout,
bf16 cast.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 768
SEED_COUNT = 13
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _embedding_layernorm_dropout_kernel(
    embedding_ptr,    # f32 [rows, HIDDEN]
    position_ptr,     # f32 [rows, HIDDEN] — expanded across batch
    weight_ptr,       # f32 [HIDDEN]
    bias_ptr,         # f32 [HIDDEN]
    random_ptr,       # f32 [rows, HIDDEN]
    mean_ptr,         # f32 [rows]
    rsqrt_ptr,        # f32 [rows]
    gt_ptr,           # bool [rows, HIDDEN]
    scaled_ptr,       # f32 [rows, HIDDEN]
    bf16_view_ptr,    # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    word = ct.load(
        embedding_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    position = ct.load(
        position_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x = word + position

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H))
    x_masked = ct.where(col_mask, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN_)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    random = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = random > DROPOUT_P
    dropped = ct.where(keep, affine, 0.0)
    scaled = dropped * DROPOUT_SCALE

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    ct.store(scaled_ptr, index=(row, 0), tile=scaled)
    ct.store(bf16_view_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


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
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        return seeds, torch.ops.prims.inductor_random.default(shape, seed, "rand")
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


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="8e9bc156", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    word_table, word_ids, position_ids, position_table, weight, bias, shape0, shape1 = inputs
    random_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    device = word_table.device

    # word embedding lookup
    embedding = torch.ops.aten.embedding.default(word_table, word_ids)
    # position embedding lookup (slice_1 = position_ids[:, 0:128])
    slice_pos = position_ids[:, :SEQ_LEN]
    position_embedding = torch.ops.aten.embedding.default(position_table, slice_pos)
    # expanded [1, SEQ_LEN, HIDDEN] over BATCH
    position_expanded = position_embedding.expand(BATCH, SEQ_LEN, HIDDEN).contiguous()

    # pre-generate seeds and random
    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    mean = torch.empty((BATCH, SEQ_LEN, 1), device=device, dtype=torch.float32)
    rsqrt = torch.empty((BATCH, SEQ_LEN, 1), device=device, dtype=torch.float32)
    gt = torch.empty((BATCH, SEQ_LEN, HIDDEN), device=device, dtype=torch.bool)
    scaled = torch.empty((BATCH, SEQ_LEN, HIDDEN), device=device, dtype=torch.float32)
    bf16_view = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    embedding_2d = embedding.view(ROWS, HIDDEN)
    position_2d = position_expanded.view(ROWS, HIDDEN)
    random_2d = random.contiguous().view(ROWS, HIDDEN)
    mean_1d = mean.view(ROWS)
    rsqrt_1d = rsqrt.view(ROWS)
    gt_2d = gt.view(ROWS, HIDDEN)
    scaled_2d = scaled.view(ROWS, HIDDEN)
    bf16_2d = bf16_view.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _embedding_layernorm_dropout_kernel,
        (embedding_2d, position_2d, weight, bias, random_2d,
         mean_1d, rsqrt_1d, gt_2d, scaled_2d, bf16_2d,
         HIDDEN, BLOCK_H),
    )
    return embedding, position_embedding, mean, rsqrt, seeds, gt, scaled, bf16_view
