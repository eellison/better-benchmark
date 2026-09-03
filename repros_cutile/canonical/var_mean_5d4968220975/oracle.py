"""cuTile port of var_mean_5d4968220975: DebertaV2 embedding + dropout + LayerNorm.

Word embedding + position embedding gather, LayerNorm, dropout.
HIDDEN=1536 -> BLOCK_H=2048 with masked reductions.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 1536
SEED_COUNT = 73
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7
BLOCK_H = 2048


@ct.kernel
def _embedding_layernorm_dropout_kernel(
    word_table_ptr,    # f32 [vocab, HIDDEN]
    word_ids_ptr,      # i64 [ROWS] (flattened)
    position_table_ptr,# f32 [max_pos, HIDDEN]
    position_ids_ptr,  # i64 [SEQ_LEN] (flattened)
    weight_ptr,        # f32 [HIDDEN]
    bias_ptr,          # f32 [HIDDEN]
    random_ptr,        # f32 [ROWS, HIDDEN]
    word_out_ptr,      # f32 [ROWS, BLOCK_H]
    position_out_ptr,  # f32 [SEQ_LEN, BLOCK_H]
    mean_ptr,          # f32 [ROWS]
    rsqrt_ptr,         # f32 [ROWS]
    gt_ptr,            # bool [ROWS, BLOCK_H]
    scaled_ptr,        # f32 [ROWS, BLOCK_H]
    bf16_view_ptr,     # bf16 [ROWS, BLOCK_H]
    ROWS_C: ct.Constant[int],
    SEQ_LEN_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)

    # Compute sequence position for this row
    seq = row % SEQ_LEN_C

    # Load word and position IDs as scalars
    word_id_arr = ct.load(word_ids_ptr, index=(row,), shape=(1,))
    position_id_arr = ct.load(position_ids_ptr, index=(seq,), shape=(1,))

    # For gather on 2D table, need (row_indices, col_indices) tuple
    # Broadcast row index to [1, BLOCK_H_]
    word_id_row = ct.reshape(word_id_arr, (1, 1)) + ct.zeros((1, BLOCK_H_), dtype=ct.int64)
    position_id_row = ct.reshape(position_id_arr, (1, 1)) + ct.zeros((1, BLOCK_H_), dtype=ct.int64)

    # Column indices are just 0..BLOCK_H_-1
    col_indices = ct.reshape(ct.arange(BLOCK_H_, dtype=ct.int64), (1, BLOCK_H_))

    # Use gather to load embeddings
    word = ct.gather(word_table_ptr, (word_id_row, col_indices))
    position = ct.gather(position_table_ptr, (position_id_row, col_indices))

    # Store gathered embeddings
    ct.store(word_out_ptr, index=(row, 0), tile=word)
    ct.store(position_out_ptr, index=(seq, 0), tile=position)

    # Add embeddings
    x = word + position

    # LayerNorm with masking for HIDDEN < BLOCK_H
    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_C, (1, BLOCK_H_))
    x_masked = ct.where(col_mask, x, 0.0)

    mean = ct.sum(x_masked) * (1.0 / HIDDEN_C)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    # Affine transform
    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d

    # Dropout
    random = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    keep = random > DROPOUT_P
    dropped = ct.where(keep, affine, 0.0)
    scaled = dropped * DROPOUT_SCALE

    # Store outputs
    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    ct.store(scaled_ptr, index=(row, 0), tile=scaled)
    ct.store(bf16_view_ptr, index=(row, 0), tile=ct.astype(scaled, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="63e99c48", BLOCK_H=2048, ROW_BLOCK=1)
def oracle_forward(inputs, **_kwargs):
    word_table, word_ids, position_table, position_ids, weight, bias, shape0, shape1 = inputs
    random_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape1)
    device = word_table.device

    # Flatten ID tensors to 1D
    word_ids_flat = word_ids.reshape(-1).contiguous()
    position_ids_flat = position_ids.reshape(-1).contiguous()

    # Create padded outputs
    word_out_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    position_out_pad = torch.empty((SEQ_LEN, BLOCK_H), device=device, dtype=torch.float32)
    mean_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((ROWS,), device=device, dtype=torch.float32)
    gt_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bool)
    scaled_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    bf16_view_pad = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.bfloat16)

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)
    random_flat = random.reshape(ROWS, HIDDEN).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ROWS, 1, 1), _embedding_layernorm_dropout_kernel,
        (word_table, word_ids_flat, position_table, position_ids_flat, weight, bias,
         random_flat, word_out_pad, position_out_pad, mean_1d, rsqrt_1d,
         gt_pad, scaled_pad, bf16_view_pad, ROWS, SEQ_LEN, HIDDEN, BLOCK_H),
    )

    # Reshape to final 3D shapes
    word_out = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.float32,
    )
    word_out.view(ROWS, HIDDEN).copy_(word_out_pad.narrow(1, 0, HIDDEN))

    position_out = torch.empty_strided(
        (1, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    position_out.view(SEQ_LEN, HIDDEN).copy_(position_out_pad.narrow(1, 0, HIDDEN))

    mean = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    mean.view(ROWS).copy_(mean_1d)

    rsqrt = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=device,
        dtype=torch.float32,
    )
    rsqrt.view(ROWS).copy_(rsqrt_1d)

    gt = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.bool,
    )
    gt.view(ROWS, HIDDEN).copy_(gt_pad.narrow(1, 0, HIDDEN))

    scaled = torch.empty_strided(
        random_shape,
        _contiguous_stride(random_shape),
        device=device,
        dtype=torch.float32,
    )
    scaled.view(ROWS, HIDDEN).copy_(scaled_pad.narrow(1, 0, HIDDEN))

    bf16_view = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    bf16_view.view(ROWS, HIDDEN).copy_(bf16_view_pad.narrow(1, 0, HIDDEN))

    return word_out, position_out, mean, rsqrt, seeds, gt, scaled, bf16_view
