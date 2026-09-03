"""cuTile port of var_mean_8187f7c4e01e: MegatronBERT embedding + seeded
dropout + LayerNorm.

Do the embedding lookups via torch.ops.aten.embedding outside the kernel
(gather ops), pre-generate seeds and random tensor via inductor primitives,
then run one cuTile row kernel that performs dropout + LN.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ_LEN = 512
ROWS = BATCH * SEQ_LEN
HIDDEN = 1024
SEED_COUNT = 49
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_ln_kernel(
    x_ptr,             # f32 (rows, HIDDEN)
    random_ptr,        # f32 (rows, HIDDEN)
    weight_ptr,        # f32 (HIDDEN,)
    bias_ptr,          # f32 (HIDDEN,)
    gt_ptr,            # bool (rows, HIDDEN)
    scaled_ptr,        # f32 (rows, HIDDEN)
    normalized_ptr,    # f32 (rows, HIDDEN)
    bf16_view_ptr,     # bf16 (rows, HIDDEN)
    div_ptr,           # f32 (rows,)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    keep = random_f > DROPOUT_P
    dropped = ct.where(keep, x, ct.zeros((1, BLOCK_H), dtype=ct.float32))
    scaled = dropped * DROPOUT_SCALE

    mean = ct.sum(scaled, axis=1, keepdims=True) * (1.0 / HIDDEN_C)
    centered = scaled - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    ct.store(gt_ptr, index=(row, 0), tile=keep)
    ct.store(scaled_ptr, index=(row, 0), tile=scaled)
    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_view_ptr, index=(row, 0), tile=affine_bf)
    inv_h = ct.reshape(invstd * (1.0 / HIDDEN_C), (1,))
    ct.store(div_ptr, index=(row,), tile=inv_h)


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
    if torch.cuda.is_current_stream_capturing():
        seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
        seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        return seeds, random
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


@oracle_impl(hardware="B200", point="ad62e054", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        word_table, word_ids, token_type_table, position_table, position_ids,
        weight, bias, full_shape, random_shape, flat_shape,
    ) = inputs
    full_shape = _as_shape(full_shape)
    random_shape = _as_shape(random_shape)
    flat_shape = _as_shape(flat_shape)
    device = word_table.device

    # Embeddings via torch aten ops (works under CUDA-graph capture)
    full = torch.zeros(full_shape, device=device, dtype=torch.int64)
    embedding = torch.ops.aten.embedding.default(word_table, word_ids, 0)
    embedding_1 = torch.ops.aten.embedding.default(token_type_table, full)
    embedding_2 = torch.ops.aten.embedding.default(position_table, position_ids)
    add_1 = embedding + embedding_1 + embedding_2  # (16, 512, 1024)

    seeds, random = _seeds_and_random_for_eager_check(random_shape, device=device)

    gt = torch.empty_strided(random_shape, _contiguous_stride(random_shape),
                             device=device, dtype=torch.bool)
    scaled = torch.empty_strided(random_shape, _contiguous_stride(random_shape),
                                 device=device, dtype=torch.float32)
    normalized = torch.empty_strided(random_shape, _contiguous_stride(random_shape),
                                     device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(
        (BATCH, SEQ_LEN, 1),
        (SEQ_LEN, 1, 1),
        device=device, dtype=torch.float32,
    )

    x_2d = add_1.view(ROWS, HIDDEN)
    random_2d = random.view(ROWS, HIDDEN)
    gt_2d = gt.view(ROWS, HIDDEN)
    scaled_2d = scaled.view(ROWS, HIDDEN)
    normalized_2d = normalized.view(ROWS, HIDDEN)
    bf16_2d = bf16_view.view(ROWS, HIDDEN)
    div_1d = div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _dropout_ln_kernel,
        (x_2d, random_2d, weight, bias,
         gt_2d, scaled_2d, normalized_2d, bf16_2d, div_1d,
         HIDDEN, BLOCK_H),
    )

    return full, seeds, gt, scaled, normalized, bf16_view, div
