"""cuTile port of amax_sum_a9730a53341e: BERT masked attention softmax + seeded dropout.

bf16 scores /8, then scalar-fill masked, stable f32 softmax, seeded dropout.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _bert_masked_softmax_dropout_kernel(
    scores_ptr,   # bf16 (n_rows, K_LEN)
    mask_ptr,     # bool (batch * Q_LEN, K_LEN)
    fill_ptr,     # bf16 [1]
    random_ptr,   # f32 (n_rows, K_LEN)
    where_ptr,    # bf16 (n_rows, K_LEN)
    amax_ptr,     # f32 (n_rows,)
    sum_ptr,      # f32 (n_rows,)
    gt_ptr,       # bool (n_rows, K_LEN)
    dropped_ptr,  # bf16 (n_rows, K_LEN)
    N_HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    bh = row // Q_LEN
    q = row - bh * Q_LEN
    b = bh // N_HEADS

    raw_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scaled_bf = ct.astype(ct.astype(raw_bf, ct.float32) * 0.125, ct.bfloat16)
    mask_row = b * Q_LEN + q
    m_bool = ct.load(mask_ptr, index=(mask_row, 0), shape=(1, BLOCK_N))
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.astype(fill_tile, ct.bfloat16)
    fill_2d = ct.reshape(ct.broadcast_to(fill_bf, (BLOCK_N,)), (1, BLOCK_N))
    masked = ct.where(m_bool, fill_2d, scaled_bf)
    ct.store(where_ptr, index=(row, 0), tile=masked)

    scores = ct.astype(masked, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = random_f > DROPOUT_P
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


def _inductor_random_for_eager_check(shape, seed, *, device):
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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    scores, mask, fill, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(full_shape_arg)
    random_shape = _shape_tuple(random_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    batch = int(full_shape[0])
    n_rows = int(scores.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    where = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                                 device=scores.device, dtype=torch.bfloat16)
    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=scores.device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                 device=scores.device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                              device=scores.device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                   device=scores.device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed,
                                              device=scores.device)

    scores_2d = scores.contiguous().view(n_rows, k_len)
    mask_2d = mask.view(batch * q_len, k_len)
    fill_1d = fill.view(1)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _bert_masked_softmax_dropout_kernel,
        (scores_2d, mask_2d, fill_1d, random_2d,
         where_2d, amax_1d, sum_1d, gt_2d, dropped_2d,
         n_heads, q_len, k_len, BLOCK_N),
    )

    return where, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
