"""cuTile port of amax_sum_192a3a5bd7ff: BERT masked-fill softmax + dropout.

Pre-generates seeded random via inductor_random. One cuTile row kernel does:
scale-by-8 bf16 rounding, boolean mask scalar-fill, fp32 softmax
(amax/exp/sum), seeded dropout with scale 1/(1-0.1).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 36
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    scores_ptr,     # bf16 (n_rows, k_len)
    mask_ptr,       # b8 (batch, 1, q_len, k_len), viewed as (batch, q_len, k_len)
    fill_ptr,       # bf16 ()
    random_ptr,     # f32 (n_rows, k_len)
    where_ptr,      # bf16 (n_rows, k_len)
    amax_ptr,       # f32 (n_rows,)
    sum_ptr,        # f32 (n_rows,)
    gt_ptr,         # b8 (n_rows, k_len)
    dropped_ptr,    # bf16 (n_rows, k_len)
    HEADS: ct.Constant[int],
    Q_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    # Decode: b = row // (heads*q_len); q = row % q_len
    bh = row // Q_LEN
    batch = bh // HEADS
    q = row - bh * Q_LEN

    score = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scaled_bf = ct.astype(ct.astype(score, ct.float32) * 0.125, ct.bfloat16)

    # mask_ptr as (batch, q_len, k_len) since dim 1 is size 1 — read as (b, q, 0..k_len).
    mask_2d = ct.load(mask_ptr, index=(batch, q, 0), shape=(1, 1, BLOCK_N))
    mask_flat = ct.reshape(mask_2d, (1, BLOCK_N))
    # fill_ptr is a 0-D tensor; passed as a 1-elem array of shape (1,).
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.astype(ct.reshape(fill_tile, (1, 1)), ct.bfloat16)
    fill_broadcast = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16) + fill_bf
    rounded = ct.where(mask_flat, fill_broadcast, scaled_bf)
    ct.store(where_ptr, index=(row, 0), tile=rounded)

    x = ct.astype(rounded, ct.float32)
    row_max = ct.max(x)
    numer = ct.exp(x - row_max)
    denom = ct.sum(numer)
    probs = numer * (1.0 / denom)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = rand_f > 0.1
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_f = ct.astype(keep, ct.float32) * probs
    scaled_dropout = dropped_f * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row, 0), tile=ct.astype(scaled_dropout, ct.bfloat16))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
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


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    scores, mask, fill, seeds, full_shape, random_shape, _expand_shape, out_shape = inputs
    full_shape = tuple(int(dim) for dim in full_shape)
    random_shape = tuple(int(dim) for dim in random_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(scores.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    device = scores.device

    where = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16)
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32)
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16)

    # scores (192, 128, 128) view -> (16, 12, 128, 128) -> flatten first three
    scores_2d = scores.view(full_shape).reshape(n_rows, k_len)
    # mask has shape (16, 1, 128, 128). squeeze dim 1 to (16, 128, 128).
    mask_3d = mask.view(full_shape[0], q_len, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len)

    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)

    # fill is a 0-D tensor; view as (1,) so cuTile can load index=(0,) shape=(1,).
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _masked_softmax_dropout_kernel,
        (scores_2d, mask_3d, fill_1d, random_2d, where_2d, amax_1d, sum_1d, gt_2d, dropped_2d,
         n_heads, q_len, BLOCK_N),
    )
    return where, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
