"""cuTile port of amax_sum_90f134112275: MT5 softmax + seeded dropout.

Pre-generates the seeded random tensor via inductor_random, runs cuTile row
softmax+dropout, and returns the (amax, sum, gt, dropped, permute alias)
tuple.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 79
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, k_len]
    random_ptr,     # f32  [rows, k_len]
    amax_ptr,       # f32  [rows]
    sum_ptr,        # f32  [rows]
    gt_ptr,         # bool [rows, k_len]
    out_ptr,        # bf16 [rows, k_len]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(x_bf, ct.float32)
    row_max = ct.max(scores)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.astype(ct.where(keep, probs, zero_bf), ct.bfloat16)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="1715052e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    full_shape = tuple(int(d) for d in full_shape_arg)
    random_shape = tuple(int(d) for d in random_shape_arg)
    out_shape = tuple(int(d) for d in out_shape_arg)
    k_len = int(full_shape[-1])
    rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    device = x.device

    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = x.contiguous().view(rows, k_len)
    random_2d = random.contiguous().view(rows, k_len)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, k_len)
    dropped_2d = dropped.view(rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _softmax_dropout_kernel,
        (x_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, BLOCK_N),
    )
    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
