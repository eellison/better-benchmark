"""cuTile port of amax_sum_6f3222a009d0: MT5/T5 attention softmax + dropout.

Pre-generates the seeded random tensor via inductor_random outside the kernel,
then runs one cuTile row kernel that does bias-add + row softmax + seeded
dropout. Handles two shape points: dda3d8e0 (MT5: [192,128,128] bf16 + [32,6,128,128] f32)
and aeb1682d (T5: [64,1024,1024] bf16 + [8,8,1024,1024] f32).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 17
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K] (dense contiguous)
    bias_ptr,       # f32 [rows, K] (dense contiguous)
    random_ptr,     # f32 [rows, K]
    rounded_ptr,    # bf16 [rows, K]
    amax_ptr,       # f32 [rows]
    sum_ptr,        # f32 [rows]
    gt_ptr,         # b8  [rows, K]
    dropped_ptr,    # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    bias = ct.load(bias_ptr, index=(row, 0), shape=(1, BLOCK_N))
    added_bf = ct.astype(ct.astype(x_bf, ct.float32) + bias, ct.bfloat16)
    ct.store(rounded_ptr, index=(row, 0), tile=added_bf)

    scores = ct.astype(added_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


def _run(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, shape3 = inputs

    full_shape = _shape_tuple(shape1)  # e.g. (32, 6, 128, 128)
    out_shape = _shape_tuple(shape3)   # e.g. (192, 128, 128)
    k_len = int(full_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    device = arg0_1.device

    def _contiguous_stride(shape):
        stride = []
        running = 1
        for dim in reversed(shape):
            stride.append(running)
            running *= int(dim)
        return tuple(reversed(stride))

    rounded = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)

    # Dense contiguous views. arg1_1 may be strided (head-inner layout),
    # so materialize it as contiguous before viewing to [rows, K].
    x_2d = arg0_1.contiguous().view(n_rows, k_len)
    bias_2d = arg1_1.expand(full_shape).contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)

    rounded_2d = rounded.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_dropout_kernel,
        (x_2d, bias_2d, random_2d, rounded_2d, amax_1d, sum_1d, gt_2d, dropped_2d, BLOCK_N),
    )
    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_N=128)
@oracle_impl(hardware="B200", point="aeb1682d", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    return _run(inputs, BLOCK_N=BLOCK_N)
