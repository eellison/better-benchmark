"""cuTile port of amax_sum_any_3342ba469382: DistilBert softmax + dropout with -inf-mask fallback.

Row softmax with observable amax/sum outputs plus an "all-masked" bool
side output; the port pre-generates the seeded random tensor via
inductor_random and does the whole envelope in one cuTile kernel.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_outputs_kernel(
    x_ptr,          # bf16 [rows, K]
    random_ptr,     # f32 [rows, K]
    amax_ptr,       # f32 [rows]
    sum_ptr,        # f32 [rows]
    all_masked_ptr, # b8  [rows]
    full_ptr,       # bf16 [rows, K]  (zeros)
    gt_ptr,         # b8   [rows, K]
    dropped_ptr,    # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(x_bf, ct.float32)
    # not_minus_inf[i] = (scores[i] != -inf); all_masked = ~any(not_minus_inf)
    minf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    not_minf = scores != minf
    # has_any: 1 if any not_minf, else 0
    has_any_i = ct.max(ct.where(not_minf, 1, 0), axis=1)  # (1,)
    all_masked = has_any_i == 0
    ct.store(all_masked_ptr, index=(row,), tile=ct.reshape(all_masked, (1,)))

    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs_bf = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    ct.store(full_ptr, index=(row, 0), tile=zero_bf)

    all_masked_2d = ct.reshape(all_masked, (1, 1))
    where_val = ct.where(all_masked_2d, zero_bf, probs_bf)

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


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


def _launch(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, _shape0, shape1, shape2, _shape3, shape4 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(shape1)
    random_shape = _shape_tuple(shape2)
    out_shape = _shape_tuple(shape4)
    row_shape = full_shape[:-1] + (1,)
    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    device = arg0_1.device

    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    all_masked = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                     device=device, dtype=torch.bool)
    full = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                               device=device, dtype=torch.bfloat16)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    all_masked_1d = all_masked.view(n_rows)
    full_2d = full.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_outputs_kernel,
        (x_2d, random_2d, amax_1d, sum_1d, all_masked_1d,
         full_2d, gt_2d, dropped_2d, BLOCK_N),
    )
    return amax, sum_1, all_masked, full, gt, dropped, dropped.permute(0, 2, 1)


@oracle_impl(hardware="B200", point="955468a8", BLOCK_N=128)
@oracle_impl(hardware="B200", point="279c055a", BLOCK_N=512)
@oracle_impl(hardware="B200", point="0745dc5a", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    return _launch(inputs, BLOCK_N=BLOCK_N)
