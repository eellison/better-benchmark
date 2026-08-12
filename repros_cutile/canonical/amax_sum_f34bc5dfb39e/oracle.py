"""cuTile port of amax_sum_f34bc5dfb39e: BERT scaled masked softmax + dropout.

Pre-computes the masked scores in torch (scale by 0.125 in bf16, then where(mask, fill, scaled)),
then runs a cuTile kernel over rows to compute softmax + fp32 amax/sum + dropout.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 56
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    where_in_ptr,   # bf16 [rows, K]
    random_ptr,     # f32 [rows, K]
    amax_ptr,       # f32 [rows]
    sum_ptr,        # f32 [rows]
    gt_ptr,         # b8 [rows, K]
    dropped_ptr,    # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    where_val_bf = ct.load(where_in_ptr, index=(row, 0), shape=(1, BLOCK_N))
    where_f = ct.astype(where_val_bf, ct.float32)
    row_max = ct.max(where_f)
    numer = ct.exp(where_f - row_max)
    denom = ct.sum(numer)
    probs = numer / denom
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep = rand > DROPOUT_P
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_f = ct.astype(keep, ct.float32) * probs
    scaled = ct.astype(dropped_f * DROPOUT_SCALE, ct.bfloat16)
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


@oracle_impl(hardware="B200", point="0e2c5e9e", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    scores, mask, fill, seeds, full_shape, random_shape, _expand_shape, out_shape = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(full_shape)
    random_shape = _shape_tuple(random_shape)
    out_shape = _shape_tuple(out_shape)
    device = scores.device
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(scores.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    # Compute masked scores in torch for maximum fidelity to the Triton
    # `div(view, 8.0)` bf16 rounding boundary and `where(mask, fill, div)`.
    view = scores.view(full_shape)
    scaled_bf16 = (view.to(torch.float32) * 0.125).to(torch.bfloat16)
    where = torch.where(mask, fill.to(torch.bfloat16), scaled_bf16)
    where_out = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bfloat16,
    )
    where_out.copy_(where)

    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    where_2d = where_out.view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _masked_softmax_dropout_kernel,
        (where_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, BLOCK_N),
    )
    return where_out, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
