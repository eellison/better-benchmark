"""cuTile port of amax_sum_92926420c266: MT5 additive-bias softmax + dropout.

Pre-materializes the bf16(score + strided-bias) tensor in torch (matches the
Repro's bf16 rounding after the fp32 add), then runs one cuTile row kernel
that does stable softmax with fp32 side outputs, seeded dropout, and the
scaled bf16 output plus its permuted alias.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 25
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    scores_ptr,     # bf16 [n_rows, K]  (already includes bias)
    random_ptr,     # f32  [n_rows, K]
    amax_ptr,       # f32  [n_rows]
    sum_ptr,        # f32  [n_rows]
    gt_ptr,         # b8   [n_rows, K]
    dropped_ptr,    # bf16 [n_rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(scores_bf, ct.float32)

    row_max = ct.max(scores, axis=1, keepdims=True)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
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


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=4, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    del BLOCK_M  # one row per program
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, shape3 = inputs
    del _shape0, _shape2

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    full_shape = _shape_tuple(shape1)
    out_shape = _shape_tuple(shape3)
    n_heads = int(full_shape[1])
    q_len = int(full_shape[2])
    k_len = int(full_shape[3])
    n_rows = int(arg0_1.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    batch = int(full_shape[0])
    device = arg0_1.device

    # --- Pre-compute the fp32-add + bf16-round in torch (bias is strided) ---
    view_bf = arg0_1.view(batch, n_heads, q_len, k_len)
    add_f = view_bf.to(torch.float32) + arg1_1  # arg1_1 is f32, strided
    rounded = add_f.to(torch.bfloat16)  # returned as convert_element_type
    scores_2d = rounded.contiguous().view(n_rows, k_len)

    amax = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                               device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, _contiguous_stride(row_shape),
                                device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, _contiguous_stride(full_shape),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.contiguous().view(n_rows, k_len)

    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_dropout_kernel,
        (scores_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, BLOCK_N),
    )

    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
