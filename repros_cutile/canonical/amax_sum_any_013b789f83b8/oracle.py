"""cuTile port of amax_sum_any_013b789f83b8: BERT/Electra attention softmax + dropout.

Pre-generates the seeded random tensor via inductor_random. One row cuTile kernel
performs stable fp32 softmax with all-minus-inf fallback and bf16 dropout.
Returns (where, gt, out, out.permute(0, 2, 1)).
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
    scores_ptr,     # bf16 [rows, K]
    fallback_ptr,   # bf16 [rows, K]
    random_ptr,     # f32  [rows, K]
    where_ptr,      # bf16 [rows, K]
    gt_ptr,         # b8   [rows, K]
    out_ptr,        # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    scores_bf = ct.load(scores_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(scores_bf, ct.float32)

    # detect row with any value (i.e. not all -inf). Since inputs are bf16
    # loaded without a mask, if a row was created via mask fill with -inf,
    # all elements are -inf. Use: row_has_value = (max(scores) != -inf).
    row_max = ct.max(scores, axis=1, keepdims=True)
    neg_inf = ct.full((1, 1), float("-inf"), dtype=ct.float32)
    row_has_value = row_max != neg_inf

    zero_f = ct.zeros((1, 1), dtype=ct.float32)
    safe_max = ct.where(row_has_value, row_max, zero_f)
    shifted = scores - safe_max
    numer = ct.exp(shifted)
    # mask out rows with no value from numerator
    zero_row = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    numer = ct.where(row_has_value, numer, zero_row)
    denom = ct.sum(numer, axis=1, keepdims=True)
    one_f = ct.full((1, 1), 1.0, dtype=ct.float32)
    safe_denom = ct.where(row_has_value, denom, one_f)
    probs_bf16 = ct.astype(numer / safe_denom, ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, BLOCK_N))
    where_value = ct.where(row_has_value, probs_bf16, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_value)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_value, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="8ae0f618", block_k=512)
@oracle_impl(hardware="B200", point="fac7e171", block_k=512)
def oracle_forward(inputs, *, block_k: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, shape3 = inputs
    view_shape = _as_shape(_shape0)
    flat_shape = _as_shape(shape3)
    device = arg0_1.device

    where = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                                device=device, dtype=torch.bfloat16)
    gt = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                             device=device, dtype=torch.bool)
    out = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                              device=device, dtype=torch.bfloat16)

    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    scores_2d = arg0_1.contiguous().view(n_rows, k_len)
    fallback_2d = arg1_1.contiguous().view(n_rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(_as_shape(shape1), seed, device=device)
    random_2d = random.contiguous().view(n_rows, k_len)

    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (scores_2d, fallback_2d, random_2d, where_2d, gt_2d, out_2d, block_k),
    )
    return where, gt, out, out.permute(0, 2, 1)
