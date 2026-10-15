"""cuTile port of amax_sum_e06a68534c7d: T5/MT5 additive-bias softmax + dropout.

Ports the Triton `_softmax_dropout_random_kernel`. Materializes the strided
bias into a contiguous tensor outside the kernel, then runs one cuTile row
kernel that computes:
  * rounded = bf16(score + bias)
  * fp32 stable softmax on `rounded`
  * dropout via random > 0.1 in bf16 space
  * bf16 dropout scaling by 1.1111...

Returns: (rounded, amax, sum_1, keep, dropped_view, permute_alias).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 9
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_random_kernel(
    score_ptr,      # bf16 [N_ROWS, KLEN]
    bias_ptr,       # f32  [N_ROWS, KLEN]
    random_ptr,     # f32  [N_ROWS, KLEN]
    rounded_ptr,    # bf16 [N_ROWS, KLEN]
    amax_ptr,       # f32 [N_ROWS]
    sum_ptr,        # f32 [N_ROWS]
    keep_ptr,       # bool [N_ROWS, KLEN]
    dropped_ptr,    # bf16 [N_ROWS, KLEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    score_bf = ct.load(score_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    bias_f = ct.load(bias_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    score_f = ct.astype(score_bf, ct.float32)
    added = score_f + bias_f
    rounded_bf = ct.astype(added, ct.bfloat16)
    ct.store(rounded_ptr, index=(row_block, 0), tile=rounded_bf)

    x = ct.astype(rounded_bf, ct.float32)
    row_max = ct.max(x, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    numer = ct.exp(x - row_max_2d)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs_bf = ct.astype(numer / denom_2d, ct.bfloat16)

    ct.store(amax_ptr, index=(row_block,), tile=row_max)
    ct.store(sum_ptr, index=(row_block,), tile=denom)

    rand_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_bf)


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


def _contiguous_4d_stride(shape):
    return (shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1)


def _contiguous_3d_stride(shape):
    return (shape[1] * shape[2], shape[2], 1)


def _reduction_stride(shape):
    return (shape[1] * shape[2], shape[2], 1, 1)


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=8, BLOCK_N=128)
@oracle_impl(hardware="B200", point="aeb1682d", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    del _shape2

    score_shape = tuple(int(dim) for dim in shape0)
    random_shape = tuple(int(dim) for dim in shape1)
    view_shape = tuple(int(dim) for dim in shape3)
    reduction_shape = (score_shape[0], score_shape[1], score_shape[2], 1)
    device = arg0_1.device

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rounded = torch.empty_strided(
        score_shape, _contiguous_4d_stride(score_shape),
        device=device, dtype=torch.bfloat16,
    )
    amax = torch.empty_strided(
        reduction_shape, _reduction_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    denom = torch.empty_strided(
        reduction_shape, _reduction_stride(reduction_shape),
        device=device, dtype=torch.float32,
    )
    keep = torch.empty_strided(
        score_shape, _contiguous_4d_stride(score_shape),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        view_shape, _contiguous_3d_stride(view_shape),
        device=device, dtype=torch.bfloat16,
    )

    k_len = int(score_shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    # arg0_1 is bf16 [flat, q, k] with flat = batch*heads. Materialize the score
    # in the 4D layout to match bias iteration order, then flatten.
    score_2d = arg0_1.view(score_shape).contiguous().view(n_rows, k_len)
    # arg1_1 is f32 with non-contiguous strides — contiguous() to normalize.
    bias_2d = arg1_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    rounded_2d = rounded.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _softmax_dropout_random_kernel,
        (score_2d, bias_2d, random_2d, rounded_2d, amax_1d, sum_1d,
         keep_2d, dropped_2d, BLOCK_M, BLOCK_N),
    )
    return rounded, amax, denom, keep, dropped, dropped.permute(0, 2, 1)
