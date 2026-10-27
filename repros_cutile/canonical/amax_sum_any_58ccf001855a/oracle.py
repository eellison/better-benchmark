"""cuTile port of amax_sum_any_58ccf001855a: safe-softmax dropout.

Pre-generates seeded random with inductor_random. One cuTile row kernel per
row does stable amax/exp/sum/div, `-inf` all-mask fallback to a bf16 tensor,
then dropout with scale 1/(1-0.1).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 4
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,           # bf16 (n_rows, k_len)
    fallback_ptr,    # bf16 (n_rows, k_len)
    random_ptr,      # f32 (n_rows, k_len)
    where_ptr,       # bf16 (n_rows, k_len)
    gt_ptr,          # b8 (n_rows, k_len)
    dropped_ptr,     # bf16 (n_rows, k_len)
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    scores = ct.astype(ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N)), ct.float32)
    row_max = ct.max(scores)
    has_any = row_max != -float("inf")
    safe_max = ct.where(has_any, row_max, 0.0)
    numer = ct.exp(scores - safe_max)
    denom = ct.sum(numer)
    denom = ct.where(has_any, denom, 1.0)
    probs = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, BLOCK_N))
    where_val = ct.where(has_any, probs, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.full((1, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="8ae0f618", BLOCK_N=512)
@oracle_impl(hardware="B200", point="fac7e171", BLOCK_N=512)
@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, _shape3 = inputs
    random_shape = tuple(int(dim) for dim in shape1)
    device = arg0_1.device
    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bool)
    dropped = torch.empty_like(arg0_1)

    x_2d = arg0_1.view(n_rows, k_len)
    fallback_2d = arg1_1.reshape(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, BLOCK_N),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
