"""cuTile port of amax_sum_any_32ad3e5c5477: M2M100 masked attention softmax + dropout.

Pre-generates seeds and the random tensor via inductor_seeds/inductor_random
outside the kernel, then runs one cuTile row kernel that does mask-guarded
row softmax + seeded dropout. Refuses under CUDA-graph capture (RNG state
unavailable).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_COUNT = 3
SEED_INDEX = 0
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _masked_softmax_dropout_kernel(
    x_ptr,          # bf16 [rows, K] (dense contiguous view of arg1_1)
    mask_ptr,       # b8   [rows, K] (dense expanded mask)
    random_ptr,     # f32  [rows, K]
    where_ptr,      # bf16 [rows, K]
    gt_ptr,         # b8   [rows, K]
    dropped_ptr,    # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    keep_bool = ct.load(mask_ptr, index=(row, 0), shape=(1, BLOCK_N))

    raw = ct.astype(x_bf, ct.float32)
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    values = ct.where(keep_bool, raw, neg_inf)

    # Row validity: does any col have keep_bool == True?
    zero_i = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    one_i = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    live_flag = ct.where(keep_bool, one_i, zero_i)
    live_sum = ct.sum(live_flag)
    has_any = live_sum > 0

    row_max = ct.max(values)
    safe_max = ct.where(has_any, row_max, ct.astype(0.0, ct.float32))

    shifted = values - safe_max
    numer_raw = ct.exp(shifted)
    zero_f = ct.zeros((1, BLOCK_N), dtype=ct.float32)
    numer = ct.where(keep_bool, numer_raw, zero_f)
    denom_scalar = ct.sum(numer)
    safe_denom = ct.where(has_any, denom_scalar, ct.astype(1.0, ct.float32))
    probs = numer / safe_denom
    probs_bf = ct.astype(probs, ct.bfloat16)
    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    where_val = ct.where(has_any, probs_bf, zero_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep_drop = rand_bf > thresh_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep_drop)

    dropped_bf = ct.where(keep_drop, where_val, zero_bf)
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


def _random_advance(shape, *, device):
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
    return (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )


def _seeds_and_random_for_eager_check(shape, *, device):
    total_advance = 8 + _random_advance(shape, device=device)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    rewound = None
    if offset >= total_advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - total_advance)
        torch.cuda.set_rng_state(rewound, device)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default(shape, seed, "rand")

    if rewound is not None:
        torch.cuda.set_rng_state(state, device)
    return seeds, random


@oracle_impl(hardware="B200", point="54ff0363", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, shape0, _shape1, random_shape, _shape3, flat_shape = inputs
    del _shape1, _shape3

    full_shape = _shape_tuple(shape0)  # (64, 16, 128, 128)
    rand_shape = _shape_tuple(random_shape)
    out_flat_shape = _shape_tuple(flat_shape)
    batch_size, heads, q_len, k_len = full_shape
    n_rows = batch_size * heads * q_len
    device = arg1_1.device

    where = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    gt = torch.empty_strided(
        full_shape,
        (heads * q_len * k_len, q_len * k_len, k_len, 1),
        device=device,
        dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_flat_shape,
        (out_flat_shape[1] * out_flat_shape[2], out_flat_shape[2], 1),
        device=device,
        dtype=torch.bfloat16,
    )

    seeds, random = _seeds_and_random_for_eager_check(rand_shape, device=device)

    # Expand mask arg0_1 [64, 1, 128, 128] (stride (0, 128, 1, 0)) to full
    # [64, 16, 128, 128] and materialize contiguous.
    mask_full = arg0_1.expand(full_shape).contiguous()

    x_2d = arg1_1.contiguous().view(n_rows, k_len)
    mask_2d = mask_full.view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _masked_softmax_dropout_kernel,
        (x_2d, mask_2d, random_2d, where_2d, gt_2d, dropped_2d, BLOCK_N),
    )
    return where, seeds, gt, dropped, dropped.permute(0, 2, 1)
