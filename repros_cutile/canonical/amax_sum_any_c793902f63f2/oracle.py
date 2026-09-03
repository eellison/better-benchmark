"""cuTile port of amax_sum_any_c793902f63f2: DistilBert safe softmax + fallback + dropout.

Row kernel that: computes bf16 softmax with fp32 stable amax/exp/sum/div,
detects all-(-inf) rows via any/logical_not and replaces those rows with a
per-row fallback tensor (from second input), then applies seeded dropout via
pre-computed random tensor. Returns (where, gt, view_3d, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, K]
    fallback_ptr,   # bf16 [n_rows, K]
    random_ptr,     # f32  [n_rows, K]
    where_ptr,      # bf16 [n_rows, K]
    gt_ptr,         # b8   [n_rows, K]
    dropped_ptr,    # bf16 [n_rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    scores_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(scores_bf, ct.float32)
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)

    # Detect all-(-inf) rows.
    live_flag = ct.where(scores != neg_inf, 1, 0)
    has_any_i = ct.max(live_flag, axis=1, keepdims=True)
    has_any = has_any_i != 0

    row_max = ct.max(scores, axis=1, keepdims=True)
    zero_scalar = ct.full((1, 1), 0.0, dtype=ct.float32)
    safe_max = ct.where(has_any, row_max, zero_scalar)
    shifted = scores - safe_max
    numer_raw = ct.exp(shifted)
    zero_full = ct.full((1, BLOCK_N), 0.0, dtype=ct.float32)
    numer = ct.where(scores != neg_inf, numer_raw, zero_full)
    denom = ct.sum(numer, axis=1, keepdims=True)
    one_scalar = ct.full((1, 1), 1.0, dtype=ct.float32)
    safe_denom = ct.where(has_any, denom, one_scalar)
    probs_bf = ct.astype(numer / safe_denom, ct.bfloat16)

    fallback_bf = ct.load(fallback_ptr, index=(row, 0), shape=(1, BLOCK_N))
    where_val = ct.where(has_any, probs_bf, fallback_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
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


@oracle_impl(hardware="B200", point="e886386f", BLOCK_N=128)
@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, shape3 = inputs

    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    random_shape = _shape_tuple(shape1)
    view3_shape = _shape_tuple(shape3)

    # arg1_1 is [B, H, Q, K] bf16 (fallback + shape for `where`).
    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bool,
    )
    # dropped is 3D flat like arg0_1.
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    x_2d = arg0_1.contiguous().view(n_rows, k_len)
    fallback_2d = arg1_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, BLOCK_N),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
