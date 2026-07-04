"""cuTile port of amax_sum_any_a8ebede53f88: MobileBERT safe softmax + dropout.

Safe softmax: for rows where all scores are -inf, output the fallback tensor
(arg1_1) instead of softmax(scores). Follows the Triton oracle's random-ptr
branch.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, k_len]
    fallback_ptr,   # bf16 [n_rows, k_len]
    random_ptr,     # f32  [n_rows, k_len]
    where_ptr,      # bf16 [n_rows, k_len]
    gt_ptr,         # b8   [n_rows, k_len]
    dropped_ptr,    # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
):
    row_block = ct.bid(0)

    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    fallback = ct.load(fallback_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    scores = ct.astype(x, ct.float32)
    # Detect all-(-inf) rows: has_any = row has some finite (non-neg-inf) value
    neg_inf_f = ct.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=ct.float32)
    live = scores > neg_inf_f  # b8 tile
    live_int = ct.astype(live, ct.int32)
    live_row_sum = ct.sum(live_int, axis=1, keepdims=True)
    zero_i = ct.full((BLOCK_M, 1), 0, dtype=ct.int32)
    has_any = live_row_sum != zero_i  # b8 [BLOCK_M, 1]

    row_max = ct.max(scores, axis=1, keepdims=True)
    safe_max = ct.where(has_any, row_max, ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32))
    numer = ct.exp(scores - safe_max)
    # For dead lanes (score = -inf), exp(0-0) = 1 but they'll get zeroed via `live` mask
    zero_f = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    numer = ct.where(live, numer, zero_f)
    denom = ct.sum(numer, axis=1, keepdims=True)
    denom_safe = ct.where(has_any, denom, ct.full((BLOCK_M, 1), 1.0, dtype=ct.float32))
    probs_bf = ct.astype(numer / denom_safe, ct.bfloat16)

    where_val = ct.where(has_any, probs_bf, fallback)
    ct.store(where_ptr, index=(row_block, 0), tile=where_val)

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled_f = ct.astype(dropped, ct.float32) * DROPOUT_SCALE_
    ct.store(dropped_ptr, index=(row_block, 0), tile=ct.astype(scaled_f, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="d59f4ab1", block_m=8, block_n=128)
def oracle_forward(inputs, *, block_m: int, block_n: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    random_shape = _as_shape(shape1)
    K = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // K)
    device = arg0_1.device

    # arg0_1 is bf16[1024,128,128], view arg1_1 shape [256,4,128,128] is where scores live
    # arg1_1 [256,4,128,128] is the fallback bf16 tensor
    # In Repro: view = arg0_1.view([256,4,128,128]); then softmax on view, then where(any_row_has_finite, softmax, arg1_1)

    x_view = arg0_1.view(arg1_1.shape).contiguous()  # bf16[256,4,128,128]

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=device,
        dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = x_view.view(n_rows, K)
    fallback_2d = arg1_1.contiguous().view(n_rows, K)
    random_2d = random.contiguous().view(n_rows, K)
    where_2d = where.view(n_rows, K)
    gt_2d = gt.view(n_rows, K)
    dropped_2d = dropped.view(n_rows, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, block_m), 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d,
         K, block_m, block_n, DROPOUT_SCALE),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
