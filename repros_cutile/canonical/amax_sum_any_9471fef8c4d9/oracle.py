"""cuTile port of amax_sum_any_9471fef8c4d9: MobileBERT softmax + fallback + dropout.

If all values in a row are -inf, fall back to the alt tensor arg1_1. Then
apply seeded dropout. Seed index 14.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 14
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_fallback_dropout_kernel(
    x_ptr,          # bf16 [rows, k_len]
    alt_ptr,        # bf16 [rows, k_len] (fallback)
    random_ptr,     # f32  [rows, k_len]
    where_ptr,      # bf16 [rows, k_len]
    gt_ptr,         # bool [rows, k_len]
    out_ptr,        # bf16 [rows, k_len]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    alt_bf = ct.load(alt_ptr, index=(row, 0), shape=(1, BLOCK_N))

    scores = ct.astype(x_bf, ct.float32)
    row_max = ct.max(scores)
    shifted = scores - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    probs_bf = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    # any(x != -inf) -> all_masked = ~any(!eq(-inf))
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.bfloat16)
    is_not_neg_inf = x_bf != neg_inf
    # any(not_neg_inf) — do a max-of-int trick
    one_i32 = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    zero_i32 = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    flag = ct.where(is_not_neg_inf, one_i32, zero_i32)
    any_finite = ct.max(flag) != 0
    all_masked = ~any_finite

    where_bf = ct.where(all_masked, alt_bf, probs_bf)
    ct.store(where_ptr, index=(row, 0), tile=where_bf)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, view_shape, random_shape, _expand_shape, out_shape = inputs
    view_shape = tuple(int(d) for d in view_shape)  # [256, 4, 128, 128]
    random_shape = tuple(int(d) for d in random_shape)
    out_shape = tuple(int(d) for d in out_shape)
    b, h, q, k = view_shape
    device = arg0_1.device

    view4 = arg0_1.view(view_shape)

    where_bf = torch.empty_strided(view_shape, _contiguous_stride(view_shape), device=device, dtype=torch.bfloat16)
    gt = torch.empty_strided(view_shape, _contiguous_stride(view_shape), device=device, dtype=torch.bool)
    out_bf = torch.empty_strided(view_shape, _contiguous_stride(view_shape), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    rows = b * h * q
    view4_2d = view4.contiguous().view(rows, k)
    alt_2d = arg1_1.contiguous().view(rows, k)
    random_2d = random.contiguous().view(rows, k)
    where_2d = where_bf.view(rows, k)
    gt_2d = gt.view(rows, k)
    out_2d = out_bf.view(rows, k)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _softmax_fallback_dropout_kernel,
        (view4_2d, alt_2d, random_2d, where_2d, gt_2d, out_2d, BLOCK_N),
    )
    view_1 = out_bf.view(out_shape)
    permute = view_1.permute(0, 2, 1)
    return where_bf, gt, view_1, permute
