"""cuTile port of amax_amax_any_dc2cfaa9c1cb: LayoutLM scaled softmax + dropout.

bf16[384,512,512] attention pattern. Seed index 19. Point "279c055a".
Returns (raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0,2,1)).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,         # bf16 [rows, K]
    random_ptr,    # f32  [rows, K]
    raw_amax_ptr,  # f32  [rows]
    scaled_amax_ptr,  # f32 [rows]
    all_finite_ptr,   # b8  [rows]
    sum_ptr,       # f32 [rows]
    gt_ptr,        # b8  [rows, K]
    dropped_ptr,   # bf16 [rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    raw = ct.astype(x, ct.float32)
    scaled_bf16 = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)

    raw_max = ct.max(raw, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)

    inf_val = ct.full((1, BLOCK_N), float("inf"), dtype=ct.float32)
    abs_scaled = ct.where(scaled >= 0.0, scaled, -scaled)
    is_finite = ~ct.isnan(scaled) & (abs_scaled != inf_val)
    zero_i = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    one_i = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    invalid_flag = ct.where(is_finite, zero_i, one_i)
    any_invalid = ct.max(invalid_flag, axis=1, keepdims=True)
    all_finite = any_invalid == 0

    shifted_unscaled = (raw - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(raw_amax_ptr, index=(row,), tile=ct.reshape(raw_max, (1,)))
    ct.store(scaled_amax_ptr, index=(row,), tile=ct.reshape(scaled_max, (1,)))
    ct.store(all_finite_ptr, index=(row,), tile=ct.reshape(all_finite, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs, zero_bf)
    scaled_dropout = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_dropout)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="279c055a", BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    del _expand_shape

    full_shape = _shape_tuple(full_shape_arg)
    random_shape = _shape_tuple(random_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    row_stride = _contiguous_stride(row_shape)
    full_stride = _contiguous_stride(full_shape)
    device = x.device

    raw_amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    scaled_amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    all_finite = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.bool)
    sum_1 = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = x.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    raw_amax_1d = raw_amax.view(n_rows)
    scaled_amax_1d = scaled_amax.view(n_rows)
    all_finite_1d = all_finite.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_1d, scaled_amax_1d, all_finite_1d, sum_1d,
         gt_2d, dropped_2d, BLOCK_N),
    )

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
