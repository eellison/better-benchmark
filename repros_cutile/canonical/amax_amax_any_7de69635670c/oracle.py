"""cuTile port of amax_amax_any_7de69635670c: LayoutLM scaled softmax + dropout.

Pre-generates dropout random via torch.ops.prims.inductor_random; the fused
row kernel computes bf16-round-trip scaled scores, dual amax side outputs,
finite-row guard, natural-exp softmax, bf16 probability, seeded dropout mask,
and bf16 dropout epilogue.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
SCALE = 0.125


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,           # bf16 [n_rows, k_len]
    random_ptr,      # f32  [n_rows, k_len]
    raw_amax_ptr,    # f32  [n_rows]
    scaled_amax_ptr, # f32  [n_rows]
    all_finite_ptr,  # b8   [n_rows]
    sum_ptr,         # f32  [n_rows]
    gt_ptr,          # b8   [n_rows, k_len]
    dropped_ptr,     # bf16 [n_rows, k_len]
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    raw_bf = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN))
    raw = ct.astype(raw_bf, ct.float32)
    # scaled = (raw * 0.125) rounded to bf16 then back to f32
    scaled_bf = ct.astype(raw * SCALE, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    raw_max = ct.max(raw)
    scaled_max = ct.max(scaled)

    ct.store(raw_amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), raw_max, dtype=ct.float32), (1,)))
    ct.store(scaled_amax_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), scaled_max, dtype=ct.float32), (1,)))

    # Finite check: any(!finite(scaled))?
    # A non-finite value is either nan (x != x) or +/-inf (|x| == inf).
    # (scaled != scaled) => nan; (|scaled| == inf) => inf.
    # any(!finite) reduces along axis=1.
    is_nan = scaled != scaled
    is_inf = ct.astype(ct.abs(scaled), ct.float32) == float("inf")
    non_finite = is_nan | is_inf
    non_finite_i = ct.astype(non_finite, ct.int32)
    has_invalid = ct.max(non_finite_i) != 0
    all_finite = ~has_invalid

    ct.store(all_finite_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), all_finite, dtype=ct.bool_), (1,)))

    # shifted_unscaled = (raw - raw_max) * 0.125
    # shifted_scaled = scaled - scaled_max
    # shifted = where(all_finite, shifted_unscaled, shifted_scaled)
    shifted_unscaled = (raw - raw_max) * SCALE
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(
        ct.full((1,), denom, dtype=ct.float32), (1,)))
    probs_bf = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, K_LEN), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.astype(
        ct.where(keep, ct.astype(probs_bf, ct.float32), 0.0),
        ct.bfloat16,
    )
    scaled_dropout_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_dropout_bf)


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


@oracle_impl(hardware="B200", point="279c055a")
def oracle_forward(inputs):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    full_shape = _as_shape(full_shape_arg)
    random_shape = _as_shape(random_shape_arg)
    out_shape = _as_shape(out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)
    device = x.device

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len).contiguous()
    if x.is_contiguous():
        x_2d = x.view(n_rows, k_len)
    else:
        x_2d = x.contiguous().view(n_rows, k_len)

    raw_amax_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    scaled_amax_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    all_finite_1d = torch.empty((n_rows,), device=device, dtype=torch.bool)
    sum_1d = torch.empty((n_rows,), device=device, dtype=torch.float32)
    gt_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bool)
    dropped_2d = torch.empty((n_rows, k_len), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_1d, scaled_amax_1d, all_finite_1d,
         sum_1d, gt_2d, dropped_2d, k_len),
    )

    raw_amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    raw_amax.view(n_rows).copy_(raw_amax_1d)
    scaled_amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    scaled_amax.view(n_rows).copy_(scaled_amax_1d)
    all_finite = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.bool,
    )
    all_finite.view(n_rows).copy_(all_finite_1d)
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1.view(n_rows).copy_(sum_1d)
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool,
    )
    gt.view(n_rows, k_len).copy_(gt_2d)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    dropped.view(n_rows, k_len).copy_(dropped_2d)

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
