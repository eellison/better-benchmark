"""cuTile port of amax_amax_any_0ec44ac10279: LayoutLM scaled softmax + dropout.

bf16[384,512,512] attention pattern. Seed index 34. Point "279c055a".
Returns (raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0,2,1)).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 34


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,
    random_ptr,
    raw_amax_ptr,
    scaled_amax_ptr,
    all_finite_ptr,
    sum_ptr,
    gt_ptr,
    dropped_ptr,
    k_len: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    # Load random
    rand_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    # Scaled softmax
    raw = ct.astype(x, ct.float32)
    scaled_bf16 = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)

    # Compute raw and scaled max
    raw_max = ct.max(raw, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)

    # Finite check
    is_nan = scaled != scaled
    is_inf = (scaled == float("inf")) | (scaled == float("-inf"))
    finite = ~(is_nan | is_inf)
    has_invalid = ct.max(ct.where(~finite, ct.full((BLOCK_M, BLOCK_N), 1, dtype=ct.int32), ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.int32)), axis=1, keepdims=True)
    all_finite = has_invalid == 0

    # Shifted scores - use unscaled path if finite, else scaled
    shifted_unscaled = (raw - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    # Softmax
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    # Store side outputs
    ct.store(raw_amax_ptr, index=(row_block,), tile=ct.reshape(ct.max(raw_max, axis=1), (BLOCK_M,)))
    ct.store(scaled_amax_ptr, index=(row_block,), tile=ct.reshape(ct.max(scaled_max, axis=1), (BLOCK_M,)))
    ct.store(all_finite_ptr, index=(row_block,), tile=ct.reshape(ct.astype(all_finite, ct.bool_), (BLOCK_M,)))
    ct.store(sum_ptr, index=(row_block,), tile=ct.reshape(ct.max(denom, axis=1), (BLOCK_M,)))

    # Dropout
    rand_bf16 = ct.astype(rand_f, ct.bfloat16)
    dropout_p = ct.astype(ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf16 > dropout_p
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    dropped = ct.astype(ct.where(keep, ct.astype(probs, ct.float32), 0.0), ct.bfloat16)
    scaled_dropout = ct.astype(ct.astype(dropped, ct.float32) * 1.1111111111111112, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_dropout)


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


@oracle_impl(hardware="B200", point="279c055a", BLOCK_M=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
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

    raw_amax_1d = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    scaled_amax_1d = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    all_finite_1d = torch.empty((n_rows,), device=x.device, dtype=torch.bool)
    sum_1_1d = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    gt_2d = torch.empty((n_rows, k_len), device=x.device, dtype=torch.bool)
    dropped_2d = torch.empty((n_rows, k_len), device=x.device, dtype=torch.bfloat16)

    # Generate random outside kernel (eager fallback path)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape,
        seed,
        device=x.device,
    )
    random_flat = random.reshape(n_rows, k_len).contiguous()

    # Flatten x to 2D for kernel
    x_flat = x.reshape(n_rows, k_len).contiguous()

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_rows, BLOCK_M), 1, 1)
    ct.launch(
        stream, grid, _scaled_softmax_dropout_kernel,
        (x_flat, random_flat, raw_amax_1d, scaled_amax_1d, all_finite_1d, sum_1_1d, gt_2d, dropped_2d, k_len, BLOCK_M, BLOCK_N),
    )

    # Reshape 1D outputs to match expected shape
    raw_amax = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    raw_amax.view(n_rows).copy_(raw_amax_1d)

    scaled_amax = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    scaled_amax.view(n_rows).copy_(scaled_amax_1d)

    all_finite = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.bool,
    )
    all_finite.view(n_rows).copy_(all_finite_1d)

    sum_1 = torch.empty_strided(
        row_shape,
        row_stride,
        device=x.device,
        dtype=torch.float32,
    )
    sum_1.view(n_rows).copy_(sum_1_1d)

    # Reshape 2D outputs to match expected shapes
    gt = torch.empty_strided(
        full_shape,
        full_stride,
        device=x.device,
        dtype=torch.bool,
    )
    gt.view(n_rows, k_len).copy_(gt_2d)

    dropped = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )
    dropped.view(n_rows, k_len).copy_(dropped_2d)

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
