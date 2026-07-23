"""cuTile port of amax_amax_any_7a1b8aec6077: LayoutLM scaled-attention softmax
+ dropout with raw/scaled amax side outputs + finite-row guard.

Ports the Triton `_scaled_softmax_dropout_kernel`. Pre-generates seeded
random tensor OUTSIDE the kernel; passes it as a f32 tensor into the cuTile
kernel where the bf16 dropout mask is derived.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 28
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,             # bf16 [N_ROWS, KLEN]
    random_ptr,        # f32 [N_ROWS, KLEN]
    raw_amax_ptr,      # f32 [N_ROWS]
    scaled_amax_ptr,   # f32 [N_ROWS]
    all_finite_ptr,    # bool [N_ROWS]
    sum_ptr,           # f32 [N_ROWS]
    gt_ptr,            # bool [N_ROWS, KLEN]
    dropped_ptr,       # bf16 [N_ROWS, KLEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    raw_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    raw = ct.astype(raw_bf, ct.float32)
    scaled_bf = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    raw_max = ct.max(raw, axis=1)
    scaled_max = ct.max(scaled, axis=1)

    # finite = (scaled == scaled) & (abs(scaled) != inf)
    is_nan = scaled != scaled
    # abs > large threshold as inf proxy — but inline abs+eq works too
    is_inf = (scaled == float("inf")) | (scaled == -float("inf"))
    is_invalid = is_nan | is_inf
    has_invalid = ct.max(ct.astype(is_invalid, ct.int32), axis=1) != 0
    all_finite = ~has_invalid

    raw_max_2d = ct.reshape(raw_max, (BLOCK_M, 1))
    scaled_max_2d = ct.reshape(scaled_max, (BLOCK_M, 1))
    shifted_unscaled = (raw - raw_max_2d) * 0.125
    shifted_scaled = scaled - scaled_max_2d
    all_finite_2d = ct.reshape(all_finite, (BLOCK_M, 1))
    shifted = ct.where(all_finite_2d, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs_bf = ct.astype(numer / denom_2d, ct.bfloat16)

    ct.store(raw_amax_ptr, index=(row_block,), tile=raw_max)
    ct.store(scaled_amax_ptr, index=(row_block,), tile=scaled_max)
    ct.store(all_finite_ptr, index=(row_block,), tile=all_finite)
    ct.store(sum_ptr, index=(row_block,), tile=denom)

    rand_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs_bf, zero_bf)
    scaled_out_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_out_bf)


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
    device = x.device

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    row_stride = _contiguous_stride(row_shape)
    full_stride = _contiguous_stride(full_shape)

    raw_amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    scaled_amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    all_finite = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.bool)
    sum_1 = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.bool)
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    x_2d = x.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    raw_amax_1d = raw_amax.view(n_rows)
    scaled_amax_1d = scaled_amax.view(n_rows)
    all_finite_1d = all_finite.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    # Post-adjustment: eager output all_finite is logical_not(any(~finite)). Our
    # kernel computes all_finite consistently. Also note eager returns
    # logical_not(any_1) which equals all_finite.

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_1d, scaled_amax_1d, all_finite_1d,
         sum_1d, gt_2d, dropped_2d, BLOCK_M, BLOCK_N),
    )
    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
