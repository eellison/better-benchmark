"""cuTile port of amax_amax_any_50e98af7b82f: LayoutLM scaled softmax + dropout.

The Repro computes (i) row-max of scaled and unscaled bf16 scores, (ii) an
all-finite guard, (iii) shifted natural exp softmax that falls back to the
scaled shifting when non-finite rows exist, (iv) seeded dropout.

Seeded on-device RNG is replaced with `torch.ops.prims.inductor_random.default`,
matching the Triton oracle's `_inductor_random_for_eager_check` fallback path.
Round-to-nearest bf16 rounding is cuTile's default so the inline PTX
`mul.rn.f32` calls collapse to plain `*` on fp32.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 25
DROPOUT_SCALE = 1.1111111111111112


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
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    raw_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    raw = ct.astype(raw_bf, ct.float32)
    scaled_bf = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    raw_max = ct.max(raw, axis=1, keepdims=True)
    scaled_max = ct.max(scaled, axis=1, keepdims=True)

    # finite check: scaled is finite (not NaN and not +/-inf).
    is_nan = scaled != scaled
    is_inf = (ct.astype(scaled, ct.float32) * 0.0) != 0.0  # nan check for inf/nan; but we'd prefer explicit
    # More explicit: finite = (scaled == scaled) & (|scaled| != inf).
    abs_scaled = ct.astype(scaled, ct.float32)
    # Use abs
    abs_scaled = ct.where(abs_scaled < 0.0, -abs_scaled, abs_scaled)
    inf = float("inf")
    finite = (~is_nan) & (abs_scaled != inf)
    has_invalid = ct.max(ct.astype(~finite, ct.int32), axis=1, keepdims=True) != 0
    all_finite = ~has_invalid  # shape [BLOCK_M, 1]

    shifted_unscaled = (raw - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.store(raw_amax_ptr, index=(row_block,), tile=ct.reshape(raw_max, (BLOCK_M,)))
    ct.store(scaled_amax_ptr, index=(row_block,), tile=ct.reshape(scaled_max, (BLOCK_M,)))
    ct.store(all_finite_ptr, index=(row_block,), tile=ct.reshape(all_finite, (BLOCK_M,)))
    ct.store(sum_ptr, index=(row_block,), tile=ct.reshape(denom, (BLOCK_M,)))

    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(random, ct.bfloat16)
    dropout_p_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32), ct.bfloat16
    )
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_bf)


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
    advance = (numel + 131071) // 131072
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

    full_shape = tuple(int(d) for d in full_shape_arg)
    random_shape = tuple(int(d) for d in random_shape_arg)
    out_shape = tuple(int(d) for d in out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    row_shape = full_shape[:-1] + (1,)

    device = x.device
    raw_amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32,
    )
    scaled_amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32,
    )
    all_finite = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.bool,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape), device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape), device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape), device=device, dtype=torch.bfloat16,
    )

    x_2d = x.reshape(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)
    raw_amax_1d = raw_amax.view(n_rows)
    scaled_amax_1d = scaled_amax.view(n_rows)
    all_finite_1d = all_finite.view(n_rows)
    sum_1d = sum_1.view(n_rows)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len)

    if n_rows % BLOCK_M != 0:
        raise NotImplementedError(f"BLOCK_M={BLOCK_M} doesn't divide n_rows={n_rows}")
    if k_len != BLOCK_N:
        raise NotImplementedError(f"BLOCK_N={BLOCK_N} != k_len={k_len}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows // BLOCK_M, 1, 1),
        _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_1d, scaled_amax_1d, all_finite_1d, sum_1d,
         gt_2d, dropped_2d, BLOCK_M, BLOCK_N),
    )

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
