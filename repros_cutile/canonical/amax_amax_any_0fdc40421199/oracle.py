"""cuTile port of amax_amax_any_0fdc40421199: LayoutLM softmax+dropout.

Bf16 scaled attention softmax with dual amax outputs, finite-row guard, seeded
dropout, and permute alias. Seeded dropout uses pre-generated inductor_random.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 22
FILL_MIN = float("-inf")


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


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,       # bf16 [rows, K]
    rand_ptr,    # f32 [rows, K]
    raw_amax_ptr,      # f32 [rows]
    scaled_amax_ptr,   # f32 [rows]
    all_finite_ptr,    # b8  [rows]
    sum_ptr,           # f32 [rows]
    gt_ptr,            # b8 [rows, K]
    dropped_ptr,       # bf16 [rows, K]
    K: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
):
    row = ct.bid(0)
    raw = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_K))
    raw_f = ct.astype(raw, ct.float32)
    scaled_bf = ct.astype(raw_f * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf, ct.float32)

    raw_max = ct.max(raw_f, keepdims=True)
    scaled_max = ct.max(scaled, keepdims=True)
    ct.store(raw_amax_ptr, index=(row,),
             tile=ct.reshape(raw_max, (1,)))
    ct.store(scaled_amax_ptr, index=(row,),
             tile=ct.reshape(scaled_max, (1,)))

    # finite check
    is_finite = (scaled == scaled)
    inf_val = ct.full((1, BLOCK_K), float("inf"), dtype=ct.float32)
    neg_inf_val = ct.full((1, BLOCK_K), float("-inf"), dtype=ct.float32)
    is_not_pos_inf = scaled != inf_val
    is_not_neg_inf = scaled != neg_inf_val
    row_finite_tile = is_finite & is_not_pos_inf & is_not_neg_inf
    # any: max as int
    finite_int = ct.astype(row_finite_tile, ct.int32)
    finite_min = ct.min(finite_int, keepdims=True)  # 0 if any not-finite, 1 otherwise
    all_finite_scalar = finite_min == 1
    # Store b8 [1]
    ct.store(all_finite_ptr, index=(row,),
             tile=ct.reshape(all_finite_scalar, (1,)))

    all_finite_bcast = ct.astype(finite_min, ct.float32)  # 0 or 1
    shifted_unscaled = (raw_f - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite_scalar, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    rand_val = ct.load(rand_ptr, index=(row, 0), shape=(1, BLOCK_K))
    rand_bf = ct.astype(rand_val, ct.bfloat16)
    p_bf = ct.full((1, BLOCK_K), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_K), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs_bf, zero_bf)
    scaled_dropout = ct.astype(
        ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_dropout)


@oracle_impl(hardware="B200", point="279c055a")
def oracle_forward(inputs):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    full_shape = tuple(int(d) for d in full_shape_arg)   # [32, 12, 512, 512]
    random_shape = tuple(int(d) for d in random_shape_arg)
    out_shape = tuple(int(d) for d in out_shape_arg)    # [384, 512, 512]
    device = x.device

    K = full_shape[-1]  # 512
    total = int(x.numel())
    rows = total // K
    row_shape = full_shape[:-1] + (1,)

    raw_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    scaled_amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    all_finite = torch.empty(row_shape, device=device, dtype=torch.bool)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    dropped = torch.empty(out_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    # Reshape everything to 2D for the kernel
    x_2d = x.reshape(rows, K)
    rand_2d = random.reshape(rows, K)
    gt_2d = gt.view(rows, K)
    dropped_2d = dropped.reshape(rows, K)
    raw_amax_1d = raw_amax.view(rows)
    scaled_amax_1d = scaled_amax.view(rows)
    all_finite_1d = all_finite.view(rows)
    sum_1_1d = sum_1.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _scaled_softmax_dropout_kernel,
        (x_2d, rand_2d,
         raw_amax_1d, scaled_amax_1d, all_finite_1d, sum_1_1d,
         gt_2d, dropped_2d, K, K, 1.1111111111111112),
    )

    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
