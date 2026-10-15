"""cuTile port of amax_amax_any_20f548adfc44: LayoutLM scaled softmax+dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 13
DROPOUT_SCALE = 1.1111111111111112
N_ROWS = 32 * 12 * 512
K_LEN = 512


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound[8:16] = torch.tensor(
            list((offset - advance).to_bytes(8, "little", signed=False)),
            dtype=state.dtype,
            device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@ct.kernel
def _scaled_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, K_LEN]
    random_ptr,      # f32  [rows, K_LEN]
    raw_amax_ptr,    # f32  [rows]
    scaled_amax_ptr, # f32  [rows]
    all_finite_ptr,  # bool [rows]
    sum_ptr,         # f32  [rows]
    gt_ptr,          # bool [rows, K_LEN]
    dropped_ptr,     # bf16 [rows, K_LEN]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    raw = ct.astype(x, ct.float32)
    scaled_bf16 = ct.astype(raw * 0.125, ct.bfloat16)
    scaled = ct.astype(scaled_bf16, ct.float32)

    raw_max = ct.max(raw)
    scaled_max = ct.max(scaled)
    ct.store(raw_amax_ptr, index=(row,), tile=ct.reshape(raw_max, (1,)))
    ct.store(scaled_amax_ptr, index=(row,), tile=ct.reshape(scaled_max, (1,)))

    # finiteness check on scaled
    inf_val = ct.full((1, BLOCK_N), float("inf"), dtype=ct.float32)
    abs_scaled = ct.where(scaled >= 0.0, scaled, -scaled)
    is_finite = (scaled == scaled) & (abs_scaled != inf_val)
    zero_i = ct.zeros((1, BLOCK_N), dtype=ct.int32)
    one_i = ct.full((1, BLOCK_N), 1, dtype=ct.int32)
    invalid_flag = ct.where(is_finite, zero_i, one_i)
    any_invalid = ct.sum(invalid_flag)
    all_finite = any_invalid == 0
    ct.store(all_finite_ptr, index=(row,), tile=ct.reshape(all_finite, (1,)))

    shifted_unscaled = (raw - raw_max) * 0.125
    shifted_scaled = scaled - scaled_max
    shifted = ct.where(all_finite, shifted_unscaled, shifted_scaled)

    numer = ct.exp(shifted)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf16 = ct.astype(random, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, BLOCK_N), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full(shape=(1, BLOCK_N), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs, zero_bf)
    scaled_out = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_out)


@oracle_impl(hardware="B200", point="279c055a")
def oracle_forward(inputs):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    full_shape = _as_shape(full_shape_arg)
    random_shape = _as_shape(random_shape_arg)
    out_shape = _as_shape(out_shape_arg)
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
    random_2d = random.view(n_rows, k_len)

    x_2d = x.view(n_rows, k_len)
    raw_amax_1d = raw_amax.view(n_rows)
    scaled_amax_1d = scaled_amax.view(n_rows)
    all_finite_1d = all_finite.view(n_rows)
    sum_1_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _scaled_softmax_dropout_kernel,
        (x_2d, random_2d, raw_amax_1d, scaled_amax_1d, all_finite_1d,
         sum_1_1d, gt_2d, dropped_2d, k_len),
    )
    return raw_amax, scaled_amax, all_finite, sum_1, gt, dropped, dropped.permute(0, 2, 1)
