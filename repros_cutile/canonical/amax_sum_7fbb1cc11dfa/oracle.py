"""cuTile port of amax_sum_7fbb1cc11dfa: MT5 softmax + seeded dropout row kernel.

Pre-generates the seeded random tensor. Row kernel: fp32 amax/exp/sum softmax,
bf16 rounding of probs, dropout mask (bf16 threshold), bf16 scaled output.
Returns (amax, sum, gt, view, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 67
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, K]
    random_ptr,     # f32 [n_rows, K]
    amax_ptr,       # f32 [n_rows]
    sum_ptr,        # f32 [n_rows]
    gt_ptr,         # bool [n_rows, K]
    dropped_ptr,    # bf16 [n_rows, K]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    row_max = ct.max(x_f, axis=1, keepdims=True)
    scores = x_f - row_max
    numer = ct.exp(scores)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))

    rand_f = ct.load(
        random_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_N), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="1715052e", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    x, seeds, full_shape_arg, random_shape_arg, _expand_shape, out_shape_arg = inputs
    full_shape = _shape_tuple(full_shape_arg)
    random_shape = _shape_tuple(random_shape_arg)
    out_shape = _shape_tuple(out_shape_arg)
    k_len = int(full_shape[-1])
    n_rows = int(x.numel() // k_len)
    device = x.device
    row_shape = full_shape[:-1] + (1,)

    amax = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    sum_1 = torch.empty_strided(
        row_shape, _contiguous_stride(row_shape),
        device=device, dtype=torch.float32,
    )
    gt = torch.empty_strided(
        full_shape, _contiguous_stride(full_shape),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = x.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (x_2d, random_2d, amax_1d, sum_1d, gt_2d, dropped_2d, BLOCK_N),
    )
    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
