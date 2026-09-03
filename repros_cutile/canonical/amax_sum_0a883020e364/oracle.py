"""cuTile port of amax_sum_0a883020e364: MT5 attention softmax+dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 73
DROPOUT_SCALE = 1.1111111111111112


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
def _softmax_dropout_kernel(
    x_ptr, random_ptr,
    amax_ptr, sum_ptr, gt_ptr, dropped_ptr,
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN))
    scores = ct.astype(x, ct.float32)
    row_max = ct.max(scores)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    centered = scores - row_max
    numer = ct.exp(centered)
    denom = ct.sum(numer)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = ct.astype(numer * (1.0 / denom), ct.bfloat16)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf16 = ct.astype(random, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, K_LEN), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full(shape=(1, K_LEN), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, probs, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


@oracle_impl(hardware="B200", point="1715052e")
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

    amax = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided(row_shape, row_stride, device=device, dtype=torch.float32)
    gt = torch.empty_strided(full_shape, full_stride, device=device, dtype=torch.bool)
    dropped = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                  device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.view(n_rows, k_len)

    x_2d = x.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    sum_1_1d = sum_1.view(n_rows)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_kernel,
        (x_2d, random_2d, amax_1d, sum_1_1d, gt_2d, dropped_2d, k_len),
    )
    return amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
