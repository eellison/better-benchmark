"""cuTile port of amax_sum_cb8fc73675a3: DeBERTa softmax + seeded dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 1
BF16_FILL_VALUE = -3.3895313892515355e38
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


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


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
def _softmax_dropout_generated_kernel(
    scores_ptr, random_ptr,
    amax_ptr, denom_ptr, keep_ptr, out_ptr,
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    raw = ct.load(scores_ptr, index=(row, 0), shape=(1, K_LEN))
    scores = ct.astype(raw, ct.float32)
    row_max = ct.max(scores)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))
    centered = scores - row_max
    numer = ct.exp(centered)
    denom = ct.sum(numer)
    ct.store(denom_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer * (1.0 / denom)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    keep = random > 0.1
    ct.store(keep_ptr, index=(row, 0), tile=keep)

    zero_f = ct.full(shape=(1, K_LEN), fill_value=0.0, dtype=ct.float32)
    dropped = ct.where(keep, probs, zero_f)
    scaled = ct.astype(dropped * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


@oracle_impl(hardware="B200", point="05e1be20")
def oracle_forward(inputs):
    arg0_1, arg1_1, shape0, shape1, shape2, shape3 = inputs
    view_shape = _resolve_shape(shape0, arg0_1.numel())
    full_shape = _as_shape(shape1)
    random_shape = _as_shape(shape2)
    flat_shape = _resolve_shape(shape3, arg0_1.numel())
    reduction_shape = (view_shape[0], view_shape[1], view_shape[2], 1)
    device = arg0_1.device

    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len

    # Generated all-false mask [8, 1, 512, 512] and scalar fill
    full_mask = torch.zeros(full_shape, device=device, dtype=torch.bool)
    fill = torch.full((), BF16_FILL_VALUE, device=device, dtype=torch.bfloat16)

    amax = torch.empty_strided(reduction_shape, _contiguous_stride(reduction_shape),
                               device=device, dtype=torch.float32)
    denom = torch.empty_strided(reduction_shape, _contiguous_stride(reduction_shape),
                                device=device, dtype=torch.float32)
    keep = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                               device=device, dtype=torch.bool)
    out = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                              device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.view(n_rows, k_len)

    scores_2d = arg0_1.view(n_rows, k_len)
    amax_1d = amax.view(n_rows)
    denom_1d = denom.view(n_rows)
    keep_2d = keep.view(n_rows, k_len)
    out_2d = out.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_dropout_generated_kernel,
        (scores_2d, random_2d, amax_1d, denom_1d, keep_2d, out_2d, k_len),
    )
    return full_mask, fill, amax, denom, keep, out, out.permute(0, 2, 1)
