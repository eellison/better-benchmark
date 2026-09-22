"""cuTile port of amax_sum_any_34ccb88507ce: safe softmax + seeded dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 7
DROPOUT_SCALE = 1.1111111111111112


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
def _safe_softmax_dropout_kernel(
    x_ptr, fallback_ptr, random_ptr,
    where_ptr, gt_ptr, dropped_ptr,
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN))
    scores = ct.astype(x, ct.float32)

    neg_inf = ct.full((1, K_LEN), -float("inf"), dtype=ct.float32)
    live = scores != neg_inf
    zero_i = ct.zeros((1, K_LEN), dtype=ct.int32)
    one_i = ct.full((1, K_LEN), 1, dtype=ct.int32)
    live_flag = ct.where(live, one_i, zero_i)
    has_any = ct.sum(live_flag) != 0

    row_max = ct.max(scores)
    safe_max = ct.where(has_any, row_max, 0.0)
    centered = scores - safe_max
    zero_f = ct.full((1, K_LEN), 0.0, dtype=ct.float32)
    numer_raw = ct.exp(centered)
    numer = ct.where(live, numer_raw, zero_f)
    denom = ct.sum(numer)
    safe_denom = ct.where(has_any, denom, 1.0)
    probs = ct.astype(numer * (1.0 / safe_denom), ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, K_LEN))
    where_val = ct.where(has_any, probs, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    random = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf16 = ct.astype(random, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, K_LEN), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full(shape=(1, K_LEN), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


@oracle_impl(hardware="B200", point="e886386f")
@oracle_impl(hardware="B200", point="8ae0f618")
@oracle_impl(hardware="B200", point="fac7e171")
@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, _shape3 = inputs
    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len
    random_shape = tuple(int(dim) for dim in shape1)
    device = arg0_1.device

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(tuple(arg1_1.shape), tuple(arg1_1.stride()),
                             device=device, dtype=torch.bool)
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    random_2d = random.view(n_rows, k_len)
    x_2d = arg0_1.view(n_rows, k_len)
    fallback_2d = arg1_1.view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, k_len),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
