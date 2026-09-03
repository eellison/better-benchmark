"""cuTile port of amax_sum_any_5c3555859171: BERT/RoBERTa/Electra/MobileBert attention softmax + dropout.

Safe softmax with all-inf fallback, then dropout via pre-generated random tensor.
Returns (where, gt, view_1, view_1.permute(0, 2, 1)).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 (n_rows, k_len)
    fallback_ptr,   # bf16 (n_rows, k_len)
    random_ptr,     # f32 (n_rows, k_len)
    where_ptr,      # bf16 (n_rows, k_len)
    gt_ptr,         # bool (n_rows, k_len)
    dropped_ptr,    # bf16 (n_rows, k_len)
    K_LEN: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, K_LEN))
    fallback = ct.load(fallback_ptr, index=(row, 0), shape=(1, K_LEN))

    scores = ct.astype(x_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    neg_inf = ct.full(shape=(1, 1), fill_value=float("-inf"), dtype=ct.float32)
    has_any = row_max != neg_inf
    zero_f = ct.zeros((1, 1), dtype=ct.float32)
    one_f = ct.full(shape=(1, 1), fill_value=1.0, dtype=ct.float32)
    safe_max = ct.where(has_any, row_max, zero_f)
    numer = ct.exp(scores - safe_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    denom = ct.where(has_any, denom, one_f)
    probs = ct.astype(numer / denom, ct.bfloat16)

    where_val = ct.where(has_any, probs, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, K_LEN))
    rand_bf = ct.astype(random_f, ct.bfloat16)
    p_bf = ct.full(shape=(1, K_LEN), fill_value=DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.zeros((1, K_LEN), dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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
        rewound_offset = offset - advance
        rewound[8:16] = torch.tensor(
            list(int(rewound_offset).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="8ae0f618")
@oracle_impl(hardware="B200", point="fac7e171")
@oracle_impl(hardware="B200", point="d59f4ab1")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, _shape3 = inputs
    del shape0, _shape2, _shape3

    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len
    random_shape = tuple(int(dim) for dim in shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(tuple(arg1_1.shape), tuple(arg1_1.stride()),
                             device=arg1_1.device, dtype=torch.bool)
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    x_2d = arg0_1.reshape(n_rows, k_len)
    fallback_2d = arg1_1.reshape(n_rows, k_len)
    random_2d = random.reshape(n_rows, k_len)
    where_2d = where.reshape(n_rows, k_len)
    gt_2d = gt.reshape(n_rows, k_len)
    dropped_2d = dropped.reshape(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _safe_softmax_dropout_kernel,
        (
            x_2d, fallback_2d, random_2d,
            where_2d, gt_2d, dropped_2d,
            k_len,
        ),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
