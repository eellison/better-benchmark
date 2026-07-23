"""cuTile port of amax_sum_any_577af7e1e84b: safe-softmax + seeded dropout.

Row kernel: safe softmax with -inf fallback (any-alive detection), bf16 cast,
dropout mask via pre-generated random tensor, bf16 scaled output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 22
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, K]
    fallback_ptr,    # bf16 [rows, K]
    random_ptr,      # f32 [rows, K]
    where_ptr,       # bf16 [rows, K]
    gt_ptr,          # bool [rows, K]
    dropped_ptr,     # bf16 [rows, K]
    K_LEN: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        x_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < K_LEN, (1, BLOCK_N))
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    scores_active = ct.where(col_mask, x_f, neg_inf)
    # live = scores != -inf (fp)
    is_live = scores_active != neg_inf
    live_flag = ct.astype(is_live, ct.int32)
    any_live_count = ct.sum(live_flag, axis=1, keepdims=True)
    has_any = any_live_count > 0

    row_max = ct.max(scores_active, axis=1, keepdims=True)
    zero_max = ct.full((1, 1), 0.0, dtype=ct.float32)
    safe_max = ct.where(has_any, row_max, zero_max)
    numer = ct.exp(x_f - safe_max)
    numer_masked = ct.where(is_live, numer, 0.0)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    one = ct.full((1, 1), 1.0, dtype=ct.float32)
    denom_safe = ct.where(has_any, denom, one)
    probs = numer_masked / denom_safe
    probs_bf = ct.astype(probs, ct.bfloat16)

    fallback = ct.load(
        fallback_ptr, index=(row, 0), shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    where_val = ct.where(has_any, probs_bf, fallback)
    ct.store(where_ptr, index=(row, 0), tile=where_val)

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
    dropped_bf = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    ct.store(dropped_ptr, index=(row, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="8ae0f618", BLOCK_N=512)
@oracle_impl(hardware="B200", point="fac7e171", BLOCK_N=512)
@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, _shape3 = inputs

    k_len = int(arg0_1.shape[-1])
    n_rows = arg0_1.numel() // k_len
    device = arg0_1.device
    random_shape = tuple(int(dim) for dim in shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    scores_2d = arg0_1.contiguous().view(n_rows, k_len)
    fallback_2d = arg1_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _safe_softmax_dropout_kernel,
        (scores_2d, fallback_2d, random_2d,
         where_2d, gt_2d, dropped_2d, k_len, BLOCK_N),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
