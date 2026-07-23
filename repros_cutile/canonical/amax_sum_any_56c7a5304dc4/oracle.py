"""cuTile port of amax_sum_any_56c7a5304dc4: MobileBERT safe softmax + dropout.

Ports the Triton `_safe_softmax_dropout_random_kernel`. When a row is entirely
-inf, the softmax falls back to a supplied bf16 fallback tensor. Seeded
on-device RNG replaced with pre-generated `torch.ops.prims.inductor_random`.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 15
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,        # bf16 [n_rows, k_len]
    fallback_ptr, # bf16 [n_rows, k_len]
    random_ptr,   # f32  [n_rows, k_len]
    where_ptr,    # bf16 [n_rows, k_len]
    gt_ptr,       # b8   [n_rows, k_len]
    dropped_ptr,  # bf16 [n_rows, k_len]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    scores_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scores = ct.astype(scores_bf, ct.float32)

    minus_inf = float("-inf")
    live_bool = scores != minus_inf
    live_i32 = ct.astype(live_bool, ct.int32)
    has_any = ct.max(live_i32, axis=1, keepdims=True) != 0  # [BLOCK_M, 1]

    row_max = ct.max(scores, axis=1, keepdims=True)
    zero_f = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)
    safe_max = ct.where(has_any, row_max, zero_f)
    numer = ct.exp(scores - safe_max)
    zero_grid = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    numer_masked = ct.where(live_bool, numer, zero_grid)
    denom = ct.sum(numer_masked, axis=1, keepdims=True)
    one_f = ct.full((BLOCK_M, 1), 1.0, dtype=ct.float32)
    denom_safe = ct.where(has_any, denom, one_f)
    probs = ct.astype(numer_masked / denom_safe, ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    where_val = ct.where(has_any, probs, fallback)
    ct.store(where_ptr, index=(row_block, 0), tile=where_val)

    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(random, ct.bfloat16)
    p_bf = ct.astype(ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where_val, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled_bf)


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


@oracle_impl(hardware="B200", point="d59f4ab1", block_m=8, block_n=128)
def oracle_forward(inputs, *, block_m: int, block_n: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs
    del _shape0, _shape2, _shape3

    device = arg0_1.device
    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    random_shape = tuple(int(d) for d in shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    x_2d = arg0_1.reshape(n_rows, k_len)
    fallback_2d = arg1_1.reshape(n_rows, k_len)
    where_2d = where.reshape(n_rows, k_len)
    gt_2d = gt.reshape(n_rows, k_len)
    dropped_2d = dropped.reshape(n_rows, k_len)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_2d = random.reshape(n_rows, k_len)

    if n_rows % block_m != 0:
        raise NotImplementedError(f"block_m={block_m} doesn't divide n_rows={n_rows}")
    if k_len != block_n:
        raise NotImplementedError(f"block_n={block_n} != k_len={k_len}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows // block_m, 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, block_m, block_n),
    )

    return where, gt, dropped, dropped.permute(0, 2, 1)
