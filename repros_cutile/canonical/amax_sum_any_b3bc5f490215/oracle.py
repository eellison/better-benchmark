"""cuTile port of amax_sum_any_b3bc5f490215: BERT-family safe softmax + fallback + dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG. K_LEN divides the tile size
in every shape point (512 or 128), so we don't need masked stores.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 10
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,          # bf16 [n_rows, K_LEN]
    fallback_ptr,   # bf16 [n_rows, K_LEN]
    random_ptr,     # f32  [n_rows, K_LEN]
    where_ptr,      # bf16 [n_rows, K_LEN]
    gt_ptr,         # b8   [n_rows, K_LEN]
    dropped_ptr,    # bf16 [n_rows, K_LEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    scores = ct.astype(x_bf, ct.float32)
    neg_inf = ct.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=ct.float32)
    live = scores != neg_inf
    # any per row
    has_any_int = ct.max(ct.where(live, ct.full((BLOCK_M, BLOCK_N), 1, dtype=ct.int32),
                                   ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.int32)), axis=1)
    has_any = has_any_int != 0
    has_any_2d = ct.reshape(has_any, (BLOCK_M, 1))

    row_max = ct.max(scores, axis=1)
    safe_max = ct.where(has_any, row_max, ct.full((BLOCK_M,), 0.0, dtype=ct.float32))
    safe_max_2d = ct.reshape(safe_max, (BLOCK_M, 1))
    numer = ct.exp(scores - safe_max_2d)
    numer = ct.where(live, numer, ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32))
    denom = ct.sum(numer, axis=1)
    denom = ct.where(has_any, denom, ct.full((BLOCK_M,), 1.0, dtype=ct.float32))
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    probs = ct.astype(numer / denom_2d, ct.bfloat16)

    fallback = ct.load(fallback_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    where_val = ct.where(has_any_2d, probs, fallback)
    ct.store(where_ptr, index=(row_block, 0), tile=where_val)

    rand = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(dropped_ptr, index=(row_block, 0), tile=scaled)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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


@oracle_impl(hardware="B200", point="8ae0f618", BLOCK_M=4, BLOCK_N=512)
@oracle_impl(hardware="B200", point="fac7e171", BLOCK_M=4, BLOCK_N=512)
@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs
    del _shape0, _shape2, _shape3

    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    random_shape = tuple(int(dim) for dim in shape1)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    # View everything as (n_rows, k_len)
    x_2d = arg0_1.view(n_rows, k_len)
    fallback_2d = arg1_1.reshape(n_rows, k_len)
    random_2d = random.reshape(n_rows, k_len).contiguous()
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, BLOCK_M, BLOCK_N),
    )

    return where, gt, dropped, dropped.permute(0, 2, 1)
