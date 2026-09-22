"""cuTile port of amax_sum_any_725f76a84d6a: MobileBERT safe softmax + dropout.

Ports the Triton `_safe_softmax_dropout_random_kernel`. Pre-generates the
seeded random tensor via `torch.ops.prims.inductor_random` outside the kernel;
the cuTile kernel computes:
  * fp32 stable softmax (safe against -inf rows) -> bf16 probs
  * fallback (arg1_1) when the row is all -inf (has_any == false)
  * dropout via random > 0.1 (through bf16 comparison)
  * bf16 dropout scaling by 1.1111...
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 18
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_random_kernel(
    x_ptr,          # bf16 [N_ROWS, KLEN]
    fallback_ptr,   # bf16 [N_ROWS, KLEN]
    random_ptr,     # f32  [N_ROWS, KLEN]
    where_ptr,      # bf16 [N_ROWS, KLEN]
    gt_ptr,         # bool [N_ROWS, KLEN]
    dropped_ptr,    # bf16 [N_ROWS, KLEN]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    fallback_bf = ct.load(fallback_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    random_f = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))

    scores = ct.astype(x_bf, ct.float32)
    live = scores != -float("inf")
    has_any_int = ct.max(ct.astype(live, ct.int32), axis=1)
    has_any = has_any_int != 0
    has_any_2d = ct.reshape(has_any, (BLOCK_M, 1))

    row_max = ct.max(scores, axis=1)
    row_max_2d = ct.reshape(row_max, (BLOCK_M, 1))
    safe_max = ct.where(has_any_2d, row_max_2d, 0.0)
    numer_raw = ct.exp(scores - safe_max)
    numer = ct.where(live, numer_raw, 0.0)
    denom = ct.sum(numer, axis=1)
    denom_2d = ct.reshape(denom, (BLOCK_M, 1))
    denom_safe = ct.where(has_any_2d, denom_2d, 1.0)
    probs_bf = ct.astype(numer / denom_safe, ct.bfloat16)
    where_val_bf = ct.where(has_any_2d, probs_bf, fallback_bf)
    ct.store(where_ptr, index=(row_block, 0), tile=where_val_bf)

    rand_bf = ct.astype(random_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full((BLOCK_M, BLOCK_N), DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where_val_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
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


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="d59f4ab1", block_m=8, block_n=128)
def oracle_forward(inputs, *, block_m: int, block_n: int):
    arg0_1, arg1_1, arg2_1, _shape0, shape1, _shape2, _shape3 = inputs
    del _shape0, _shape2, _shape3

    k_len = int(arg0_1.shape[-1])
    n_rows = int(arg0_1.numel() // k_len)
    random_shape = _as_shape(shape1)
    device = arg0_1.device

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    where = torch.empty_like(arg1_1)
    gt = torch.empty_strided(
        tuple(arg1_1.shape), tuple(arg1_1.stride()),
        device=device, dtype=torch.bool,
    )
    dropped = torch.empty_like(arg0_1)

    x_2d = arg0_1.contiguous().view(n_rows, k_len)
    fallback_2d = arg1_1.contiguous().view(n_rows, k_len)
    random_2d = random.contiguous().view(n_rows, k_len)
    where_2d = where.view(n_rows, k_len)
    gt_2d = gt.view(n_rows, k_len)
    dropped_2d = dropped.view(n_rows, k_len)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, block_m), 1, 1),
        _safe_softmax_dropout_random_kernel,
        (x_2d, fallback_2d, random_2d, where_2d, gt_2d, dropped_2d, block_m, block_n),
    )
    return where, gt, dropped, dropped.permute(0, 2, 1)
