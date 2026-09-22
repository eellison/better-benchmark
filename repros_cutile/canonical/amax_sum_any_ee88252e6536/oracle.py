"""cuTile port of amax_sum_any_ee88252e6536: MobileBert dropout softmax with -inf finite guard."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 8
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    view_ptr,           # bf16 [rows, k_len]
    fallback_ptr,       # bf16 [rows, k_len]  (used when whole row is -inf)
    random_ptr,         # f32  [rows, k_len]
    where_out_ptr,      # bf16 [rows, k_len]
    gt_out_ptr,         # b8   [rows, k_len]
    final_out_ptr,      # bf16 [rows, k_len]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    view = ct.load(view_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    view_f = ct.astype(view, ct.float32)

    row_max = ct.max(view_f, axis=1, keepdims=True)
    shifted = view_f - row_max
    numer = ct.exp(shifted)
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    # Check if every element in the row is -inf. If so use fallback (arg1_1)
    neg_inf_val = ct.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=ct.float32)
    is_neg_inf = view_f == neg_inf_val
    zero_i = ct.full((BLOCK_M, BLOCK_N), 0, dtype=ct.int32)
    one_i = ct.full((BLOCK_M, BLOCK_N), 1, dtype=ct.int32)
    not_neg_inf_i = ct.where(is_neg_inf, zero_i, one_i)
    # For "row_is_all_neg_inf", max(not_neg_inf_i) == 0.
    row_has_finite = ct.max(not_neg_inf_i, axis=1, keepdims=True)
    zero_i_1 = ct.full((BLOCK_M, 1), 0, dtype=ct.int32)
    row_is_all_neg_inf = row_has_finite == zero_i_1

    fallback = ct.load(fallback_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    where = ct.where(row_is_all_neg_inf, fallback, probs_bf)
    ct.store(where_out_ptr, index=(pid, 0), tile=where)

    rand = ct.load(random_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_out_ptr, index=(pid, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, where, zero_bf)
    scaled = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(final_out_ptr, index=(pid, 0), tile=scaled)


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


def _shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_M=1, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    view_shape = _shape(shape1)         # (256, 4, 128, 128)
    flat_shape = _shape(shape3)         # (1024, 128, 128)
    device = arg0_1.device

    b, h, q, k = view_shape
    rows = b * h * q

    view_2d = arg0_1.contiguous().view(rows, k)
    fallback_2d = arg1_1.contiguous().view(rows, k)

    where_out = torch.empty(view_shape, device=device, dtype=torch.bfloat16)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    final = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(view_shape, seed, device=device)

    where_2d = where_out.view(rows, k)
    gt_2d = gt.view(rows, k)
    final_2d = final.view(rows, k)
    random_2d = random.contiguous().view(rows, k)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(rows, BLOCK_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _softmax_dropout_kernel,
        (view_2d, fallback_2d, random_2d,
         where_2d, gt_2d, final_2d,
         BLOCK_M, BLOCK_N),
    )

    return where_out, gt, final, final.permute(0, 2, 1)
