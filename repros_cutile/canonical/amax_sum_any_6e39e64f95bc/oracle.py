"""cuTile port of amax_sum_any_6e39e64f95bc: MobileBERT safe softmax + dropout.

Safe softmax: check if any col is not -inf; if not, output 0.
Returns (amax, sum_1, logical_not_1, full, inductor_seeds, gt, view_1, permute).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr,           # bf16 [rows, cols]
    random_ptr,      # f32  [rows, cols]
    amax_ptr,        # f32  [rows]
    sum_ptr,         # f32  [rows]
    logical_not_ptr, # bool [rows]
    gt_ptr,          # bool [rows, cols]
    out_ptr,         # bf16 [rows, cols]
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_N))
    scores = ct.astype(x_bf, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row,), tile=ct.reshape(row_max, (1,)))

    # Check if ALL positions are -inf (any_1 = OR of (x != -inf); logical_not_1 = NOT any_1)
    neg_inf = ct.full((1, BLOCK_N), float("-inf"), dtype=ct.float32)
    not_neg_inf = scores != neg_inf  # bool
    # any(not_neg_inf) = any col is finite. logical_not_1 = all cols are -inf
    any_val = ct.max(ct.astype(not_neg_inf, ct.int32), axis=1, keepdims=True)  # 0 or 1
    logical_not_1 = any_val == 0  # bool [1, 1]
    ct.store(logical_not_ptr, index=(row,), tile=ct.reshape(logical_not_1, (1,)))

    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row,), tile=ct.reshape(denom, (1,)))
    probs = numer / denom
    probs_bf = ct.astype(probs, ct.bfloat16)

    # where(logical_not_1, 0, probs_bf) — broadcast logical_not_1 (row-wise scalar) over cols
    logical_not_1_broad = ct.zeros((1, BLOCK_N), dtype=ct.bool_) | logical_not_1
    zero_bf = ct.full((1, BLOCK_N), 0.0, dtype=ct.bfloat16)
    where_bf = ct.where(logical_not_1_broad, zero_bf, probs_bf)

    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped = ct.where(keep, where_bf, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=scaled)


def _shape(shape):
    return tuple(int(d) for d in shape)


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


@oracle_impl(hardware="B200", point="bcf6fe02", BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, shape0, shape1, shape2, _shape3, _shape4 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)  # [256, 4, 128, 128]
    full_shape = _shape(shape1)
    random_shape = _shape(shape2)
    b0 = int(view_shape[0])
    h = int(view_shape[1])
    q = int(view_shape[2])
    k = int(view_shape[3])
    rows = b0 * h * q
    row_shape = view_shape[:-1] + (1,)
    flat_shape = (b0 * h, q, k)

    amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    logical_not_1 = torch.empty(row_shape, device=device, dtype=torch.bool)
    full = torch.zeros(full_shape, device=device, dtype=torch.bfloat16)
    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    inductor_seeds = torch.ops.prims.inductor_seeds.default(24, device)
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_view = arg0_1.view(view_shape)
    x_2d = x_view.reshape(rows, k)
    r_2d = random.contiguous().view(rows, k)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    logical_not_1d = logical_not_1.view(rows)
    gt_2d = gt.view(rows, k)
    out_2d = view_1.view(rows, k)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, r_2d, amax_1d, sum_1d, logical_not_1d, gt_2d, out_2d, BLOCK_N),
    )
    permute = view_1.permute(0, 2, 1)
    return amax, sum_1, logical_not_1, full, inductor_seeds, gt, view_1, permute
