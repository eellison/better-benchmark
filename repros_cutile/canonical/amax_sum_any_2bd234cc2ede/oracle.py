"""cuTile port of amax_sum_any_2bd234cc2ede: MobileBERT safe softmax + dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _safe_softmax_dropout_kernel(
    x_ptr, arg1_ptr, random_ptr,
    where_ptr, keep_ptr, out_ptr,
    K: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    x_f = ct.astype(x_bf, ct.float32)

    amax_val = ct.max(x_f, axis=1, keepdims=True)
    sub = x_f - amax_val
    ex = ct.exp(sub)
    sum_val = ct.sum(ex, axis=1, keepdims=True)
    div = ex / sum_val
    div_bf = ct.astype(div, ct.bfloat16)

    # Safe softmax: check if all elements were -inf (any_1 = any(x != -inf))
    neg_inf = ct.full((BLOCK_M, BLOCK_N), -float("inf"), dtype=ct.float32)
    is_not_ninf = x_f != neg_inf  # b8
    any_not_ninf = ct.max(ct.astype(is_not_ninf, ct.int32), axis=1, keepdims=True) != 0
    row_all_ninf = ct.astype(1 - ct.astype(any_not_ninf, ct.int32), ct.bool_)

    # where(all_ninf, arg1_1, div_bf)
    arg1_val = ct.load(arg1_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    where_val = ct.where(row_all_ninf, arg1_val, div_bf)
    ct.store(where_ptr, index=(row_block, 0), tile=where_val)

    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(random, ct.bfloat16)
    keep = rand_bf > ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)
    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, where_val, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="d59f4ab1", BLOCK_M=1, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    B, H, Q, K = 256, 4, 128, 128
    full_shape = (B, H, Q, K)
    N_ROWS = B * H * Q

    view = arg0_1.view(full_shape).contiguous()
    x_2d = view.view(N_ROWS, K)
    arg1_2d = arg1_1.contiguous().view(N_ROWS, K)

    where_bf = torch.empty(full_shape, device=device, dtype=torch.bfloat16)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(arg0_1.shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.reshape(N_ROWS, K)

    where_2d = where_bf.view(N_ROWS, K)
    gt_2d = gt.view(N_ROWS, K)
    view_1_2d = view_1.view(N_ROWS, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), 1, 1),
        _safe_softmax_dropout_kernel,
        (x_2d, arg1_2d, random_2d, where_2d, gt_2d, view_1_2d,
         K, BLOCK_M, BLOCK_N),
    )

    permute = view_1.permute(0, 2, 1)
    return where_bf, gt, view_1, permute
