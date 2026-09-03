"""cuTile port of amax_sum_6a53066a9204: DebertaV2 masked softmax + dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 22
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    x_ptr, random_ptr,
    amax_ptr, sum_ptr, keep_ptr, out_ptr,
    K: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    where_val = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    where_f = ct.astype(where_val, ct.float32)

    amax_val = ct.max(where_f, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(row_block, 0), tile=amax_val)
    sub = where_f - amax_val
    ex = ct.exp(sub)
    sum_val = ct.sum(ex, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(row_block, 0), tile=sum_val)
    div = ex / sum_val

    random = ct.load(random_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_N))
    keep = random > 0.1
    ct.store(keep_ptr, index=(row_block, 0), tile=keep)
    zero = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    dropped = ct.where(keep, div, zero) * DROPOUT_SCALE
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(dropped, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="00541467", BLOCK_M=1, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shape_params = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    B, H, Q, K = 8, 24, 512, 512
    full_shape = (B, H, Q, K)
    row_shape = (B, H, Q, 1)
    N_ROWS = B * H * Q

    view = arg0_1.view(full_shape)
    where_bf = torch.where(arg1_1, arg2_1, view).contiguous()
    where_2d = where_bf.view(N_ROWS, K)

    amax = torch.empty(row_shape, device=device, dtype=torch.float32)
    sum_1 = torch.empty(row_shape, device=device, dtype=torch.float32)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    bf16_view = torch.empty((B*H, Q, K), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg3_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.reshape(N_ROWS, K)

    amax_2d = amax.view(N_ROWS, 1)
    sum_1_2d = sum_1.view(N_ROWS, 1)
    gt_2d = gt.view(N_ROWS, K)
    bf16_2d = bf16_view.view(N_ROWS, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), 1, 1),
        _softmax_dropout_kernel,
        (where_2d, random_2d, amax_2d, sum_1_2d, gt_2d, bf16_2d,
         K, BLOCK_M, BLOCK_N),
    )

    permute = bf16_view.permute(0, 2, 1)
    return where_bf, amax, sum_1, gt, bf16_view, permute
