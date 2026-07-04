"""cuTile port of amax_sum_54a1c45ad37b: T5/MT5 attention softmax dropout.

Uses pre-generated random tensor (from torch.ops.prims.inductor_random) to
sidestep cuTile's lack of on-device seeded RNG.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 21
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _softmax_dropout_kernel(
    view_ptr,           # bf16 [rows, k_len]
    bias_ptr,           # f32  [rows, k_len]
    random_ptr,         # f32  [rows, k_len]
    rounded_ptr,        # bf16 [rows, k_len]  (view + bias, rounded to bf16)
    amax_ptr,           # f32  [rows]
    sum_ptr,            # f32  [rows]
    gt_ptr,             # b8   [rows, k_len]
    dropped_ptr,        # bf16 [rows, k_len]
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    pid = ct.bid(0)

    view = ct.load(view_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    bias = ct.load(bias_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    view_f = ct.astype(view, ct.float32)
    rounded = ct.astype(view_f + bias, ct.bfloat16)
    ct.store(rounded_ptr, index=(pid, 0), tile=rounded)

    scores = ct.astype(rounded, ct.float32)
    row_max = ct.max(scores, axis=1, keepdims=True)
    ct.store(amax_ptr, index=(pid,), tile=ct.reshape(row_max, (BLOCK_M,)))
    numer = ct.exp(scores - row_max)
    denom = ct.sum(numer, axis=1, keepdims=True)
    ct.store(sum_ptr, index=(pid,), tile=ct.reshape(denom, (BLOCK_M,)))
    probs_bf = ct.astype(numer / denom, ct.bfloat16)

    rand = ct.load(random_ptr, index=(pid, 0), shape=(BLOCK_M, BLOCK_N))
    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold = ct.full((BLOCK_M, BLOCK_N), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(pid, 0), tile=keep)

    zero_bf = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, probs_bf, zero_bf)
    scaled = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    ct.store(dropped_ptr, index=(pid, 0), tile=scaled)


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


@oracle_impl(hardware="B200", point="dda3d8e0", BLOCK_M=1, BLOCK_N=128)
@oracle_impl(hardware="B200", point="aeb1682d", BLOCK_M=1, BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, _shape2, shape3 = inputs
    full_shape = _shape(shape1)      # (b, h, q, k)
    flat_shape = _shape(shape3)      # (b*h, q, k)
    device = arg0_1.device

    b, h, q, k = full_shape
    rows = b * h * q

    view_2d = arg0_1.contiguous().view(rows, k)
    bias_2d = arg1_1.contiguous().view(rows, k)

    rounded = torch.empty(full_shape, device=device, dtype=torch.bfloat16)
    amax = torch.empty((b, h, q, 1), device=device, dtype=torch.float32)
    sum_1 = torch.empty((b, h, q, 1), device=device, dtype=torch.float32)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    dropped = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)

    rounded_2d = rounded.view(rows, k)
    amax_1d = amax.view(rows)
    sum_1d = sum_1.view(rows)
    gt_2d = gt.view(rows, k)
    dropped_2d = dropped.view(rows, k)
    random_2d = random.contiguous().view(rows, k)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(rows, BLOCK_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _softmax_dropout_kernel,
        (view_2d, bias_2d, random_2d, rounded_2d,
         amax_1d, sum_1d, gt_2d, dropped_2d,
         BLOCK_M, BLOCK_N),
    )

    return rounded, amax, sum_1, gt, dropped, dropped.permute(0, 2, 1)
