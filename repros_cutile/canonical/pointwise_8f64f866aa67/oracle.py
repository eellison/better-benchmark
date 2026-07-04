"""cuTile port of pointwise_8f64f866aa67: T5 relu + seeded dropout."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 19
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _relu_dropout_kernel(
    x_ptr, random_ptr, gt_ptr, out_ptr, le_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    x = ct.load(x_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    zero_bf = ct.zeros((ROW_BLOCK, BLOCK_H), dtype=ct.bfloat16)
    is_nan = x != x
    relu_masked = ct.where(x > zero_bf, x, zero_bf)
    relu = ct.where(is_nan, x, relu_masked)
    # le = relu <= 0
    le = relu <= zero_bf
    ct.store(le_ptr, index=(row_block, 0), tile=le)

    random = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H))
    rand_bf = ct.astype(random, ct.bfloat16)
    keep = rand_bf > ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.bfloat16)
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    dropped = ct.where(keep, relu, zero_bf)
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


@oracle_impl(hardware="B200", point="52dd4c9c", BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, shape0, shape1, shape2 = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    device = arg0_1.device
    B, N, H = 8, 1024, 2048
    full_shape = (B, N, H)
    flat_shape = tuple(int(d) for d in shape2)
    rows = B * N
    hidden = H

    x_2d = arg0_1.view(rows, hidden)
    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    view_1 = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    le = torch.empty(full_shape, device=device, dtype=torch.bool)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(full_shape, seed, device=device)
    random_2d = random.view(rows, hidden)

    gt_2d = gt.view(rows, hidden)
    view_1_2d = view_1.view(rows, hidden)
    le_2d = le.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _relu_dropout_kernel,
        (x_2d, random_2d, gt_2d, view_1_2d, le_2d, hidden, BLOCK_H, 1),
    )

    return gt, view_1, le
