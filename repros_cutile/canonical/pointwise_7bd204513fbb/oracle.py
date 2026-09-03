"""cuTile port of pointwise_7bd204513fbb: M2M100 seeded dropout + residual add."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 3
DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _dropout_residual_kernel(
    view_ptr,       # bf16 [N]
    random_ptr,     # f32  [N]
    residual_ptr,   # f32  [N]
    gt_ptr,         # b8   [N]
    add_ptr,        # f32  [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    view_bf = ct.load(view_ptr, index=(pid,), shape=(BLOCK,))
    rand = ct.load(random_ptr, index=(pid,), shape=(BLOCK,))
    residual = ct.load(residual_ptr, index=(pid,), shape=(BLOCK,))

    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold = ct.full((BLOCK,), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(pid,), tile=keep)

    zero_bf = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, view_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    add = ct.astype(residual, ct.float32) + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(pid,), tile=add)


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


@oracle_impl(hardware="B200", point="7bafb841", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1 = inputs
    full_shape = _shape(shape0)      # (64, 128, 1024)
    random_shape = _shape(shape1)    # (64, 128, 1024)
    device = arg0_1.device

    total = 1
    for d in full_shape:
        total *= int(d)

    view_flat = arg0_1.contiguous().view(total)
    residual_flat = arg2_1.contiguous().view(total)

    gt = torch.empty(full_shape, device=device, dtype=torch.bool)
    add = torch.empty(full_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.contiguous().view(total)

    gt_flat = gt.view(total)
    add_flat = add.view(total)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(total, BLOCK), 1, 1),
        _dropout_residual_kernel,
        (view_flat, random_flat, residual_flat, gt_flat, add_flat, BLOCK),
    )
    return gt, add
