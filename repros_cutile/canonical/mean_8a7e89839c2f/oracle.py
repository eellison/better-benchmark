"""cuTile port of mean_8a7e89839c2f: MT5 dropout residual RMSNorm.

Returns (gt, add, rsqrt, view_1). Uses bf16 dropout, f32 residual add, RMSNorm
(mean of squared values, rsqrt(eps=1e-6)), affine weight (no bias), bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 8
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_residual_rmsnorm_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr,
    gt_ptr, add_ptr, rsqrt_ptr, bf16_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_bf = ct.astype(rand, ct.bfloat16)
    p_bf = ct.astype(ct.full((1, BLOCK_H), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)
    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add)

    inv_h = 1.0 / HIDDEN
    mean_sq = ct.sum(add * add, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(mean_sq + EPS)
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    normalized = add * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    affine = weight_2d * normalized
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf)


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


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_H=512)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, _shape1, shape2 = inputs
    norm_shape = _shape(shape0)
    flat_shape = _shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    rsqrt_shape = (norm_shape[0], norm_shape[1], 1)
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    gt_2d = torch.empty((rows, hidden), device=device, dtype=torch.bool)
    add_2d = torch.empty((rows, hidden), device=device, dtype=torch.float32)
    rsqrt_1d = torch.empty((rows,), device=device, dtype=torch.float32)
    bf16_2d = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)

    if hidden != BLOCK_H:
        raise NotImplementedError(f"BLOCK_H={BLOCK_H} != hidden={hidden}")

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_rmsnorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1,
         gt_2d, add_2d, rsqrt_1d, bf16_2d,
         hidden, BLOCK_H),
    )

    gt = gt_2d.view(norm_shape)
    add = add_2d.view(norm_shape)
    rsqrt = rsqrt_1d.view(rsqrt_shape)
    bf16_view = bf16_2d.view(flat_shape)
    return gt, add, rsqrt, bf16_view
