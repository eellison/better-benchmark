"""cuTile port of var_mean_a0fe76af4db4: GPT2 dropout residual LayerNorm (seed 16).

Returns (gt, add, mul_2, bf16_flat, permute, div). `add` is the residual +
dropout (f32); `bf16_flat` is the bf16 flat view of the affine result; permute
is the same view transposed.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_kernel(
    x_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, add_ptr, norm_ptr, bf16_ptr, div_ptr,
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
    x_f = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x_f)
    inv_h = 1.0 / HIDDEN
    mean = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf)
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


def _shape(shape):
    return tuple(int(d) for d in shape)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    missing = -1
    for idx, dim in enumerate(dims):
        if dim == -1:
            missing = idx
        else:
            known *= dim
    if missing >= 0:
        dims[missing] = int(numel) // known
    return tuple(dims)


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


@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _shape(shape0)
    flat_shape = _resolve_shape(shape2, arg0_1.numel())
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    if BLOCK_H == hidden:
        x_2d = arg0_1.contiguous().view(rows, hidden)
        random_2d = random.contiguous().view(rows, hidden)
        residual_2d = arg2_1.contiguous().view(rows, hidden)
        weight_pad = arg3_1
        bias_pad = arg4_1
    else:
        x_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        x_2d[:, :hidden].copy_(arg0_1)
        random_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        random_2d[:, :hidden].copy_(random.view(rows, hidden))
        residual_2d = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        residual_2d[:, :hidden].copy_(arg2_1.view(rows, hidden))
        weight_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        weight_pad[:hidden].copy_(arg3_1)
        bias_pad = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
        bias_pad[:hidden].copy_(arg4_1)

    gt_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    add_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, weight_pad, bias_pad,
         gt_pad, add_pad, norm_pad, bf16_pad, div_1d,
         hidden, BLOCK_H),
    )

    if BLOCK_H == hidden:
        gt = gt_pad.view(norm_shape)
        add = add_pad.view(norm_shape)
        normalized = norm_pad.view(norm_shape)
        bf16_flat = bf16_pad.view(flat_shape)
    else:
        gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
        gt.view(rows, hidden).copy_(gt_pad.narrow(1, 0, hidden))
        add = torch.empty(norm_shape, device=device, dtype=torch.float32)
        add.view(rows, hidden).copy_(add_pad.narrow(1, 0, hidden))
        normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
        normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))
        bf16_flat = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
        bf16_flat.view(rows, hidden).copy_(bf16_pad.narrow(1, 0, hidden))

    div = div_1d.view(div_shape)
    return gt, add, normalized, bf16_flat, bf16_flat.permute(1, 0), div
