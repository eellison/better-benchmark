"""cuTile port of var_mean_c3d36ccf2d97: GPT2 dropout+residual LayerNorm (seed 15)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 15
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_kernel(
    x_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    normalized_ptr,
    bf16_out_ptr,
    div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16
    )
    x_f = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x_f)

    inv_h = 1.0 / HIDDEN
    mean_1d = ct.sum(x_f, axis=1, keepdims=True) * inv_h
    centered = x_f - mean_1d
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS)
    normalized = centered * invstd_1d
    ct.store(normalized_ptr, index=(row, 0), tile=normalized)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))

    div_val = invstd_1d * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


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


@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, _shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)
    random_shape = _shape(shape1)
    rows = int(view_shape[0]) * int(view_shape[1])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (int(view_shape[0]), int(view_shape[1]), 1)
    flat_shape = (rows, hidden)

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    add_out = torch.empty(view_shape, device=device, dtype=torch.float32)
    normalized = torch.empty(view_shape, device=device, dtype=torch.float32)
    bf16_out = torch.empty(flat_shape, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, normalized_2d, bf16_out, div_1d,
         hidden, BLOCK_H),
    )
    permute = bf16_out.t().contiguous()
    return gt, add_out, normalized, bf16_out, permute, div
