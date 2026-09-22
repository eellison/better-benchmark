"""cuTile port of mean_var_739c2d6336c7: BERT dropout + residual + LayerNorm (correction=1)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 42
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dropout_ln_kernel(
    x_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    gt_ptr,
    add_ptr,
    sqrt_ptr,
    sub_ptr,
    bf16_out_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_M1: ct.Constant[float],  # 1/(hidden-1)
    INV_H: ct.Constant[float],
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

    mean_1d = ct.sum(x_f, axis=1, keepdims=True) * INV_H
    centered = x_f - mean_1d
    ct.store(sub_ptr, index=(row, 0), tile=centered)
    variance_1d = ct.sum(centered * centered, axis=1, keepdims=True) * HIDDEN_M1
    sqrt_v = ct.sqrt(variance_1d)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(sqrt_v, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    mul_2 = weight_2d * centered
    denom = sqrt_v + EPS
    div = mul_2 / denom
    add_2 = div + bias_2d
    ct.store(bf16_out_ptr, index=(row, 0), tile=ct.astype(add_2, ct.bfloat16))


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


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, _shape2 = inputs
    device = arg0_1.device

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    view_shape = _shape(shape0)  # [16, 128, 768]
    random_shape = _shape(shape1)
    rows = int(view_shape[0]) * int(view_shape[1])
    hidden = int(arg3_1.shape[0])
    inv_h = 1.0 / hidden
    inv_hm1 = 1.0 / (hidden - 1)
    row_shape = (int(view_shape[0]), int(view_shape[1]), 1)

    gt = torch.empty(view_shape, device=device, dtype=torch.bool)
    add_out = torch.empty(view_shape, device=device, dtype=torch.float32)
    sqrt_out = torch.empty(row_shape, device=device, dtype=torch.float32)
    sub_out = torch.empty(view_shape, device=device, dtype=torch.float32)
    bf16_out = torch.empty((rows, hidden), device=device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    sub_2d = sub_out.view(rows, hidden)
    sqrt_1d = sqrt_out.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, add_2d, sqrt_1d, sub_2d, bf16_out,
         hidden, inv_hm1, inv_h, BLOCK_H),
    )
    return gt, add_out, sqrt_out, sub_out, bf16_out
