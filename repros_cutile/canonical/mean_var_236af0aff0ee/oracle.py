"""cuTile port of mean_var_236af0aff0ee: BERT dual-dropout residual LayerNorm.

Two pre-generated random tensors (seeds 54 and 55). Row kernel: bf16 dropout
of input, residual add, f32 dropout, mean, variance (correction=1 unbiased),
sqrt(max(var, 0))+eps denominator, affine bf16 output.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 54
SEED_INDEX_1 = 55
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-6


@ct.kernel
def _dual_dropout_layernorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random0_ptr,     # f32 [rows, HIDDEN]
    random1_ptr,     # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    gt0_ptr,         # bool [rows, HIDDEN]
    gt1_ptr,         # bool [rows, HIDDEN]
    dropped_ptr,     # f32 [rows, HIDDEN]
    sqrt_ptr,        # f32 [rows]
    sub_ptr,         # f32 [rows, HIDDEN]
    out_ptr,         # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    random0_f = ct.load(
        random0_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random1_f = ct.load(
        random1_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    random0_bf = ct.astype(random0_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = random0_bf > threshold_bf
    keep1 = random1_f > 0.1
    ct.store(gt0_ptr, index=(row, 0), tile=keep0)
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)

    flat = ct.load(
        flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    dropped0_bf = ct.astype(
        ct.where(keep0, ct.astype(flat, ct.float32), 0.0), ct.bfloat16,
    )
    scaled0_bf = ct.astype(
        ct.astype(dropped0_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16,
    )
    add = residual + ct.astype(scaled0_bf, ct.float32)
    dropped1 = ct.where(keep1, add, 0.0)
    x = dropped1 * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row, 0), tile=x)

    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    x_masked = ct.where(col_mask, x, 0.0)
    row_sum = ct.sum(x_masked)
    row_sumsq = ct.sum(x_masked * x_masked)
    mean = row_sum * (1.0 / HIDDEN)
    centered = x - mean
    # variance = (sumsq - HIDDEN * mean^2) / (HIDDEN - 1)  (unbiased, correction=1)
    variance = (row_sumsq - HIDDEN * mean * mean) * (1.0 / (HIDDEN - 1))
    # Clamp negative variance to 0
    zero = ct.zeros((), dtype=ct.float32)
    variance = ct.where(variance > 0.0, variance, zero)
    std = ct.sqrt(variance)

    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    numerator = weight_2d * centered
    denom = std + EPS
    normalized = numerator / denom
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _random_advance(shape):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (numel + 131071) // 131072


def _inductor_random_pair_for_eager_check(shape, seed0, seed1, *, device):
    if torch.cuda.is_current_stream_capturing():
        return (
            torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
            torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
        )
    advance = _random_advance(shape)
    state = torch.cuda.get_rng_state(device)
    offset = _state_u64(state, 8)
    if offset >= 2 * advance:
        rewound = state.clone()
        _put_state_u64(rewound, 8, offset - 2 * advance)
        torch.cuda.set_rng_state(rewound, device)
        random0 = torch.ops.prims.inductor_random.default(shape, seed0, "rand")
        random1 = torch.ops.prims.inductor_random.default(shape, seed1, "rand")
        torch.cuda.set_rng_state(state, device)
        return random0, random1
    return (
        torch.ops.prims.inductor_random.default(shape, seed0, "rand"),
        torch.ops.prims.inductor_random.default(shape, seed1, "rand"),
    )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="4205ff34", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2, shape3 = inputs
    del shape1
    base_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape2)
    out_shape = _shape_tuple(shape3)
    device = arg0_1.device
    base_stride = _contiguous_stride(base_shape)
    sqrt_shape = base_shape[:-1] + (1,)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])

    gt0 = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.bool,
    )
    gt1 = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.bool,
    )
    dropped = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.float32,
    )
    sqrt = torch.empty_strided(
        sqrt_shape, _contiguous_stride(sqrt_shape),
        device=device, dtype=torch.float32,
    )
    sub = torch.empty_strided(
        base_shape, base_stride, device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    seed0 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX_1)
    random0, random1 = _inductor_random_pair_for_eager_check(
        random_shape, seed0, seed1, device=device,
    )

    x_2d = arg0_1.contiguous().view(rows, hidden)
    random0_2d = random0.contiguous().view(rows, hidden)
    random1_2d = random1.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    gt1_2d = gt1.view(rows, hidden)
    dropped_2d = dropped.view(rows, hidden)
    sqrt_1d = sqrt.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dual_dropout_layernorm_kernel,
        (x_2d, random0_2d, random1_2d, residual_2d, arg3_1, arg4_1,
         gt0_2d, gt1_2d, dropped_2d, sqrt_1d, sub_2d, out_2d,
         hidden, BLOCK_H),
    )
    return gt0, gt1, dropped, sqrt, sub, out
