"""cuTile port of mean_var_3d664201719d: BERT dropout residual LayerNorm (mean+var).

Uses correction=1 for var. Pre-generates seeded random. Returns gt/add/sqrt/sub/view.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 22
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
VAR_CORRECTION = 1.0
DENOM_EPS = 1.0e-6


@ct.kernel
def _dropout_mean_var_kernel(
    x_ptr,           # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN)
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    bias_ptr,        # f32 (HIDDEN,)
    gt_ptr,          # bool (rows, HIDDEN)
    add_ptr,         # f32 (rows, HIDDEN)
    sqrt_ptr,        # f32 (rows,)
    sub_ptr,         # f32 (rows, HIDDEN)
    out_ptr,         # bf16 (rows, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_2d = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)

    rand_bf = ct.astype(random_f, ct.bfloat16)
    p_bf = ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=add_val)

    add_masked = ct.where(col_mask_2d, add_val, zero_f)
    row_sum = ct.sum(add_masked, axis=1, keepdims=True)
    row_sum_sq = ct.sum(add_masked * add_masked, axis=1, keepdims=True)
    mean = row_sum * (1.0 / HIDDEN_F)
    centered = add_val - mean
    # variance = (row_sum_sq - HIDDEN*mean^2) / (HIDDEN - correction)
    variance = (row_sum_sq - HIDDEN_F * mean * mean) * (1.0 / (HIDDEN_F - VAR_CORRECTION))
    # Clamp to >=0 before sqrt to avoid NaN
    zero_scalar = ct.zeros((1, 1), dtype=ct.float32)
    variance = ct.where(variance > zero_scalar, variance, zero_scalar)
    std = ct.sqrt(variance)
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))
    ct.store(sub_ptr, index=(row, 0), tile=centered)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    numerator = weight_2d * centered
    denom = std + DENOM_EPS
    normalized = numerator / denom
    affine = normalized + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _inductor_random_for_eager_check(shape, seed, *, device):
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")

    numel = 1
    for dim in shape:
        numel *= int(dim)
    advance = (numel + 131071) // 131072
    state = torch.cuda.get_rng_state(device)
    offset = int.from_bytes(bytes(state[8:16].tolist()), "little")
    if offset >= advance:
        rewound = state.clone()
        rewound_offset = offset - advance
        rewound[8:16] = torch.tensor(
            list(int(rewound_offset).to_bytes(8, "little", signed=False)),
            dtype=state.dtype, device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


@oracle_impl(hardware="B200", point="4205ff34")
def oracle_forward(inputs):
    x, seeds, residual, weight, bias, view_shape, random_shape, out_shape = inputs
    view_shape = _as_shape(view_shape)
    random_shape = _as_shape(random_shape)
    out_shape = _as_shape(out_shape)
    stat_shape = view_shape[:-1] + (1,)
    view_stride = _contiguous_stride(view_shape)
    rows = int(x.shape[0])
    hidden = int(weight.shape[0])
    BLOCK_H = _next_pow2(hidden)

    gt = torch.empty_strided(view_shape, view_stride, device=x.device, dtype=torch.bool)
    add = torch.empty_strided(view_shape, view_stride, device=x.device, dtype=torch.float32)
    sqrt = torch.empty_strided(stat_shape, _contiguous_stride(stat_shape),
                               device=x.device, dtype=torch.float32)
    sub = torch.empty_strided(view_shape, view_stride, device=x.device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=x.device, dtype=torch.bfloat16)

    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=x.device)

    random_2d = random.reshape(rows, hidden)
    residual_2d = residual.reshape(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    sqrt_1d = sqrt.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_mean_var_kernel,
        (
            x, random_2d, residual_2d, weight, bias,
            gt_2d, add_2d, sqrt_1d, sub_2d, out_2d,
            hidden, BLOCK_H, float(hidden),
        ),
    )
    return gt, add, sqrt, sub, out
