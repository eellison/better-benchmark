"""cuTile port of mean_var_01d350472d40: BERT dropout-add-dropout-LayerNorm."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX_0 = 4
SEED_INDEX_1 = 5
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
DENOM_EPS = 1.0e-6


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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
        rewound[8:16] = torch.tensor(
            list((offset - advance).to_bytes(8, "little", signed=False)),
            dtype=state.dtype,
            device=state.device,
        )
        torch.cuda.set_rng_state(rewound, device)
        random = torch.ops.prims.inductor_random.default(shape, seed, "rand")
        torch.cuda.set_rng_state(state, device)
        return random
    return torch.ops.prims.inductor_random.default(shape, seed, "rand")


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _dropout_dropout_ln_kernel(
    x_ptr, random0_ptr, random1_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt0_ptr, gt1_ptr, dropped_ptr, sqrt_ptr, sub_ptr, out_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    VAR_DIV: ct.Constant[float],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand0_f32 = ct.load(random0_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                        padding_mode=ct.PaddingMode.ZERO)
    rand1_f32 = ct.load(random1_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                        padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))

    rand0_bf16 = ct.astype(rand0_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN_PAD), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep0 = rand0_bf16 > dropout_p_bf16
    ct.store(gt0_ptr, index=(row, 0), tile=keep0)

    zero_bf16 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.bfloat16)
    first_dropped = ct.where(keep0, x, zero_bf16)
    first_scaled = ct.astype(ct.astype(first_dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add_val = residual + ct.astype(first_scaled, ct.float32)

    keep1 = rand1_f32 > DROPOUT_P
    ct.store(gt1_ptr, index=(row, 0), tile=keep1)
    zero_f32 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.float32)
    norm_input = ct.where(keep1, add_val, zero_f32) * DROPOUT_SCALE
    ct.store(dropped_ptr, index=(row, 0), tile=norm_input)

    reduce_input = ct.where(valid_2d, norm_input, zero_f32)
    row_sum = ct.sum(reduce_input)
    row_sum_sq = ct.sum(reduce_input * reduce_input)
    mean_val = row_sum / HIDDEN_F
    centered = norm_input - mean_val
    variance_sum = row_sum_sq - row_sum * mean_val
    variance = variance_sum / VAR_DIV
    std = ct.sqrt(ct.maximum(variance, 0.0))
    ct.store(sqrt_ptr, index=(row,), tile=ct.reshape(std, (1,)))
    centered_masked = ct.where(valid_2d, centered, zero_f32)
    ct.store(sub_ptr, index=(row, 0), tile=centered_masked)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD))
    affine = (weight_2d * centered) / (std + DENOM_EPS) + bias_2d
    affine_bf16_masked = ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf16)
    ct.store(out_ptr, index=(row, 0), tile=affine_bf16_masked)


@oracle_impl(hardware="B200", point="4205ff34")
def oracle_forward(inputs):
    (x, seeds, residual, weight, bias,
     view_shape, random0_shape, random1_shape, out_shape) = inputs
    view_shape = _as_shape(view_shape)
    random0_shape = _as_shape(random0_shape)
    random1_shape = _as_shape(random1_shape)
    out_shape = _as_shape(out_shape)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    stat_shape = (view_shape[0], view_shape[1], 1)
    device = x.device

    gt0 = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                              device=device, dtype=torch.bool)
    gt1 = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                              device=device, dtype=torch.bool)
    dropped = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                                  device=device, dtype=torch.float32)
    sqrt_out = torch.empty_strided(stat_shape, _contiguous_stride(stat_shape),
                                   device=device, dtype=torch.float32)
    sub = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                              device=device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=device, dtype=torch.bfloat16)

    seed0 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_0)
    seed1 = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX_1)
    random0 = _inductor_random_for_eager_check(random0_shape, seed0, device=device)
    random1 = _inductor_random_for_eager_check(random1_shape, seed1, device=device)

    random0_2d = random0.reshape(rows, hidden)
    random1_2d = random1.reshape(rows, hidden)
    residual_2d = residual.reshape(rows, hidden)
    gt0_2d = gt0.view(rows, hidden)
    gt1_2d = gt1.view(rows, hidden)
    dropped_2d = dropped.view(rows, hidden)
    sqrt_1d = sqrt_out.view(rows)
    sub_2d = sub.view(rows, hidden)
    out_2d = out.view(rows, hidden)

    # variance denominator = HIDDEN - correction; correction=1.0
    var_div = float(hidden) - 1.0
    hidden_pad = _next_pow2(hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_dropout_ln_kernel,
        (x, random0_2d, random1_2d, residual_2d, weight, bias,
         gt0_2d, gt1_2d, dropped_2d, sqrt_1d, sub_2d, out_2d,
         hidden, hidden_pad, float(hidden), var_div),
    )
    return gt0, gt1, dropped, sqrt_out, sub, out
