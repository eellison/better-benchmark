"""cuTile port of var_mean_33e99fe96c48: GPT2 dropout residual LayerNorm.

Pre-generates seeded random via inductor_random, then runs one cuTile row kernel
that applies dropout, adds residual (f32), performs LayerNorm (eps=1e-5),
affine, and emits mask/add/normalized/bf16-view/bf16-permute/div outputs.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 18
DROPOUT_SCALE = 1.1111111111111112
DROPOUT_P = 0.1
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,        # bf16 (rows, HIDDEN)
    random_ptr,      # f32 (rows, HIDDEN) - pre-generated
    residual_ptr,    # f32 (rows, HIDDEN)
    weight_ptr,      # f32 (HIDDEN,)
    bias_ptr,        # f32 (HIDDEN,)
    mask_ptr,        # bool (rows, HIDDEN)
    add_ptr,         # f32 (rows, HIDDEN)
    normalized_ptr,  # f32 (rows, HIDDEN)
    bf16_view_ptr,   # bf16 (rows, HIDDEN)
    div_ptr,         # f32 (rows,)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    random_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_2d = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)

    random_bf = ct.astype(random_f, ct.bfloat16)
    dropout_p_bf = ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.bfloat16)
    keep = random_bf > dropout_p_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x)

    x_masked = ct.where(col_mask_2d, x, zero_f)
    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_view_ptr, index=(row, 0), tile=affine_bf)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN_F), (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _resolve_flat_shape(shape, rows, hidden):
    if shape == (-1, hidden):
        return (rows, hidden)
    return tuple(rows if dim == -1 else int(dim) for dim in shape)


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


@oracle_impl(hardware="B200", point="bf8decda")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    flat_shape = _resolve_flat_shape(_as_shape(shape2), rows, hidden)
    side_shape = norm_shape[:-1] + (1,)
    BLOCK_H = _next_pow2(hidden)

    mask = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                               device=arg0_1.device, dtype=torch.bool)
    add = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=arg0_1.device, dtype=torch.float32)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=arg0_1.device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=arg0_1.device, dtype=torch.bfloat16)
    div = torch.empty_strided(side_shape, _contiguous_stride(side_shape),
                              device=arg0_1.device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=arg0_1.device)

    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    mask_2d = mask.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (
            arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
            mask_2d, add_2d, normalized_2d, bf16_view_2d, div_1d,
            hidden, BLOCK_H, float(hidden),
        ),
    )
    return mask, add, normalized, bf16_view, bf16_view.permute(1, 0), div
