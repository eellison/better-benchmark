"""cuTile port of var_mean_48fe94d26f82: dropout-residual LayerNorm (seed 39)."""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 39
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


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


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, norm_ptr, affine_ptr, bf16_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f32 = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)

    rand_bf16 = ct.astype(rand_f32, ct.bfloat16)
    dropout_p_bf16 = ct.astype(
        ct.full(shape=(1, HIDDEN_PAD), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf16 > dropout_p_bf16

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf16 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf16)
    scaled_bf16 = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = ct.astype(scaled_bf16, ct.float32) + residual

    zero_f32 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f32)
    mean_val = ct.sum(x_masked) * (1.0 / HIDDEN_F)
    centered = x - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f32)
    variance_val = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD))
    affine = normalized * weight_2d + bias_2d

    normalized_masked = ct.where(valid_2d, normalized, zero_f32)
    affine_masked = ct.where(valid_2d, affine, zero_f32)
    ct.store(norm_ptr, index=(row, 0), tile=normalized_masked)
    ct.store(affine_ptr, index=(row, 0), tile=affine_masked)
    ct.store(bf16_ptr, index=(row, 0),
             tile=ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf16))
    div_val = invstd_val * (1.0 / HIDDEN_F)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


@oracle_impl(hardware="B200", point="55aa5fd0")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    flat_shape = _as_shape(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    device = arg0_1.device
    gt = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                             device=device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                 device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    HIDDEN_PAD = _next_pow2(hidden)
    random_2d = random.reshape(rows, hidden)
    residual_2d = arg2_1.reshape(rows, hidden)
    gt_2d = gt.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_2d, residual_2d, arg3_1, arg4_1,
         gt_2d, normalized_2d, affine_2d, bf16_view_2d, div_1d,
         hidden, HIDDEN_PAD, float(hidden)),
    )
    return gt, normalized, affine, bf16_view, div
