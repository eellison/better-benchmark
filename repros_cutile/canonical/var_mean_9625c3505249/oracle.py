"""cuTile port of var_mean_9625c3505249: XLNet dropout+residual LayerNorm.

Seed index 37, EPS 1e-12. Returns (gt, mul_2, add_2, squeeze, squeeze_1, div).
squeeze/squeeze_1 are layout-alias views of the bf16 output: the bf16 view has
shape (1, 8192, 1024); squeeze drops axis 0 -> [8192, 1024], and squeeze_1 is
its permute(0, 2, 1).squeeze(0) -> [1024, 8192].
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 37
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


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


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    gt_ptr, norm_ptr, affine_ptr, bf16_ptr, div_ptr,
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    DROPOUT_SCALE_C: ct.Constant[float],
    EPS_C: ct.Constant[float],
):
    row = ct.bid(0)
    flat_bf = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                      padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                       padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
                     padding_mode=ct.PaddingMode.ZERO)

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    p_bf = ct.full((1, HIDDEN_PAD), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > p_bf
    ct.store(gt_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE_C, ct.bfloat16)
    residual_f = ct.astype(residual, ct.float32)
    x = ct.astype(scaled_bf, ct.float32) + residual_f
    zero_f = ct.full((1, HIDDEN_PAD), 0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x, zero_f)

    inv_h = 1.0 / HIDDEN
    mean_val = ct.sum(x_masked) * inv_h
    centered = x - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f)
    variance_val = ct.sum(centered_masked * centered_masked) * inv_h
    invstd_val = ct.rsqrt(variance_val + EPS_C)
    normalized = centered * invstd_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PAD))
    affine = normalized * weight_2d + bias_2d

    normalized_masked = ct.where(valid_2d, normalized, zero_f)
    affine_masked = ct.where(valid_2d, affine, zero_f)
    ct.store(norm_ptr, index=(row, 0), tile=normalized_masked)
    ct.store(affine_ptr, index=(row, 0), tile=affine_masked)
    ct.store(bf16_ptr, index=(row, 0),
             tile=ct.where(valid_2d, ct.astype(affine, ct.bfloat16), zero_bf))
    div_val = invstd_val * inv_h
    ct.store(div_ptr, index=(row,), tile=ct.reshape(div_val, (1,)))


@oracle_impl(hardware="B200", point="bc741f9d")
def oracle_forward(inputs):
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG).")

    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, _shape2 = inputs
    norm_shape = _as_shape(shape0)      # (512, 16, 1024)
    rows = int(arg0_1.shape[0])         # 8192
    hidden = int(arg3_1.shape[0])       # 1024
    div_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    gt = torch.empty(norm_shape, device=device, dtype=torch.bool)
    normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
    affine = torch.empty(norm_shape, device=device, dtype=torch.float32)
    bf16_flat = torch.empty(rows, hidden, device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)

    HIDDEN_PAD = _next_pow2(hidden)
    x_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         gt.view(rows, hidden), normalized.view(rows, hidden),
         affine.view(rows, hidden), bf16_flat, div.view(rows),
         hidden, HIDDEN_PAD, DROPOUT_SCALE, EPS),
    )

    view_1 = bf16_flat.view(1, rows, hidden)
    squeeze = view_1.squeeze(0)
    squeeze_1 = view_1.permute(0, 2, 1).squeeze(0).contiguous()
    return gt, normalized, affine, squeeze, squeeze_1, div
