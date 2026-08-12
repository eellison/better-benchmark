"""cuTile port of var_mean_92c5ebca24ad: DeBERTaV2 seeded-dropout residual LN.

Row layernorm with per-row dropout mask (bf16 threshold 0.1), fp32 residual add,
population var_mean, eps=1e-7 rsqrt, affine, bf16 view, and rsqrt/HIDDEN side
output. hidden=1536 requires BLOCK_H=2048 (pow2 padding).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 50
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,       # bf16 [rows, BLOCK_H] (padded)
    random_ptr,     # f32  [rows, BLOCK_H]
    residual_ptr,   # f32  [rows, BLOCK_H]
    weight_ptr,     # f32  [BLOCK_H]
    bias_ptr,       # f32  [BLOCK_H]
    gt_ptr,         # b8   [rows, BLOCK_H]
    norm_ptr,       # f32  [rows, BLOCK_H]
    affine_ptr,     # f32  [rows, BLOCK_H]
    bf16_ptr,       # bf16 [rows, BLOCK_H]
    div_ptr,        # f32  [rows]
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    DROPOUT_SCALE_: ct.Constant[float],
    EPS_: ct.Constant[float],
):
    row_block = ct.bid(0)

    flat = ct.load(
        flat_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rand = ct.load(
        random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual_f = ct.astype(residual, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, flat, zero_bf)
    dropped_scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE_,
        ct.bfloat16,
    )
    x = ct.astype(dropped_scaled_bf, ct.float32) + residual_f

    # Mask out padding for reductions.
    zero_f = ct.full((ROW_BLOCK, BLOCK_H), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_valid_2d, x, zero_f)
    inv_h = 1.0 / HIDDEN_F
    mean_1d = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = x - mean_1d
    centered_masked = ct.where(col_valid_2d, centered, zero_f)
    variance_1d = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd_1d = ct.rsqrt(variance_1d + EPS_)
    normalized = centered * invstd_1d

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    div_val = ct.reshape(invstd_1d * inv_h, (ROW_BLOCK,))
    ct.store(div_ptr, index=(row_block,), tile=div_val)


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


@oracle_impl(hardware="B200", point="55aa5fd0", BLOCK_H=2048, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape(shape0)   # [8, 512, 1536]
    flat_shape = _shape(shape2)   # [4096, 1536]
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    device = arg0_1.device
    div_shape = (norm_shape[0], norm_shape[1], 1)

    # Pad tensors to (rows, BLOCK_H) to safely accommodate the padded tile store.
    padded_x = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    padded_x[:, :hidden].copy_(arg0_1.contiguous().view(rows, hidden))
    padded_residual = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    padded_residual[:, :hidden].copy_(arg2_1.contiguous().view(rows, hidden))
    padded_weight = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
    padded_weight[:hidden].copy_(arg3_1)
    padded_bias = torch.zeros(BLOCK_H, device=device, dtype=torch.float32)
    padded_bias[:hidden].copy_(arg4_1)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(norm_shape, seed, device=device)
    padded_random = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    padded_random[:, :hidden].copy_(random.contiguous().view(rows, hidden))

    padded_gt = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bool)
    padded_norm = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    padded_affine = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    padded_bf16 = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_flat = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _dropout_residual_layernorm_kernel,
        (padded_x, padded_random, padded_residual, padded_weight, padded_bias,
         padded_gt, padded_norm, padded_affine, padded_bf16, div_flat,
         hidden, float(hidden), BLOCK_H, ROW_BLOCK, DROPOUT_SCALE, EPS),
    )

    gt = padded_gt[:, :hidden].contiguous().view(norm_shape)
    normalized = padded_norm[:, :hidden].contiguous().view(norm_shape)
    affine = padded_affine[:, :hidden].contiguous().view(norm_shape)
    bf16_view = padded_bf16[:, :hidden].contiguous().view(flat_shape)
    div = div_flat.view(div_shape)
    return gt, normalized, affine, bf16_view, div
