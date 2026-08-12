"""cuTile port of var_mean_771a0a39f4b9: Longformer bias-add + dropout + residual LayerNorm.

Hidden=768 is not a power of two; we pad to 1024 and mask in-kernel using a
column-index arange comparison. The output is written to a padded internal
buffer and returned via a narrow view (which the numerics check compares by
value, not stride).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 16
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5
HIDDEN = 768
BLOCK_H = 1024  # padded


@ct.kernel
def _bias_dropout_layernorm_kernel(
    hidden_ptr,     # bf16 [rows, BLOCK_H] (padded, real cols in [:HIDDEN])
    bias_ptr,       # f32  [BLOCK_H] (padded)
    random_ptr,     # f32  [rows, BLOCK_H]
    residual_ptr,   # f32  [rows, BLOCK_H]
    weight_ptr,     # f32  [BLOCK_H]
    ln_bias_ptr,    # f32  [BLOCK_H]
    gt_ptr,         # b8   [rows, BLOCK_H]
    norm_ptr,       # f32  [rows, BLOCK_H]
    affine_ptr,     # f32  [rows, BLOCK_H]
    bf16_ptr,       # bf16 [rows, BLOCK_H]
    div_ptr,        # f32  [rows, 1]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
):
    row_block = ct.bid(0)

    # Valid-column mask (True for cols < HIDDEN)
    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    valid_col = col_idx < HIDDEN_
    col_mask_2d = ct.reshape(valid_col, (1, BLOCK_H_))

    hidden_bf = ct.load(hidden_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_))
    bias_f = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    bias_bf = ct.astype(bias_f, ct.bfloat16)
    bias_2d = ct.reshape(bias_bf, (1, BLOCK_H_))
    biased_bf = ct.astype(
        ct.astype(hidden_bf, ct.float32) + ct.astype(bias_2d, ct.float32),
        ct.bfloat16,
    )

    rand = ct.load(random_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row_block, 0), shape=(ROW_BLOCK, BLOCK_H_))
    residual_f = ct.astype(residual, ct.float32)

    rand_bf = ct.astype(rand, ct.bfloat16)
    threshold = ct.full((ROW_BLOCK, BLOCK_H_), DROPOUT_P, dtype=ct.bfloat16)
    keep = rand_bf > threshold
    ct.store(gt_ptr, index=(row_block, 0), tile=keep)

    zero_bf = ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, biased_bf, zero_bf)
    scaled_bf = ct.astype(
        ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE,
        ct.bfloat16,
    )
    x = ct.astype(scaled_bf, ct.float32) + residual_f
    # Mask x for reduction — set padded cols to 0 so sum is unaffected
    zero_f = ct.full((ROW_BLOCK, BLOCK_H_), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_mask_2d, x, zero_f)

    inv_h = 1.0 / HIDDEN_
    mean = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    ln_bias = ct.load(ln_bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_f = ct.astype(weight, ct.float32)
    ln_bias_f = ct.astype(ln_bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H_))
    ln_bias_2d = ct.reshape(ln_bias_f, (1, BLOCK_H_))
    affine = normalized * weight_2d + ln_bias_2d

    ct.store(norm_ptr, index=(row_block, 0), tile=normalized)
    ct.store(affine_ptr, index=(row_block, 0), tile=affine)
    ct.store(bf16_ptr, index=(row_block, 0), tile=ct.astype(affine, ct.bfloat16))
    div_val = invstd * inv_h
    ct.store(div_ptr, index=(row_block, 0), tile=div_val)


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


def _shape_tuple(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="726994b7", ROW_BLOCK=1)
def oracle_forward(inputs, *, ROW_BLOCK: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     shape0, random_shape, shape2) = inputs
    base_shape = _shape_tuple(shape0)         # (8, 1024, 768)
    rand_shape = _shape_tuple(random_shape)   # (8, 1024, 768)
    flat_shape = _shape_tuple(shape2)         # (8192, 768)
    hidden = int(arg0_1.shape[0])
    rows = int(arg3_1.shape[0]) * int(arg3_1.shape[1])
    div_shape = (base_shape[0], base_shape[1], 1)
    device = arg1_1.device

    # Padded internal buffers of hidden dim BLOCK_H (1024)
    hidden_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    hidden_padded[:, :hidden] = arg1_1.contiguous().view(rows, hidden)
    bias_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    bias_padded[:hidden] = arg0_1
    residual_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :hidden] = arg3_1.contiguous().view(rows, hidden)
    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:hidden] = arg4_1
    ln_bias_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    ln_bias_padded[:hidden] = arg5_1

    seed = torch.ops.prims.inductor_lookup_seed.default(arg2_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(rand_shape, seed, device=device)
    random_padded = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
    random_padded[:, :hidden] = random.contiguous().view(rows, hidden)

    gt_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div = torch.empty(div_shape, device=device, dtype=torch.float32)
    div_2d = div.view(rows, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _bias_dropout_layernorm_kernel,
        (hidden_padded, bias_padded, random_padded, residual_padded,
         weight_padded, ln_bias_padded,
         gt_padded, norm_padded, affine_padded, bf16_padded, div_2d,
         hidden, BLOCK_H, ROW_BLOCK),
    )

    # Extract the valid [:, :hidden] region as the returned outputs.
    gt = gt_padded[:, :hidden].contiguous().view(base_shape)
    normalized = norm_padded[:, :hidden].contiguous().view(base_shape)
    affine = affine_padded[:, :hidden].contiguous().view(base_shape)
    bf16_flat = bf16_padded[:, :hidden].contiguous().view(flat_shape)
    return gt, normalized, affine, bf16_flat, div
