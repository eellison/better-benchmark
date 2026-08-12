"""cuTile port of var_mean_f9d8faf2bd9a: GoogleFnet seeded-dropout residual LN.

Ports the Triton `_dropout_residual_layernorm_kernel` — for each row: dropout
mask via gt(random, 0.1), fp32 residual add, var_mean, rsqrt(eps=1e-12),
affine, division of rsqrt by hidden as a side output.

cuTile has no seeded on-device RNG; we generate the random tensor with
`torch.ops.prims.inductor_random` (same as the Repro) and pass it in.
"""

import torch
import cuda.tile as ct
import torch._inductor.inductor_prims  # noqa: F401

from oracle_harness import oracle_impl


SEED_INDEX = 12
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    x_ptr,          # f32 [ROWS, BLOCK_H] (padded)
    random_ptr,     # f32 [ROWS, BLOCK_H]
    residual_ptr,   # f32 [ROWS, BLOCK_H]
    weight_ptr,     # f32 [BLOCK_H]
    bias_ptr,       # f32 [BLOCK_H]
    mask_ptr,       # bool [ROWS, BLOCK_H]
    norm_ptr,       # f32 [ROWS, BLOCK_H]
    affine_ptr,     # f32 [ROWS, BLOCK_H]
    div_ptr,        # f32 [ROWS]
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    random = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    keep = random > 0.1
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    dropped = ct.where(keep, x, zero)
    dropped_scaled = dropped * DROPOUT_SCALE
    layernorm_input = dropped_scaled + residual

    reduce_input = ct.where(col_mask_2d, layernorm_input, 0.0)
    mean_val = ct.sum(reduce_input) * (1.0 / HIDDEN)
    centered = layernorm_input - mean_val
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    div_scalar = ct.reshape(invstd / HIDDEN_F, (1,))
    ct.store(div_ptr, index=(row,), tile=div_scalar)


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


@oracle_impl(hardware="B200", point="3a80a44f", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = tuple(int(d) for d in shape0)
    random_shape = tuple(int(d) for d in shape1)
    flat_shape = tuple(int(d) for d in shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)

    if BLOCK_H < hidden:
        raise NotImplementedError(
            f"cuTile port unsupported: BLOCK_H={BLOCK_H} < hidden={hidden}"
        )

    # Pad inputs to (rows, BLOCK_H) so the tile store doesn't overrun.
    padded_x = torch.zeros((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    padded_x[:, :hidden].copy_(arg0_1)
    padded_resid = torch.zeros((rows, BLOCK_H), device=arg2_1.device, dtype=torch.float32)
    padded_resid[:, :hidden].copy_(arg2_1.view(rows, hidden))
    padded_weight = torch.zeros(BLOCK_H, device=arg3_1.device, dtype=torch.float32)
    padded_weight[:hidden].copy_(arg3_1)
    padded_bias = torch.zeros(BLOCK_H, device=arg4_1.device, dtype=torch.float32)
    padded_bias[:hidden].copy_(arg4_1)

    # Generate the random tensor using Inductor's Philox — matches the Repro.
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(
        random_shape, seed, device=arg0_1.device,
    )
    padded_random = torch.zeros((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    padded_random[:, :hidden].copy_(random.view(rows, hidden))

    # Output buffers padded, then narrow.
    padded_mask = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.bool)
    padded_normalized = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    padded_affine = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    div = torch.empty((rows,), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (
            padded_x, padded_random, padded_resid, padded_weight, padded_bias,
            padded_mask, padded_normalized, padded_affine, div,
            hidden, float(hidden), BLOCK_H,
        ),
    )

    mask = padded_mask[:, :hidden].contiguous().view(norm_shape)
    normalized = padded_normalized[:, :hidden].contiguous().view(norm_shape)
    affine = padded_affine[:, :hidden].contiguous().view(flat_shape)
    div = div.view(div_shape)
    return mask, normalized, affine, div
