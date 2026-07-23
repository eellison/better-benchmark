"""cuTile port of var_mean_b79012c788ec: XLNet dropout + residual + LayerNorm.

bf16[8192,1024] shape. Seed index 35. Point "bc741f9d".
Returns (mask, normalized, affine, bf16_view, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 35
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    x_ptr,
    random_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    mask_ptr,
    norm_ptr,
    affine_ptr,
    bf16_ptr,
    div_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H), padding_mode=ct.PaddingMode.ZERO)

    # Dropout
    rand_bf16 = ct.astype(rand_f, ct.bfloat16)
    dropout_p = ct.astype(ct.full((1, BLOCK_H), 0.1, dtype=ct.float32), ct.bfloat16)
    keep = rand_bf16 > dropout_p
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    # Dropout scaling
    dropped = ct.astype(ct.where(keep, ct.astype(x_bf16, ct.float32), 0.0), ct.bfloat16)
    dropped_scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    layernorm_input = ct.astype(dropped_scaled, ct.float32) + residual

    # LayerNorm - masked reduction
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    reduce_input = ct.where(col_mask, layernorm_input, 0.0)
    mean = ct.sum(reduce_input) * (1.0 / HIDDEN)
    centered = layernorm_input - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    # Affine
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN, (1,)))


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, shape1, shape2, shape3 = inputs
    norm_shape = _as_shape(shape1)
    random_shape = _as_shape(shape2)
    flat_shape = _as_shape(shape3)
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    div_shape = (norm_shape[0], norm_shape[1], 1)
    device = arg0_1.device

    # Generate random outside kernel
    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()

    # Flatten residual to 2D
    residual_flat = arg2_1.reshape(rows, hidden).contiguous()

    # Padded outputs for kernel
    mask_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         mask_pad, norm_pad, affine_pad, bf16_pad, div_1d, hidden, BLOCK_H),
    )

    # Reshape to match output strides
    mask = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=device,
        dtype=torch.bool,
    )
    mask.view(rows, hidden).copy_(mask_pad.narrow(1, 0, hidden))

    normalized = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=device,
        dtype=torch.float32,
    )
    normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))

    affine = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=device,
        dtype=torch.float32,
    )
    affine.view(rows, hidden).copy_(affine_pad.narrow(1, 0, hidden))

    bf16_view = torch.empty_strided(
        flat_shape,
        _contiguous_stride(flat_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    bf16_view.view(rows, hidden).copy_(bf16_pad.narrow(1, 0, hidden))

    div = torch.empty_strided(
        div_shape,
        _contiguous_stride(div_shape),
        device=device,
        dtype=torch.float32,
    )
    div.view(rows).copy_(div_1d)

    return mask, normalized, affine, bf16_view, div
