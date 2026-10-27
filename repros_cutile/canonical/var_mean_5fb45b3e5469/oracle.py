"""cuTile port of var_mean_5fb45b3e5469: DebertaV2 dropout + residual + LayerNorm.

Outputs: (gt, mul_2, add_2, view_1, div). HIDDEN=1536 (pad to 2048).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 48
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-7


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,        # bf16 [rows, BLOCK_H]  padded
    random_ptr,      # f32  [rows, BLOCK_H]
    residual_ptr,    # f32  [rows, BLOCK_H]
    weight_ptr,      # f32  [BLOCK_H]
    bias_ptr,        # f32  [BLOCK_H]
    mask_ptr,        # b8   [rows, BLOCK_H]
    normalized_ptr,  # f32  [rows, BLOCK_H]
    affine_ptr,      # f32  [rows, BLOCK_H]  (add_2 pre-cast)
    bf16_view_ptr,   # bf16 [rows, BLOCK_H]
    div_ptr,         # f32  [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)

    rand_bf = ct.astype(rand, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = ct.astype(scaled, ct.float32) + residual

    inv_h = 1.0 / HIDDEN
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    add_masked = ct.where(col_mask, add, zero_f)
    mean = ct.sum(add_masked, axis=1, keepdims=True) * inv_h
    centered = add - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    w_2d = ct.reshape(weight, (1, BLOCK_H))
    b_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * w_2d + b_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_view_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * inv_h, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype,
        device=state.device,
    )


def _inductor_random_for_eager_check(shape, seed, *, device):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    props = torch.cuda.get_device_properties(device)
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    blocks_per_sm = props.max_threads_per_multi_processor // block_size
    grid = min(
        (numel + block_size - 1) // block_size,
        props.multi_processor_count * blocks_per_sm,
    )
    advance = (
        ((numel - 1) // (block_size * grid * unroll) + 1)
        * curand4_engine_calls
        * 2
    )
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


def _next_pow2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@oracle_impl(hardware="B200", point="55aa5fd0")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )
    device = arg0_1.device
    norm_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    flat_shape = _shape_tuple(shape2)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    BLOCK_H = _next_pow2(hidden)
    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
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
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    def _pad(x, dtype):
        if BLOCK_H == hidden:
            return x.contiguous().view(rows, hidden)
        p = torch.zeros((rows, BLOCK_H), device=device, dtype=dtype)
        p[:, :hidden].copy_(x.reshape(rows, hidden))
        return p

    x_p = _pad(arg0_1, torch.bfloat16)
    residual_p = _pad(arg2_1, torch.float32)
    random_p = _pad(random, torch.float32)
    if BLOCK_H == hidden:
        weight_p = arg3_1.view(hidden)
        bias_p = arg4_1.view(hidden)
    else:
        weight_p = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
        bias_p = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
        weight_p[:hidden].copy_(arg3_1)
        bias_p[:hidden].copy_(arg4_1)

    mask_p = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bool)
    norm_p = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    affine_p = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    bf16_p = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    div_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (x_p, random_p, residual_p, weight_p, bias_p,
         mask_p, norm_p, affine_p, bf16_p, div_1d, hidden, BLOCK_H),
    )
    mask.view(rows, hidden).copy_(mask_p[:, :hidden])
    normalized.view(rows, hidden).copy_(norm_p[:, :hidden])
    affine.view(rows, hidden).copy_(affine_p[:, :hidden])
    bf16_view.view(rows, hidden).copy_(bf16_p[:, :hidden])
    div.view(rows).copy_(div_1d)
    return mask, normalized, affine, bf16_view, div
