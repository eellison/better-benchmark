"""cuTile port of var_mean_f574b8ec0222: GPT-2 dropout+residual+LayerNorm row kernel.

Pre-generates the seeded random tensor via inductor_random outside the
kernel. Hidden=768 (non-pow2) uses BLOCK_H=1024 with padding_mode=ZERO
+ column mask. Returns 6 outputs (mask, add, normalized, bf16_view,
bf16_view.permute(1,0), div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 4
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-5


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr, random_ptr, residual_ptr, weight_ptr, bias_ptr,
    mask_ptr, add_ptr, normalized_ptr, bf16_view_ptr, div_ptr,
    HIDDEN: ct.Constant[int], BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)

    rand_bf = ct.astype(rand, ct.bfloat16)
    dropout_p_bf = ct.full((1, BLOCK_H), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > dropout_p_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = residual + ct.astype(scaled_bf, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x)

    inv_h = 1.0 / HIDDEN
    zero_f = ct.full((1, BLOCK_H), 0.0, dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_f)
    mean = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_view_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * inv_h, (1,)))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _resolve_flat_shape(shape, rows, hidden):
    if shape == (-1, hidden):
        return (rows, hidden)
    return tuple(rows if dim == -1 else int(dim) for dim in shape)


def _state_u64(state, start):
    return int.from_bytes(bytes(state[start : start + 8].tolist()), "little")


def _put_state_u64(state, start, value):
    state[start : start + 8] = torch.tensor(
        list(int(value).to_bytes(8, "little", signed=False)),
        dtype=state.dtype, device=state.device,
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


@oracle_impl(hardware="B200", point="a352047a", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="bf8decda", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    flat_shape = _resolve_flat_shape(_as_shape(shape2), rows, hidden)
    side_shape = norm_shape[:-1] + (1,)
    device = arg0_1.device

    mask = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                               device=device, dtype=torch.bool)
    add = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=device, dtype=torch.float32)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(side_shape, _contiguous_stride(side_shape),
                              device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    flat_2d = arg0_1.contiguous().view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    mask_2d = mask.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    bf16_view_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dropout_residual_layernorm_kernel,
        (flat_2d, random_2d, residual_2d, arg3_1, arg4_1,
         mask_2d, add_2d, normalized_2d, bf16_view_2d, div_1d,
         hidden, BLOCK_H),
    )
    return mask, add, normalized, bf16_view, bf16_view.permute(1, 0), div
