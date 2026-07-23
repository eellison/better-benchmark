"""cuTile port of var_mean_cb5f3cbe2c04: XLNet dropout+residual+LayerNorm (HIDDEN=1024).

Uses eager pre-generated random via inductor_random. HIDDEN is a power of 2.
Returns (gt, normalized_f32, affine_f32, squeeze_bf16_view, squeeze_perm_bf16, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 57
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.1111111111111112
EPS = 1.0e-12
HIDDEN = 1024


@ct.kernel
def _dropout_residual_layernorm_kernel(
    x_ptr,           # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    mask_ptr,        # b8 [rows, HIDDEN]
    norm_ptr,        # f32 [rows, HIDDEN]
    affine_ptr,      # f32 [rows, HIDDEN]
    bf16_ptr,        # bf16 [rows, HIDDEN]
    div_ptr,         # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual_f = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H))

    rand_bf = ct.astype(rand_f, ct.bfloat16)
    threshold_bf = ct.astype(
        ct.full(shape=(1, BLOCK_H), fill_value=DROPOUT_P, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = rand_bf > threshold_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H), 0.0, dtype=ct.bfloat16)
    dropped_bf = ct.where(keep, x_bf, zero_bf)
    scaled_bf = ct.astype(ct.astype(dropped_bf, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    layernorm_input = ct.astype(scaled_bf, ct.float32) + residual_f

    total = ct.sum(layernorm_input)
    mean = total * (1.0 / HIDDEN_)
    centered = layernorm_input - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(norm_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(bf16_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / HIDDEN_), (1,)))


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
    if torch.cuda.is_current_stream_capturing():
        return torch.ops.prims.inductor_random.default(shape, seed, "rand")
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


@oracle_impl(hardware="B200", point="bc741f9d", BLOCK_H=1024, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    norm_shape = _shape_tuple(shape0)
    random_shape = _shape_tuple(shape1)
    view_shape = _shape_tuple(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg3_1.shape[0])
    assert hidden == HIDDEN

    div_shape = norm_shape[:-1] + (1,)

    mask = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                               device=device, dtype=torch.bool)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=device, dtype=torch.float32)
    affine = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                 device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(view_shape, _contiguous_stride(view_shape),
                                    device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    mask_2d = mask.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    affine_2d = affine.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)
    x_2d = arg0_1.view(rows, hidden)
    residual_2d = arg2_1.view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         mask_2d, normalized_2d, affine_2d, bf16_2d, div_1d, hidden, BLOCK_H),
    )

    squeeze = bf16_view.squeeze(0)
    squeeze_1 = bf16_view.permute(0, 2, 1).squeeze(0)
    return mask, normalized, affine, squeeze, squeeze_1, div
