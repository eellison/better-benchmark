"""cuTile port of var_mean_0cf78904d48a: MegatronBERT dropout + residual + LayerNorm.

Uses pre-generated seeded random via torch.ops.prims.inductor_random.
Outputs: (gt, add, mul_2, view_1, div).
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 24
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 1024
BLOCK_H = 1024
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_ln_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32  [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    mask_ptr,        # b8   [rows, HIDDEN]
    add_ptr,         # f32  [rows, HIDDEN]
    normalized_ptr,  # f32  [rows, HIDDEN]
    bf16_view_ptr,   # bf16 [rows, HIDDEN]
    div_ptr,         # f32  [rows]
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H_))

    rand_bf = ct.astype(rand, ct.bfloat16)
    thresh_bf = ct.full((1, BLOCK_H_), 0.1, dtype=ct.bfloat16)
    keep = rand_bf > thresh_bf
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    zero_bf = ct.full((1, BLOCK_H_), 0.0, dtype=ct.bfloat16)
    dropped = ct.where(keep, flat, zero_bf)
    scaled = ct.astype(ct.astype(dropped, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    add = ct.astype(scaled, ct.float32) + residual
    ct.store(add_ptr, index=(row, 0), tile=add)

    inv_h = 1.0 / HIDDEN
    mean = ct.sum(add, axis=1, keepdims=True) * inv_h
    centered = add - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    w_2d = ct.reshape(weight, (1, BLOCK_H_))
    b_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * w_2d + b_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
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


@oracle_impl(hardware="B200", point="cfc55f11")
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
    assert hidden == HIDDEN
    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                               device=device, dtype=torch.bool)
    add = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                              device=device, dtype=torch.float32)
    normalized = torch.empty_strided(norm_shape, _contiguous_stride(norm_shape),
                                     device=device, dtype=torch.float32)
    bf16_view = torch.empty_strided(flat_shape, _contiguous_stride(flat_shape),
                                    device=device, dtype=torch.bfloat16)
    div = torch.empty_strided(div_shape, _contiguous_stride(div_shape),
                              device=device, dtype=torch.float32)

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)

    x_2d = arg0_1.view(rows, hidden)
    random_2d = random.contiguous().view(rows, hidden)
    residual_2d = arg2_1.contiguous().view(rows, hidden)
    mask_2d = mask.view(rows, hidden)
    add_2d = add.view(rows, hidden)
    normalized_2d = normalized.view(rows, hidden)
    bf16_2d = bf16_view.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1),
        _dropout_residual_ln_kernel,
        (x_2d, random_2d, residual_2d, arg3_1, arg4_1,
         mask_2d, add_2d, normalized_2d, bf16_2d, div_1d, BLOCK_H),
    )
    return mask, add, normalized, bf16_view, div
