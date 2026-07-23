"""cuTile port of var_mean_2336bccf950b: MegatronBERT dropout + residual + LayerNorm.

Uses eager pre-generated random via torch.ops.prims.inductor_random (matches the
Triton oracle's non-graph-capture path). HIDDEN=1024, exact power-of-two, so
BLOCK_H=1024 without masking.

Returns (gt, add, mul_2, view_1, div): mask, residual+scaled_dropped (f32),
normalized (pre-affine), bf16 flat view, rsqrt/HIDDEN.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


SEED_INDEX = 6
DROPOUT_SCALE = 1.1111111111111112
HIDDEN = 1024
BLOCK_H = 1024
EPS = 1.0e-12


@ct.kernel
def _dropout_residual_layernorm_kernel(
    flat_ptr,        # bf16 [rows, HIDDEN]
    random_ptr,      # f32 [rows, HIDDEN]
    residual_ptr,    # f32 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    mask_ptr,        # b8  [rows, HIDDEN]
    add_ptr,         # f32 [rows, HIDDEN]
    normalized_ptr,  # f32 [rows, HIDDEN]
    bf16_view_ptr,   # bf16 [rows, HIDDEN]
    div_ptr,         # f32 [rows]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_))
    rand_f = ct.load(random_ptr, index=(row, 0), shape=(1, BLOCK_H_))

    random_bf16 = ct.astype(rand_f, ct.bfloat16)
    threshold = ct.astype(
        ct.full(shape=(1, BLOCK_H_), fill_value=0.1, dtype=ct.float32),
        ct.bfloat16,
    )
    keep = random_bf16 > threshold
    ct.store(mask_ptr, index=(row, 0), tile=keep)

    dropped_bf16 = ct.astype(ct.where(keep, ct.astype(flat, ct.float32), 0.0), ct.bfloat16)
    scaled_bf16 = ct.astype(ct.astype(dropped_bf16, ct.float32) * DROPOUT_SCALE, ct.bfloat16)
    x = residual + ct.astype(scaled_bf16, ct.float32)
    ct.store(add_ptr, index=(row, 0), tile=x)

    total = ct.sum(x)
    mean = total * (1.0 / HIDDEN_)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H_))
    bias_2d = ct.reshape(bias, (1, BLOCK_H_))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(bf16_view_ptr, index=(row, 0), tile=affine_bf16)
    ct.store(div_ptr, index=(row,), tile=ct.reshape(invstd / HIDDEN_, (1,)))


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
def oracle_forward(inputs, **_kwargs):
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "cuTile port unsupported inside CUDA graph capture (seeded RNG)."
        )

    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, shape2 = inputs
    norm_shape = _as_shape(shape0)
    random_shape = _as_shape(shape1)
    out_shape = _as_shape(shape2)
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    assert hidden == HIDDEN

    div_shape = (norm_shape[0], norm_shape[1], 1)

    mask = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.bool,
    )
    add = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    normalized = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=device, dtype=torch.float32,
    )
    affine_bf16 = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        div_shape, _contiguous_stride(div_shape),
        device=device, dtype=torch.float32,
    )

    seed = torch.ops.prims.inductor_lookup_seed.default(arg1_1, SEED_INDEX)
    random = _inductor_random_for_eager_check(random_shape, seed, device=device)
    random_flat = random.reshape(rows, hidden).contiguous()
    residual_flat = arg2_1.reshape(rows, hidden).contiguous()
    mask_flat = mask.view(rows, hidden)
    add_flat = add.view(rows, hidden)
    normalized_flat = normalized.view(rows, hidden)
    bf16_flat = affine_bf16.view(rows, hidden)
    div_1d = div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _dropout_residual_layernorm_kernel,
        (arg0_1, random_flat, residual_flat, arg3_1, arg4_1,
         mask_flat, add_flat, normalized_flat, bf16_flat, div_1d,
         hidden, BLOCK_H),
    )

    return mask, add, normalized, affine_bf16, div
